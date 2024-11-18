#!/usr/bin python3

import numpy as np

import casadi as ca

import copy
import pdb
from datetime import datetime
import os
import shutil
import pathlib
import itertools
from typing import List

from mpclab_common.models.belief_models import CasadiDecoupledMultiAgentBeliefModel
from mpclab_common.pytypes import VehicleState, VehiclePrediction

from mpclab_controllers.abstract_controller import AbstractController
from mpclab_controllers.utils.controllerTypes import iLQGParams

class iLQG_MA_RHC(AbstractController):
    def __init__(self, belief_dynamics: CasadiDecoupledMultiAgentBeliefModel, costs, params=iLQGParams()):
        self.belief_dynamics = belief_dynamics
        self.M = self.belief_dynamics.M # Number of agents

        self.N = params.N
        self.max_iters = params.max_iters
        self.verbose = params.verbose
        self.code_gen = params.code_gen
        self.jit = params.jit
        self.opt_flag = params.opt_flag
        self.solver_name = params.solver_name
        if params.solver_dir is not None:
            self.solver_dir = os.path.join(params.solver_dir, self.solver_name)

        self.line_search = params.line_search

        if not params.enable_jacobians:
            jac_opts = dict(enable_fd=False, enable_jacobian=False, enable_forward=False, enable_reverse=False)
        else:
            jac_opts = dict()

        if self.code_gen:
            if self.jit:
                self.options = dict(jit=True, jit_name=self.solver_name, compiler='shell', jit_options=dict(compiler='gcc', flags=['-%s' % self.opt_flag], verbose=self.verbose), **jac_opts)
            else:
                self.options = dict(jit=False, **jac_opts)
                self.c_file_name = self.solver_name + '.c'
                self.so_file_name = self.solver_name + '.so'
                if params.solver_dir is not None:
                    self.solver_dir = pathlib.Path(params.solver_dir).expanduser().joinpath(self.solver_name)
        else:
            self.options = dict(jit=False, **jac_opts)

        self.debug = params.debug
        if self.debug: self.verbose = True

        # The costs should be a list of dicts of casadi functions with keys 'stage' and 'terminal'
        if len(costs) != self.M:
            raise ValueError('Number of agents: %i, but only %i cost functions were provided' % (self.M, len(costs)))
        self.costs_sym = costs

        self.trajectory_cost_prev = np.inf

        self.belief_input_predictions = [VehiclePrediction() for _ in range(self.M)]

        self.n_b = self.belief_dynamics.n_b
        self.n_u = self.belief_dynamics.n_u
        self.n_q = self.belief_dynamics.n_q

        if not isinstance(params.tol, np.ndarray):
            raise ValueError('Absolute convergence tolerance must be a numpy array')
        elif len(params.tol) != 1 and len(params.tol) != self.M:
            raise ValueError('Absolute convergence tolerance should be an array of length 1 or %i' % self.M)
        if len(params.tol) == 1:
            self.tol = params.tol*np.ones(self.M)
        else:
            self.tol = params.tol
        if not isinstance(params.rel_tol, np.ndarray):
            raise ValueError('Relative convergence tolerance must be a numpy array')
        elif len(params.rel_tol) != 1 and len(params.rel_tol) != self.M:
            raise ValueError('Relative convergence tolerance should be an array of length 1 or %i' % self.M)
        if len(params.rel_tol) == 1:
            self.rel_tol = params.rel_tol*np.ones(self.M)
        else:
            self.rel_tol = params.rel_tol

        self.control_reg_init = params.control_reg
        self.belief_reg_init = params.belief_reg

        # backtracking line search coefficients
        alpha = [np.kron(np.power(10, np.linspace(0, -2, 3).reshape((-1,1))),np.ones(self.belief_dynamics.dynamics_models[i].n_u)) for i in range(self.M)]
        self.alpha = list(itertools.product(*alpha))
        self.weights = np.array([0.7, 0.3])

        self.b_nom = np.zeros((self.N+1, self.n_b))
        self.u_nom = np.zeros((self.N, self.n_u))

        self.u_prev = np.zeros(self.n_u)

        # Symbolic backward pass
        self.back_sym = [None for _ in range(self.N+1)]
        self.forw_sym = [None for _ in range(self.N+1)]
        self.ilqg_sym = None
        if self.debug:
            self.debug_fns = dict()

        self.f_dynamics = None

        if params.solver_dir:
            self._load_solver()
        else:
            self._build_solver()

        self.initialized = True

    def initialize(self, u_init=None):
        if u_init is not None:
            if u_init.ndim == 1:
                self.u_nom = np.tile(u_init, (self.N, 1))
                self.u_prev = u_init
            elif u_init.ndim == 2:
                self.u_nom = u_init
                self.u_prev = u_init[0]
            else:
                raise ValueError('Input initialization has too many dimensions')

        self.initialized = True

    def step(self, belief_states: List[VehicleState], env_state=None):
        info = self.solve(belief_states)

        self.belief_dynamics.bu2state(belief_states, None, self.u_nom[0])
        self.belief_dynamics.bu2prediction(self.belief_input_predictions, self.b_nom, self.u_nom)
        for b in self.belief_input_predictions:
            b.t = belief_states[0].t

        self.u_prev = self.u_nom[0]

        return

    def solve(self, belief_states: List[VehicleState]):
        if self.verbose:
            print('Solving for %s' % self.solver_name)
        init_start = datetime.now()
        converged = False

        # Rollout nominal belief trajectory using input sequence from last solve
        self.b_nom[0] = self.belief_dynamics.state2b(belief_states)
        self.u_nom = np.vstack((self.u_nom[1:], self.u_nom[-1].reshape((1,-1)), self.u_prev.reshape((1,-1))))

        cost_nom = np.zeros(self.M)
        for k in range(self.N):
            # Compute stage cost
            args_cost = [self.b_nom[k], self.u_nom[k], self.u_nom[k-1]]
            for i in range(self.M):
                cost_nom[i] += self.costs_sym[i]['stage'](*args_cost)

            # Update belief
            args = [self.b_nom[k], self.u_nom[k]]
            self.b_nom[k+1] = self.belief_dynamics.f_g(*args).toarray().squeeze()

        # Compute terminal cost
        for i in range(self.M):
            cost_nom[i] += self.costs_sym[i]['term'](self.b_nom[-1])
        cost_prev = copy.copy(cost_nom)

        b_nom = copy.copy(self.b_nom.T)
        u_nom = copy.copy(self.u_nom.T)

        self.control_reg = copy.copy(self.control_reg_init)
        self.belief_reg = copy.copy(self.belief_reg_init)
        init_time = (datetime.now()-init_start).total_seconds()

        # Do iLQG iterations
        converged = False
        for i in range(self.max_iters):
            # Do backward and forward pass
            iter_start_time = datetime.now()
            ilqg_out = self.ilqg_sym(*np.hsplit(b_nom, b_nom.shape[1]), *np.hsplit(u_nom, u_nom.shape[1]), self.belief_reg, self.control_reg)

            if self.line_search:
                avg_cost = np.inf
                for j in range(len(self.alpha)):
                    b_j, u_j, cost_j = ilqg_out[j*3:(j+1)*3]
                    cost_j = cost_j.toarray().squeeze()
                    if np.any(np.isnan(cost_j)):
                        print(cost_j)
                        pdb.set_trace()
                    if cost_j.dot(self.weights) < avg_cost:
                        avg_cost = cost_j.dot(self.weights)
                        b_new, u_new = b_j, u_j
                        cost = cost_j
            else:
                b_new, u_new, cost = ilqg_out
                cost = cost.toarray().squeeze()
                if np.any(np.isnan(cost)):
                    print(cost)
                    pdb.set_trace()

            back_forw_time = (datetime.now()-iter_start_time).total_seconds()

            if np.all(np.abs(cost - cost_nom) <= self.tol):
                converged = True
                if self.verbose:
                    print('it: %i | %g s | nom cost: %s | curr cost: %s'  % (i, back_forw_time, str(cost_nom), str(cost)))
                break

            if np.all(cost <= cost_nom):
                s = 'nom cost: %s | curr cost: %s | prev cost: %s | improvement, update accepted, lowering reg' % (str(cost_nom), str(cost), str(cost_prev))
                self.control_reg = self.control_reg/3
                self.belief_reg = self.belief_reg/3

                b_nom = copy.copy(b_new.toarray())
                u_nom = np.hstack((u_new.toarray(), self.u_prev.reshape((-1,1))))
                cost_nom = copy.copy(cost)
            else:
                if np.all(np.abs(cost - cost_prev) <= self.rel_tol):
                    s = 'nom cost: %s | curr cost: %s | prev cost: %s | no improvement but cost sequence converged, update accepted, not changing reg' % (str(cost_nom), str(cost), str(cost_prev))
                    b_nom = copy.copy(b_new.toarray())
                    u_nom = np.hstack((u_new.toarray(), self.u_prev.reshape((-1,1))))
                    cost_nom = copy.copy(cost)
                else:
                    s = 'nom cost: %s | curr cost: %s | prev cost: %s | no improvement, update rejected, increasing reg' % (str(cost_nom), str(cost), str(cost_prev))
                    self.control_reg = self.control_reg*2
                    self.belief_reg = self.belief_reg*2
                # s = 'nom cost: %s | curr cost: %s | prev cost: %s | no improvement, update rejected, increasing reg' % (str(cost_nom), str(cost), str(cost_prev))
                # self.control_reg = self.control_reg*2
                # self.belief_reg = self.belief_reg*2

            cost_prev = copy.copy(cost)
            iter_time = (datetime.now()-iter_start_time).total_seconds()
            if self.verbose:
                print('it: %i | %g s | total: %g | %s' % (i, back_forw_time, iter_time, s))

        self.b_nom = copy.copy(b_new.toarray().T)
        self.u_nom = copy.copy(u_new.toarray().T)
        self.trajectory_cost_prev = copy.copy(cost)

        ilqg_time = (datetime.now()-init_start).total_seconds()

        if converged:
            return_status = 'Converged in %i iterations | final cost: %s | init time: %g | total time: %g' % (i+1, str(cost_nom), init_time, ilqg_time)
        else:
            return_status = 'Max its reached, solution did not converge | final cost: %s | init time: %g | total time: %g' % (str(cost_nom), init_time, ilqg_time)
        if self.verbose:
            print(return_status)
            print('==========================')

        info = {'success': converged, 'return_status': return_status, 'solve_time': ilqg_time, 'info': None, 'output': None}

        return info

    def _build_solver(self):
        # =================================
        # Create cost computation functions
        # =================================
        # Placeholder symbolic variables for nominal sequence
        b_ph = ca.MX.sym('b_ph', self.n_b) # State
        u_ph = ca.MX.sym('u_ph', self.n_u) # Input
        um_ph = ca.MX.sym('um_ph', self.n_u) # Previous input

        self.sym_stage_costs, self.sym_term_costs = [], []
        for i in range(self.M):
            C = self.costs_sym[i]['stage'](b_ph, u_ph, um_ph)
            Dbb_C, _ = ca.hessian(C, b_ph)
            Db_C = ca.jacobian(C, b_ph)
            Duu_C, _ = ca.hessian(C, u_ph)
            Du_C = ca.jacobian(C, u_ph)
            Dub_C = ca.jacobian(Du_C, b_ph)

            args_in = [b_ph, u_ph, um_ph]
            args_out = [C, Db_C, Du_C, ca.tril(Dbb_C), ca.tril(Duu_C), Dub_C]
            self.sym_stage_costs.append(ca.Function('stage_cost_agent_%i' % i, args_in, args_out))

            V = self.costs_sym[i]['term'](b_ph)
            Dbb_V, _ = ca.hessian(V, b_ph)
            Db_V = ca.jacobian(V, b_ph)

            args_in = [b_ph]
            args_out = [V, Db_V, ca.tril(Dbb_V)]
            self.sym_term_costs.append(ca.Function('term_cost_agent_%i' % i, args_in, args_out))

        # ==============================================
        # Create state-action value computation function
        # ==============================================
        # # Stage cost placeholders
        C_ph = ca.MX.sym('C_ph', 1)
        Db_C_ph = ca.MX.sym('Db_C_ph', self.n_b)
        Du_C_ph = ca.MX.sym('Du_C_ph', self.n_u)
        Dbb_C_ph = ca.MX.sym('Dbb_C_ph', ca.Sparsity.lower(self.n_b))
        Duu_C_ph = ca.MX.sym('Duu_C_ph', ca.Sparsity.lower(self.n_u))
        Dub_C_ph = ca.MX.sym('Dub_C_ph', self.n_u, self.n_b)

        # Dynamics placeholders
        Db_g_ph = ca.MX.sym('Db_g_ph', self.n_b, self.n_b)
        Du_g_ph = ca.MX.sym('Du_g_ph', self.n_b, self.n_u)
        W_ph = [ca.MX.sym('W_%i_ph' % j, self.n_b) for j in range(self.n_q)]
        Db_W_ph = [ca.MX.sym('Db_W_%i_ph' % j, self.n_b, self.n_b) for j in range(self.n_q)]
        Du_W_ph = [ca.MX.sym('Du_W_%i_ph' % j, self.n_b, self.n_u) for j in range(self.n_q)]

        # Value function at next time step placeholders
        Vp_ph = ca.MX.sym('Vp_ph', 1)
        Db_Vp_ph = ca.MX.sym('Db_Vp_ph', self.n_b)
        Dbb_Vp_ph = ca.MX.sym('Dbb_Vp_ph', ca.Sparsity.lower(self.n_b))

        # State regularization placeholder
        mu_b_ph = ca.MX.sym('mu_b_ph', 1)

        # Constant term of quadratic approx of Q
        Q = C_ph + Vp_ph

        # Linear term of quadratic approx of Q
        Db_Q = Db_C_ph + Db_g_ph.T @ Db_Vp_ph
        Du_Q = Du_C_ph + Du_g_ph.T @ Db_Vp_ph

        # Quadratic term of quadratic approx of Q
        mu_b_tmp = ca.SX.sym('mu_b_tmp', 1)
        Dbb_Vp_tmp = ca.SX.sym('Dbb_Vp_tmp', ca.Sparsity.lower(self.n_b))
        Db_W_tmp = ca.SX.sym('Db_W_tmp', self.n_b, self.n_b)
        Du_W_tmp = ca.SX.sym('Du_W_tmp', self.n_b, self.n_u)
        W_tmp = ca.SX.sym('W_tmp', self.n_b)
        Dbb_Vp_reg_tmp = Dbb_Vp_tmp + mu_b_tmp*ca.DM.eye(self.n_b)
        Q_j = ca.bilin(ca.tril2symm(Dbb_Vp_tmp), W_tmp, W_tmp)/2
        Db_Q_j = ca.sparsify(Db_W_tmp.T @ ca.tril2symm(Dbb_Vp_tmp) @ W_tmp)
        Du_Q_j = ca.sparsify(Du_W_tmp.T @ ca.tril2symm(Dbb_Vp_tmp) @ W_tmp)
        Dbb_Q_j = ca.tril(Db_W_tmp.T @ ca.tril2symm(Dbb_Vp_reg_tmp) @ Db_W_tmp)
        Duu_Q_j = ca.tril(Du_W_tmp.T @ ca.tril2symm(Dbb_Vp_reg_tmp) @ Du_W_tmp)
        Dub_Q_j = Du_W_tmp.T @ ca.tril2symm(Dbb_Vp_reg_tmp) @ Db_W_tmp
        args_in = [Dbb_Vp_tmp, W_tmp, Db_W_tmp, Du_W_tmp, mu_b_tmp]
        args_out = [Q_j, Db_Q_j, Du_Q_j, Dbb_Q_j, Duu_Q_j, Dub_Q_j]
        Q_eval_j = ca.Function('Q_eval_j', args_in, args_out)

        Dbb_Vp_reg = Dbb_Vp_ph + mu_b_ph*ca.DM.eye(self.n_b)
        Dbb_Q = Dbb_C_ph + ca.tril(Db_g_ph.T @ ca.tril2symm(Dbb_Vp_reg) @ Db_g_ph)
        Duu_Q = Duu_C_ph + ca.tril(Du_g_ph.T @ ca.tril2symm(Dbb_Vp_reg) @ Du_g_ph)
        Dub_Q = Dub_C_ph + Du_g_ph.T @ ca.tril2symm(Dbb_Vp_reg) @ Db_g_ph
        for j in range(self.n_q):
            Q_j, Db_Q_j, Du_Q_j, Dbb_Q_j, Duu_Q_j, Dub_Q_j = Q_eval_j(Dbb_Vp_ph, W_ph[j], Db_W_ph[j], Du_W_ph[j], mu_b_ph)
            Q += Q_j; Db_Q += Db_Q_j; Du_Q += Du_Q_j; Dbb_Q += Dbb_Q_j; Duu_Q += Duu_Q_j; Dub_Q += Dub_Q_j

        # Stage cost placeholders
        # C_ph = ca.SX.sym('C_ph', 1)
        # Db_C_ph = ca.SX.sym('Db_C_ph', self.n_b)
        # Du_C_ph = ca.SX.sym('Du_C_ph', self.n_u)
        # Dbb_C_ph = ca.SX.sym('Dbb_C_ph', ca.Sparsity.lower(self.n_b))
        # Duu_C_ph = ca.SX.sym('Duu_C_ph', ca.Sparsity.lower(self.n_u))
        # Dub_C_ph = ca.SX.sym('Dub_C_ph', self.n_u, self.n_b)

        # # Dynamics placeholders
        # Db_g_ph = ca.SX.sym('Db_g_ph', self.n_b, self.n_b)
        # Du_g_ph = ca.SX.sym('Du_g_ph', self.n_b, self.n_u)
        # W_ph = [ca.SX.sym('W_%i_ph' % j, self.n_b) for j in range(self.n_q)]
        # Db_W_ph = [ca.SX.sym('Db_W_%i_ph' % j, self.n_b, self.n_b) for j in range(self.n_q)]
        # Du_W_ph = [ca.SX.sym('Du_W_%i_ph' % j, self.n_b, self.n_u) for j in range(self.n_q)]

        # # Value function at next time step placeholders
        # Vp_ph = ca.SX.sym('Vp_ph', 1)
        # Db_Vp_ph = ca.SX.sym('Db_Vp_ph', self.n_b)
        # Dbb_Vp_ph = ca.SX.sym('Dbb_Vp_ph', ca.Sparsity.lower(self.n_b))

        # # State regularization placeholder
        # mu_b_ph = ca.SX.sym('mu_b_ph', 1)

        # # Constant term of quadratic approx of Q
        # Q = C_ph + Vp_ph

        # # Linear term of quadratic approx of Q
        # Db_Q = Db_C_ph + Db_g_ph.T @ Db_Vp_ph
        # Du_Q = Du_C_ph + Du_g_ph.T @ Db_Vp_ph

        # # Quadratic term of quadratic approx of Q
        # Dbb_Vp_reg = Dbb_Vp_ph + mu_b_ph*ca.DM.eye(self.n_b)
        # Dbb_Q = Dbb_C_ph + ca.tril(Db_g_ph.T @ ca.tril2symm(Dbb_Vp_reg) @ Db_g_ph)
        # Duu_Q = Duu_C_ph + ca.tril(Du_g_ph.T @ ca.tril2symm(Dbb_Vp_reg) @ Du_g_ph)
        # Dub_Q = Dub_C_ph + Du_g_ph.T @ ca.tril2symm(Dbb_Vp_reg) @ Db_g_ph

        # for j in range(self.n_q):
        #     Q += ca.bilin(ca.tril2symm(Dbb_Vp_reg), W_ph[j], W_ph[j])/2
        #     Db_Q += ca.sparsify(Db_W_ph[j].T @ ca.tril2symm(Dbb_Vp_ph) @ W_ph[j])
        #     Du_Q += ca.sparsify(Du_W_ph[j].T @ ca.tril2symm(Dbb_Vp_ph) @ W_ph[j])
        #     Dbb_Q += ca.tril(Db_W_ph[j].T @ ca.tril2symm(Dbb_Vp_reg) @ Db_W_ph[j])
        #     Duu_Q += ca.tril(Du_W_ph[j].T @ ca.tril2symm(Dbb_Vp_reg) @ Du_W_ph[j])
        #     Dub_Q += Du_W_ph[j].T @ ca.tril2symm(Dbb_Vp_reg) @ Db_W_ph[j]
               
        args_in = [C_ph, Db_C_ph, Du_C_ph, Dbb_C_ph, Duu_C_ph, Dub_C_ph] \
                    + [Vp_ph, Db_Vp_ph, Dbb_Vp_ph] \
                    + [Db_g_ph, Du_g_ph] \
                    + W_ph + Db_W_ph + Du_W_ph \
                    + [mu_b_ph]
        args_out = [Q, Db_Q, Du_Q, Dbb_Q, Duu_Q, Dub_Q]
        self.sym_state_action_value = ca.Function('state_action_value', args_in, args_out)

        # ==============================================
        # Create feedback policy computation function
        # ==============================================
        # Stage-action value function placeholders
        Q_ph = ca.SX.sym('Q_ph', 1)
        Db_Q_ph = ca.SX.sym('Db_Q_ph', self.n_b)
        Du_Q_ph = ca.SX.sym('Du_Q_ph', self.n_u)
        Dbb_Q_ph = ca.SX.sym('Dbb_Q_ph', ca.Sparsity.lower(self.n_b))
        Duu_Q_ph = ca.SX.sym('Duu_Q_ph', ca.Sparsity.lower(self.n_u))
        Dub_Q_ph = ca.SX.sym('Dub_Q_ph', self.n_u, self.n_b)

        Du_Q_hat_ph = ca.SX.sym('Du_Q_hat_ph', self.n_u)
        Duu_Q_hat_ph = ca.SX.sym('Duu_Q_hat_ph', self.n_u, self.n_u)
        Dub_Q_hat_ph = ca.SX.sym('Dub_Q_hat_ph', self.n_u, self.n_b)

        # Input regularization placeholder
        mu_u_ph = ca.SX.sym('mu_u_ph', 1)

        L = Duu_Q_hat_ph + mu_u_ph*ca.DM.eye(self.n_u)
        # K_fb = -ca.solve(L, Dub_Q_hat_ph)
        # k_ff = -ca.solve(L, Du_Q_hat_ph)
        K_fb = -ca.inv(L) @ Dub_Q_hat_ph
        k_ff = -ca.inv(L) @ Du_Q_hat_ph 

        args_in = [Du_Q_hat_ph, Duu_Q_hat_ph, Dub_Q_hat_ph, mu_u_ph]
        args_out = [K_fb, k_ff]
        self.sym_feedback_policy = ca.Function('feedback_policy', args_in, args_out)

        # ==============================================
        # Create value function computation function
        # ==============================================
        # Feedback policy placeholders
        K_fb_ph = ca.SX.sym('K_fb_ph', self.n_u, self.n_b)
        k_ff_ph = ca.SX.sym('k_ff_ph', self.n_u)

        V = Q_ph + ca.dot(Du_Q_ph, k_ff_ph) + ca.bilin(ca.tril2symm(Duu_Q_ph), k_ff_ph, k_ff_ph)/2
        Db_V = ca.sparsify(Db_Q_ph + K_fb_ph.T @ ca.tril2symm(Duu_Q_ph) @ k_ff_ph + K_fb_ph.T @ Du_Q_ph + Dub_Q_ph.T @ k_ff_ph)
        Dbb_V = ca.tril(ca.tril2symm(Dbb_Q_ph) + K_fb_ph.T @ ca.tril2symm(Duu_Q_ph) @ K_fb_ph + K_fb_ph.T @ Dub_Q_ph + Dub_Q_ph.T @ K_fb_ph)

        args_in = [Q_ph, Db_Q_ph, Du_Q_ph, Dbb_Q_ph, Duu_Q_ph, Dub_Q_ph, K_fb_ph, k_ff_ph]
        args_out = [V, Db_V, Dbb_V]
        self.sym_value = ca.Function('value', args_in, args_out)

        # ==============================================
        # Do iLQG
        # ==============================================
        # Nominal state and input
        b_nom = [ca.MX.sym('b_nom_%i' % k, self.n_b) for k in range(self.N+1)] # [b_k, ..., b_k+N]
        u_nom = [ca.MX.sym('u_nom_%i' % k, self.n_u) for k in range(self.N+1)] # [u_k, ..., u_k+N-1, u_k-1]

        mu_b = ca.MX.sym('mu_b', 1)
        mu_u = ca.MX.sym('mu_u', 1)

        ilqg_in = b_nom + u_nom + [mu_b, mu_u]

        K_fb = [None for _ in range(self.N)]
        k_ff = [None for _ in range(self.N)]

        # Placeholder value function approximations for each agent
        V_out = [None for _ in range(self.M)]
        Q_out = [None for _ in range(self.M)]

        for i in range(self.M):
            # Terminal value function is just the terminal cost
            V_out[i] = list(self.sym_term_costs[i](b_nom[-1]))

        # Iterating backwards through time
        for k in range(self.N-1, -1, -1):
            # Compute linear approx of belief dynamics
            W_k = ca.horzsplit(self.belief_dynamics.f_W(b_nom[k], u_nom[k]), 1)

            Db_g_k = self.belief_dynamics.f_Db_g(b_nom[k], u_nom[k])
            Du_g_k = self.belief_dynamics.f_Du_g(b_nom[k], u_nom[k])

            Db_W_k = list(self.belief_dynamics.f_Db_W(b_nom[k], u_nom[k]))
            Du_W_k = list(self.belief_dynamics.f_Du_W(b_nom[k], u_nom[k]))

            # Compute Jacobian and Hessian of stage cost function at current time step
            Du_Q_hat, Duu_Q_hat, Dub_Q_hat = [], [], []
            u_start = 0
            args_C = [b_nom[k], u_nom[k], u_nom[k-1]]
            for i in range(self.M):
                C_out = list(self.sym_stage_costs[i](*args_C))

                # Compute quadratic approx of state-action value function Q
                args_Q = C_out + V_out[i] + [Db_g_k, Du_g_k] + W_k + Db_W_k + Du_W_k + [mu_b]
                Q_out[i] = list(self.sym_state_action_value(*args_Q))

                n_u = self.belief_dynamics.dynamics_models[i].n_u
                Du_Q_hat.append(Q_out[i][2][u_start:u_start+n_u])
                Duu_Q_hat.append(ca.tril2symm(Q_out[i][4])[u_start:u_start+n_u,:])
                Dub_Q_hat.append(Q_out[i][5][u_start:u_start+n_u,:])
                u_start += n_u

            # Compute feedback policy
            args_policy = [ca.vertcat(*Du_Q_hat), ca.vertcat(*Duu_Q_hat), ca.vertcat(*Dub_Q_hat), mu_u]
            K_fb[k], k_ff[k] = self.sym_feedback_policy(*args_policy)

            for i in range(self.M):
                # Compute quadratic approx of value function V
                args_V = Q_out[i] + [K_fb[k], k_ff[k]]
                V_out[i] = list(self.sym_value(*args_V))

        # Iterating forwards through time for each line search coefficient
        ilqg_out = []
        if not self.line_search:
            self.alpha = [[np.ones(self.belief_dynamics.dynamics_models[i].n_u) for i in range(self.M)]]
        for j, a in enumerate(self.alpha):
            b_new = [ca.MX.sym('b_new_ph_%i' % k, self.n_b) for k in range(self.N+1)]
            u_new = [ca.MX.sym('u_new_ph_%i' % k, self.n_u) for k in range(self.N+1)]
            trajectory_cost = ca.MX.zeros(self.M)

            b_new[0] = b_nom[0]
            u_new[-1] = u_nom[-1]
            for k in range(self.N):
                # Compute control action
                u_new[k] = u_nom[k] + np.concatenate(a)*k_ff[k] + K_fb[k] @ (b_new[k]-b_nom[k])

                # Update belief
                b_new[k+1] = self.belief_dynamics.f_g(b_new[k], u_new[k])

                # Compute stage cost
                args_cost = [b_new[k], u_new[k], u_new[k-1]]

                for i in range(self.M):
                    trajectory_cost[i] += self.costs_sym[i]['stage'](*args_cost)

            # Compute terminal cost
            for i in range(self.M):
                trajectory_cost[i] += self.costs_sym[i]['term'](b_new[-1])

            ilqg_out += [ca.horzcat(*b_new), ca.horzcat(*u_new[:-1]), trajectory_cost]

        # Store ilqg iteration
        self.ilqg_sym = ca.Function('ilqg_iter', ilqg_in, ilqg_out)

        if self.code_gen and not self.jit:
            generator = ca.CodeGenerator(self.c_file_name)
            generator.add(self.ilqg_sym)

            # Set up paths
            cur_dir = pathlib.Path.cwd()
            gen_path = cur_dir.joinpath(self.solver_name)
            c_path = gen_path.joinpath(self.c_file_name)
            if gen_path.exists():
                shutil.rmtree(gen_path)
            gen_path.mkdir(parents=True)

            os.chdir(gen_path)
            if self.verbose:
                print('- Generating C code for solver %s at %s' % (self.solver_name, str(gen_path)))
            generator.generate()
            pdb.set_trace()
            # Compile into shared object
            so_path = gen_path.joinpath(self.so_file_name)
            if self.verbose:
                print('- Compiling shared object %s from %s' % (so_path, c_path))
                print('- Executing "gcc -fPIC -shared -%s %s -o %s"' % (self.opt_flag, c_path, so_path))
            os.system('gcc -fPIC -shared -%s %s -o %s' % (self.opt_flag, c_path, so_path))

            # Swtich back to working directory
            os.chdir(cur_dir)

            install_dir = self.install()
            # Load solver
            self._load_solver(install_dir.joinpath(self.so_file_name))
            pdb.set_trace()

    def _load_solver(self, solver_path=None):
        if solver_path is None:
            solver_path = pathlib.Path(self.solver_dir, self.so_file_name).expanduser()
        if self.verbose:
            print('- Loading solver from %s' % str(solver_path))
        self.ilqg_sym = ca.external('ilqg_iter', str(solver_path))

    def get_prediction(self, target=False):
        if target:
            return self.belief_input_predictions[0], self.belief_input_predictions[1:]
        else:
            return self.belief_input_predictions[0]

if __name__ == '__main__':
    pass

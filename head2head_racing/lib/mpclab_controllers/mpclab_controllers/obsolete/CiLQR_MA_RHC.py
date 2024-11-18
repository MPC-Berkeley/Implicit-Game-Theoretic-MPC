#!/usr/bin python3

import numpy as np
import matplotlib.pyplot as plt

import casadi as ca

import copy
import pdb
from datetime import datetime
import os
import shutil
import pathlib
import itertools
from typing import List

from mpclab_common.models.dynamics_models import CasadiDecoupledMultiAgentDynamicsModel
from mpclab_common.pytypes import VehicleState, VehiclePrediction

from mpclab_controllers.abstract_controller import AbstractController
from mpclab_controllers.utils.controllerTypes import iLQRParams

class CiLQR_MA_RHC(AbstractController):
    def __init__(self, joint_dynamics: CasadiDecoupledMultiAgentDynamicsModel, costs, params=iLQRParams()):
        self.joint_dynamics = joint_dynamics
        self.M = self.joint_dynamics.n_a # Number of agents

        self.N = params.N
        self.max_iters = params.max_iters
        self.verbose = params.verbose
        self.code_gen = params.code_gen
        self.jit = params.jit
        self.opt_flag = params.opt_flag
        self.solver_name = params.solver_name
        if params.solver_dir is not None:
            self.solver_dir = os.path.join(params.solver_dir, self.solver_name)

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

        # The costs should be a list of dicts of casadi functions with keys 'stage' and 'terminal'
        if len(costs) != self.M:
            raise ValueError('Number of agents: %i, but only %i cost functions were provided' % (self.M, len(costs)))
        self.costs_sym = costs

        self.trajectory_cost_prev = np.inf

        self.state_input_predictions = [VehiclePrediction() for _ in range(self.M)]

        self.n_u = self.joint_dynamics.n_u
        self.n_q = self.joint_dynamics.n_q

        self.tol = params.tol
        self.rel_tol = params.rel_tol

        self.control_reg_init = params.control_reg
        self.state_reg_init = params.state_reg

        self.debug_plot = params.debug_plot
        self.pause_on_plot = params.pause_on_plot
        if self.debug_plot:
            plt.ion()
            self.fig = plt.figure(figsize=(10,5))
            self.ax_xy = self.fig.add_subplot(1,2,1)
            # self.joint_dynamics.dynamics_models[0].track.remove_phase_out()
            self.joint_dynamics.dynamics_models[0].track.plot_map(self.ax_xy, close_loop=False)
            self.l1_xy, self.l2_xy = self.ax_xy.plot([], [], 'bo', [], [], 'go')
            self.ax_a = self.fig.add_subplot(2,2,2)
            self.l1_a, self.l2_a = self.ax_a.plot([], [], '-bo', [], [], '-go')
            self.ax_s = self.fig.add_subplot(2,2,4)
            self.l1_s, self.l2_s = self.ax_s.plot([], [], '-bo', [], [], '-go')
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()

        # backtracking line search coefficients
        self.line_search = params.line_search
        if self.line_search:
            alpha = [np.kron(np.power(10, np.linspace(0, -2, 3).reshape((-1,1))),np.ones(self.joint_dynamics.dynamics_models[i].n_u)) for i in range(self.M)]
            self.alpha = list(itertools.product(*alpha))
            self.weights = np.array([0.7, 0.3])
        else:
            self.alpha_init = params.alpha_init
            self.gamma = params.gamma

        self.q_nom = np.zeros((self.N+1, self.n_q))
        self.u_nom = np.zeros((self.N, self.n_u))

        self.u_prev = np.zeros(self.n_u)

        # Symbolic backward pass
        self.back_sym = [None for _ in range(self.N+1)]
        self.forw_sym = [None for _ in range(self.N+1)]
        self.ilqg_sym = None

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

    def set_warm_start(self, q_ws: np.ndarray, u_ws: np.ndarray):
        if q_ws.shape[0] != self.N+1 or q_ws.shape[1] != self.n_q:
            raise(RuntimeError('Warm start state sequence of shape (%i,%i) is incompatible with required shape (%i,%i)' % (q_ws.shape[0],q_ws.shape[1],self.N+1,self.n_q)))
        if u_ws.shape[0] != self.N or u_ws.shape[1] != self.n_u:
            raise(RuntimeError('Warm start state sequence of shape (%i,%i) is incompatible with required shape (%i,%i)' % (u_ws.shape[0],u_ws.shape[1],self.N,self.n_u)))

        self.q_nom = q_ws
        self.u_nom = u_ws

    def step(self, vehicle_states: List[VehicleState], env_state=None):
        info = self.solve(vehicle_states)

        self.joint_dynamics.qu2state(vehicle_states, None, self.u_nom[0])
        self.joint_dynamics.qu2prediction(self.state_input_predictions, self.q_nom, self.u_nom)
        for q in self.state_input_predictions:
            q.t = vehicle_states[0].t

        self.u_prev = self.u_nom[0]

        return

    def solve(self, vehicle_states: List[VehicleState]):
        if self.verbose:
            print('Solving for %s' % self.solver_name)
        init_start = datetime.now()
        converged = False

        # Rollout nominal state trajectory using input sequence from last solve
        self.q_nom[0] = self.joint_dynamics.state2q(vehicle_states)
        self.u_nom = np.vstack((self.u_nom[1:], self.u_nom[-1].reshape((1,-1)), self.u_prev.reshape((1,-1))))

        cost_nom = np.zeros(self.M)
        for k in range(self.N):
            # Compute stage cost
            args_cost = [self.q_nom[k], self.u_nom[k], self.u_nom[k-1]]
            for i in range(self.M):
                cost_nom[i] += self.costs_sym[i]['stage'](*args_cost)

            # Update state
            args = [self.q_nom[k], self.u_nom[k]]
            self.q_nom[k+1] = self.joint_dynamics.fd(*args).toarray().squeeze()

        # Compute terminal cost
        for i in range(self.M):
            cost_nom[i] += self.costs_sym[i]['term'](self.q_nom[-1])
        cost_prev = copy.copy(cost_nom)

        q_nom = ca.DM(self.q_nom.T)
        u_nom = ca.DM(self.u_nom.T)

        self.control_reg = copy.copy(self.control_reg_init)
        self.state_reg = copy.copy(self.state_reg_init)
        if not self.line_search:
            alpha = copy.copy(self.alpha_init)

        if self.debug_plot:
            self._update_debug_plot(q_nom, u_nom)
            if self.pause_on_plot:
                ilqg_in = ca.horzsplit(q_nom, 1) + ca.horzsplit(u_nom, 1) + [self.state_reg, self.control_reg]
                pdb.set_trace()

        # Do iLQG iterations
        converged = False
        init_time = (datetime.now()-init_start).total_seconds()
        for i in range(self.max_iters):
            # Do backward and forward pass
            iter_start_time = datetime.now()
            ilqg_in = ca.horzsplit(q_nom, 1) + ca.horzsplit(u_nom, 1) + [self.state_reg, self.control_reg]
            if not self.line_search:
                ilqg_in += [alpha]
            ilqg_out = self.ilqg_sym(*ilqg_in)

            if self.line_search:
                avg_cost = np.inf
                for j in range(len(self.alpha)):
                    q_j, u_j, cost_j = ilqg_out[j*3:(j+1)*3]
                    cost_j = cost_j.toarray().squeeze()
                    if np.any(np.isnan(cost_j)):
                        print(cost_j)
                        pdb.set_trace()
                    if cost_j.dot(self.weights) < avg_cost:
                        avg_cost = cost_j.dot(self.weights)
                        q_new, u_new = q_j, u_j
                        cost = cost_j
            else:
                q_new, u_new, cost = ilqg_out
                cost = cost.toarray().squeeze()
                alpha *= self.gamma # Decay step size
                if np.any(np.isnan(cost)):
                    print(cost)
                    pdb.set_trace()

            back_forw_time = (datetime.now()-iter_start_time).total_seconds()

            if self.verbose:
                print(np.linalg.norm((q_new-q_nom).toarray().flatten(), ord=np.inf))
                print(np.linalg.norm(cost-cost_nom, ord=np.inf))
            if np.linalg.norm((q_new-q_nom).toarray().flatten(), ord=np.inf) <= self.tol:
            # if np.linalg.norm(cost-cost_nom, ord=np.inf) <= self.tol:
                converged = True
                if self.verbose:
                    print('it: %i | %g s | nom cost: %s | curr cost: %s'  % (i, back_forw_time, str(cost_nom), str(cost)))
                break
            
            s = 'nom cost: %s | curr cost: %s | prev cost: %s' % (str(cost_nom), str(cost), str(cost_prev))
            q_nom = copy.copy(q_new)
            u_nom = ca.horzcat(u_new, ca.DM(self.u_prev))
            cost_nom = copy.copy(cost)

            cost_prev = copy.copy(cost)
            iter_time = (datetime.now()-iter_start_time).total_seconds()
            if self.verbose:
                print('it: %i | %g s | total: %g | %s' % (i, back_forw_time, iter_time, s))

            if self.debug_plot:
                self._update_debug_plot(q_nom, u_nom)
                if self.pause_on_plot:
                    pdb.set_trace()

        self.q_nom = copy.copy(q_new.toarray().T)
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

        if self.debug_plot:
            plt.ioff()

        info = {'success': converged, 'return_status': return_status, 'solve_time': ilqg_time, 'info': None, 'output': None}

        return info

    def _build_solver(self):
        # =================================
        # Create cost computation functions
        # =================================
        # Placeholder symbolic variables for nominal sequence
        q_ph = ca.MX.sym('q_ph', self.n_q) # State
        u_ph = ca.MX.sym('u_ph', self.n_u) # Input
        um_ph = ca.MX.sym('um_ph', self.n_u) # Previous input

        self.sym_stage_costs, self.sym_term_costs = [], []
        for i in range(self.M):
            C = self.costs_sym[i]['stage'](q_ph, u_ph, um_ph)
            Dqq_C, _ = ca.hessian(C, q_ph)
            Dq_C = ca.jacobian(C, q_ph)
            Duu_C, _ = ca.hessian(C, u_ph)
            Du_C = ca.jacobian(C, u_ph)
            Duq_C = ca.jacobian(Du_C, q_ph)

            args_in = [q_ph, u_ph, um_ph]
            args_out = [C, Dq_C, Du_C, Dqq_C, Duu_C, Duq_C]
            self.sym_stage_costs.append(ca.Function('stage_cost_agent_%i' % i, args_in, args_out))

            V = self.costs_sym[i]['term'](q_ph)
            Dqq_V, _ = ca.hessian(V, q_ph)
            Dq_V = ca.jacobian(V, q_ph)

            args_in = [q_ph]
            args_out = [V, Dq_V, Dqq_V]
            self.sym_term_costs.append(ca.Function('term_cost_agent_%i' % i, args_in, args_out))

        # ==============================================
        # Create state-action value computation function
        # ==============================================
        # # Stage cost placeholders
        C_ph = ca.MX.sym('C_ph', 1)
        Dq_C_ph = ca.MX.sym('Dq_C_ph', self.n_q)
        Du_C_ph = ca.MX.sym('Du_C_ph', self.n_u)
        Dqq_C_ph = ca.MX.sym('Dqq_C_ph', self.n_q, self.n_q)
        Duu_C_ph = ca.MX.sym('Duu_C_ph', self.n_u, self.n_u)
        Duq_C_ph = ca.MX.sym('Duq_C_ph', self.n_u, self.n_q)

        # Dynamics placeholders
        Dq_f_ph = ca.MX.sym('Dq_f_ph', self.n_q, self.n_q)
        Du_f_ph = ca.MX.sym('Du_f_ph', self.n_q, self.n_u)
 
        # Value function at next time step placeholders
        Vp_ph = ca.MX.sym('Vp_ph', 1)
        Dq_Vp_ph = ca.MX.sym('Dq_Vp_ph', self.n_q)
        Dqq_Vp_ph = ca.MX.sym('Dqq_Vp_ph', self.n_q, self.n_q)

        # State regularization placeholder
        q_reg_ph = ca.MX.sym('q_reg_ph', 1)

        # Constant term of quadratic approx of Q
        Q = C_ph + Vp_ph

        # Linear term of quadratic approx of Q
        Dq_Q = Dq_C_ph + Dq_f_ph.T @ Dq_Vp_ph
        Du_Q = Du_C_ph + Du_f_ph.T @ Dq_Vp_ph

        # Quadratic term of quadratic approx of Q
        Dqq_Vp_reg = Dqq_Vp_ph + q_reg_ph*ca.DM.eye(self.n_q)
        Dqq_Q = Dqq_C_ph + Dq_f_ph.T @ Dqq_Vp_reg @ Dq_f_ph
        Duu_Q = Duu_C_ph + Du_f_ph.T @ Dqq_Vp_reg @ Du_f_ph
        Duq_Q = Duq_C_ph + Du_f_ph.T @ Dqq_Vp_reg @ Dq_f_ph
               
        args_in = [C_ph, Dq_C_ph, Du_C_ph, Dqq_C_ph, Duu_C_ph, Duq_C_ph] \
                    + [Vp_ph, Dq_Vp_ph, Dqq_Vp_ph] \
                    + [Dq_f_ph, Du_f_ph] \
                    + [q_reg_ph]
        args_out = [Q, Dq_Q, Du_Q, Dqq_Q, Duu_Q, Duq_Q]
        self.sym_state_action_value = ca.Function('state_action_value', args_in, args_out)

        # ==============================================
        # Create feedback policy computation function
        # ==============================================
        # Stage-action value function placeholders
        Q_ph = ca.SX.sym('Q_ph', 1)
        Dq_Q_ph = ca.SX.sym('Dq_Q_ph', self.n_q)
        Du_Q_ph = ca.SX.sym('Du_Q_ph', self.n_u)
        Dqq_Q_ph = ca.SX.sym('Dqq_Q_ph', self.n_q, self.n_q)
        Duu_Q_ph = ca.SX.sym('Duu_Q_ph', self.n_u, self.n_u)
        Duq_Q_ph = ca.SX.sym('Duq_Q_ph', self.n_u, self.n_q)

        Du_Q_hat_ph = ca.SX.sym('Du_Q_hat_ph', self.n_u)
        Duu_Q_hat_ph = ca.SX.sym('Duu_Q_hat_ph', self.n_u, self.n_u)
        Duq_Q_hat_ph = ca.SX.sym('Duq_Q_hat_ph', self.n_u, self.n_q)

        # Input regularization placeholder
        u_reg_ph = ca.SX.sym('u_reg_ph', 1)

        L = Duu_Q_hat_ph + u_reg_ph*ca.DM.eye(self.n_u)
        K_fb = -ca.solve(L, Duq_Q_hat_ph)
        k_ff = -ca.solve(L, Du_Q_hat_ph)

        args_in = [Du_Q_hat_ph, Duu_Q_hat_ph, Duq_Q_hat_ph, u_reg_ph]
        args_out = [K_fb, k_ff]
        self.sym_feedback_policy = ca.Function('feedback_policy', args_in, args_out)

        # ==============================================
        # Create value function computation function
        # ==============================================
        # Feedback policy placeholders
        K_fb_ph = ca.SX.sym('K_fb_ph', self.n_u, self.n_q)
        k_ff_ph = ca.SX.sym('k_ff_ph', self.n_u)

        V = Q_ph + ca.dot(Du_Q_ph, k_ff_ph) + ca.bilin(Duu_Q_ph, k_ff_ph, k_ff_ph)/2
        Dq_V = ca.sparsify(Dq_Q_ph + K_fb_ph.T @ Duu_Q_ph @ k_ff_ph + K_fb_ph.T @ Du_Q_ph + Duq_Q_ph.T @ k_ff_ph)
        Dqq_V = Dqq_Q_ph + K_fb_ph.T @ Duu_Q_ph @ K_fb_ph + K_fb_ph.T @ Duq_Q_ph + Duq_Q_ph.T @ K_fb_ph

        args_in = [Q_ph, Dq_Q_ph, Du_Q_ph, Dqq_Q_ph, Duu_Q_ph, Duq_Q_ph, K_fb_ph, k_ff_ph]
        args_out = [V, Dq_V, Dqq_V]
        self.sym_value = ca.Function('value', args_in, args_out)

        # ==============================================
        # Do iLQG
        # ==============================================
        # Nominal state and input
        q_nom = [ca.MX.sym('q_nom_%i' % k, self.n_q) for k in range(self.N+1)] # [q_k, ..., q_k+N]
        u_nom = [ca.MX.sym('u_nom_%i' % k, self.n_u) for k in range(self.N+1)] # [u_k, ..., u_k+N-1, u_k-1]

        q_reg = ca.MX.sym('q_reg', 1)
        u_reg = ca.MX.sym('u_reg', 1)

        ilqg_in = q_nom + u_nom + [q_reg, u_reg]

        K_fb = [None for _ in range(self.N)]
        k_ff = [None for _ in range(self.N)]

        # Placeholder value function approximations for each agent
        V_out = [None for _ in range(self.M)]
        Q_out = [None for _ in range(self.M)]

        if self.debug_plot:
            self.C_debug_fns = [[None for _ in range(self.N)] for _ in range(self.M)]
            self.Q_debug_fns = [[None for _ in range(self.N)] for _ in range(self.M)]
            self.V_debug_fns = [[None for _ in range(self.N+1)] for _ in range(self.M)]

        for i in range(self.M):
            # Terminal value function is just the terminal cost
            args_V = [q_nom[-1]]
            V_out[i] = list(self.sym_term_costs[i](*args_V))
            if self.debug_plot:
                self.V_debug_fns[i][-1] = ca.Function('V_debug_N_%i' % i, ilqg_in, V_out[i])

        # Iterating backwards through time
        for k in range(self.N-1, -1, -1):
            # Compute linear approx of dynamics
            Dq_f_k = self.joint_dynamics.fAd(q_nom[k], u_nom[k])
            Du_f_k = self.joint_dynamics.fBd(q_nom[k], u_nom[k])

            # Compute Jacobian and Hessian of stage cost function at current time step
            Du_Q_hat, Duu_Q_hat, Duq_Q_hat = [], [], []
            u_start = 0
            args_C = [q_nom[k], u_nom[k], u_nom[k-1]]
            for i in range(self.M):
                C_out = list(self.sym_stage_costs[i](*args_C))

                # Compute quadratic approx of state-action value function Q
                args_Q = C_out + V_out[i] + [Dq_f_k, Du_f_k] + [q_reg]
                Q_out[i] = list(self.sym_state_action_value(*args_Q))

                # Stack stationarity conditions for optimality
                n_u = self.joint_dynamics.dynamics_models[i].n_u
                Du_Q_hat.append(Q_out[i][2][u_start:u_start+n_u])
                Duu_Q_hat.append(Q_out[i][4][u_start:u_start+n_u,:])
                Duq_Q_hat.append(Q_out[i][5][u_start:u_start+n_u,:])
                u_start += n_u

                if self.debug_plot:
                    self.C_debug_fns[i][k] = ca.Function('C_debug_%i_%i' % (k, i), ilqg_in, C_out)
                    self.Q_debug_fns[i][k] = ca.Function('Q_debug_%i_%i' % (k, i), ilqg_in, Q_out[i])

            # Compute feedback policy
            args_policy = [ca.vertcat(*Du_Q_hat), ca.vertcat(*Duu_Q_hat), ca.vertcat(*Duq_Q_hat), u_reg]
            K_fb[k], k_ff[k] = self.sym_feedback_policy(*args_policy)

            for i in range(self.M):
                # Compute quadratic approx of value function V
                args_V = Q_out[i] + [K_fb[k], k_ff[k]]
                V_out[i] = list(self.sym_value(*args_V))

                if self.debug_plot:
                    self.V_debug_fns[i][k] = ca.Function('V_debug_%i_%i' % (k, i), ilqg_in, V_out[i])

        # Iterating forwards through time for each line search coefficient
        if self.line_search:
            ilqg_out = []
            for j, a in enumerate(self.alpha):
                q_new = [ca.MX.sym('q_new_ph_%i' % k, self.n_q) for k in range(self.N+1)]
                u_new = [ca.MX.sym('u_new_ph_%i' % k, self.n_u) for k in range(self.N+1)]
                trajectory_cost = ca.MX.zeros(self.M)

                q_new[0] = q_nom[0]
                u_new[-1] = u_nom[-1]
                for k in range(self.N):
                    # Compute control action
                    u_new[k] = u_nom[k] + np.concatenate(a)*k_ff[k] + K_fb[k] @ (q_new[k]-q_nom[k])
                    # Update belief
                    q_new[k+1] = self.joint_dynamics.fd(q_new[k], u_new[k])

                    # Compute stage cost
                    args_cost = [q_new[k], u_new[k], u_new[k-1]]
                    for i in range(self.M):
                        trajectory_cost[i] += self.costs_sym[i]['stage'](*args_cost)

                # Compute terminal cost
                for i in range(self.M):
                    trajectory_cost[i] += self.costs_sym[i]['term'](q_new[-1])

                ilqg_out += [ca.horzcat(*q_new), ca.horzcat(*u_new[:-1]), trajectory_cost]
        else:
            q_new = [ca.MX.sym('q_new_ph_%i' % k, self.n_q) for k in range(self.N+1)]
            u_new = [ca.MX.sym('u_new_ph_%i' % k, self.n_u) for k in range(self.N+1)]
            alpha = ca.MX.sym('alpha', 1)
            ilqg_in += [alpha]
            trajectory_cost = ca.MX.zeros(self.M)

            q_new[0] = q_nom[0]
            u_new[-1] = u_nom[-1]
            for k in range(self.N):
                # Compute control action
                u_new[k] = u_nom[k] + alpha*k_ff[k] + K_fb[k] @ (q_new[k]-q_nom[k])
                # Update belief
                q_new[k+1] = self.joint_dynamics.fd(q_new[k], u_new[k])

                # Compute stage cost
                args_cost = [q_new[k], u_new[k], u_new[k-1]]
                for i in range(self.M):
                    trajectory_cost[i] += self.costs_sym[i]['stage'](*args_cost)

            # Compute terminal cost
            for i in range(self.M):
                trajectory_cost[i] += self.costs_sym[i]['term'](q_new[-1])

            ilqg_out = [ca.horzcat(*q_new), ca.horzcat(*u_new[:-1]), trajectory_cost]
        
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

    def get_prediction(self):
        return self.state_input_predictions

    def _update_debug_plot(self, q_nom, u_nom):
        self.l1_xy.set_data(q_nom.toarray()[0,:], q_nom.toarray()[1,:])
        self.l2_xy.set_data(q_nom.toarray()[4,:], q_nom.toarray()[5,:])
        self.ax_xy.set_aspect('equal')
        self.l1_a.set_data(np.arange(self.N), u_nom.toarray()[0,:-1])
        self.l1_s.set_data(np.arange(self.N), u_nom.toarray()[1,:-1])
        self.l2_a.set_data(np.arange(self.N), u_nom.toarray()[2,:-1])
        self.l2_s.set_data(np.arange(self.N), u_nom.toarray()[3,:-1])
        self.ax_a.relim()
        self.ax_a.autoscale_view()
        self.ax_s.relim()
        self.ax_s.autoscale_view()
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

if __name__ == '__main__':
    pass

#!/usr/bin python3

import numpy as np
import scipy as sp
import casadi as ca

import cvxpy as cp
# from cvxpygen import cpg

import os
import pathlib
import copy
import shutil
import pdb
import time
import itertools

# import matplotlib
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from typing import List, Dict

from mpclab_common.models.dynamics_models import CasadiDecoupledMultiAgentDynamicsModel
from mpclab_common.pytypes import VehicleState, VehiclePrediction

from mpclab_controllers.abstract_controller import AbstractController
from mpclab_controllers.utils.controllerTypes import DGSQPParams

class DGSQP(AbstractController):
    def __init__(self, joint_dynamics: CasadiDecoupledMultiAgentDynamicsModel, 
                       costs: List[List[ca.Function]], 
                       agent_constraints: List[ca.Function], 
                       shared_constraints: List[ca.Function],
                       bounds: Dict[str, VehicleState],
                       params=DGSQPParams(),
                       print_method=print,
                       xy_plot=None,
                       use_mx=False):
        self.joint_dynamics = joint_dynamics
        self.M = self.joint_dynamics.n_a
        if print_method is None:
            self.print_method = lambda s: None
        else:
            self.print_method = print_method
        self.qp_interface       = params.qp_interface
        self.qp_solver          = params.qp_solver

        self.N                  = params.N

        self.reg                = params.reg
        self.line_search_iters  = params.line_search_iters
        self.nonmono_ls         = params.nms
        self.sqp_iters          = params.sqp_iters
        self.conv_approx        = True
        self.merit_function     = params.merit_function

        # Convergence tolerance for SQP
        self.p_tol              = params.p_tol
        self.d_tol              = params.d_tol
        self.rel_tol_req        = 3

        # Line search parameters
        self.beta               = params.beta
        self.tau                = params.tau

        self.verbose            = params.verbose
        self.save_iter_data     = params.save_iter_data
        if params.time_limit is None:
            self.time_limit = np.inf
        else:
            self.time_limit = params.time_limit

        self.code_gen           = params.code_gen
        self.jit                = params.jit
        self.opt_flag           = params.opt_flag
        self.solver_name        = params.solver_name

        if use_mx:
            self.ca_sym = ca.MX.sym
        else:
            self.ca_sym = ca.SX.sym

        self.debug_plot         = params.debug_plot
        self.pause_on_plot      = params.pause_on_plot
        self.local_pos          = params.local_pos

        # self.hessian_approximation = 'none'
        # self.hessian_approximation = 'bfgs'
        self.hessian_approximation = params.hessian_approximation

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
            if params.solver_dir is not None:
                self.solver_dir = pathlib.Path(params.solver_dir).expanduser()
                self.so_file_name = params.so_name

        if self.debug_plot:
            plt.ion()
            self.fig = plt.figure(figsize=(10,5))
            self.ax_xy = self.fig.add_subplot(1,2,1)
            self.ax_a = self.fig.add_subplot(2,2,2)
            self.ax_s = self.fig.add_subplot(2,2,4)
            # self.joint_dynamics.dynamics_models[0].track.remove_phase_print_method()
            # if self.joint_dynamics.dynamics_models[0].track is not None:
            #     self.joint_dynamics.dynamics_models[0].track.plot_map(self.ax_xy, close_loop=False)
            if xy_plot is not None:
                xy_plot(self.ax_xy)
            self.colors = ['b', 'g', 'r', 'm', 'c']
            self.l_xy, self.l_a, self.l_s = [], [], []
            for i in range(self.M):
                self.l_xy.append(self.ax_xy.plot([], [], f'{self.colors[i]}o')[0])
                self.l_a.append(self.ax_a.plot([], [], f'-{self.colors[i]}o')[0])
                self.l_s.append(self.ax_s.plot([], [], f'-{self.colors[i]}o')[0])
            self.ax_a.set_ylabel('accel')
            self.ax_s.set_ylabel('steering')
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()

        # The costs should be a dict of casadi functions with keys 'stage' and 'terminal'
        if len(costs) != self.M:
            raise ValueError('Number of agents: %i, but %i cost functions were provided' % (self.M, len(costs)))
        self.costs_sym = costs

        # The constraints should be a list (of length N+1) of casadi functions such that constraints[i] <= 0
        # if len(constraints) != self.N+1:
        #     raise ValueError('Horizon length: %i, but %i constraint functions were provided' % (self.N+1, len(constraints)))
        self.constraints_sym = agent_constraints
        self.shared_constraints_sym = shared_constraints

        # Process box constraints
        self.state_ub, self.state_lb, self.input_ub, self.input_lb = [], [], [], []
        self.state_ub_idxs, self.state_lb_idxs, self.input_ub_idxs, self.input_lb_idxs = [], [], [], []
        for a in range(self.M):
            su, iu = self.joint_dynamics.dynamics_models[a].state2qu(bounds['ub'][a])
            sl, il = self.joint_dynamics.dynamics_models[a].state2qu(bounds['lb'][a])
            self.state_ub.append(su)
            self.state_lb.append(sl)
            self.input_ub.append(iu)
            self.input_lb.append(il)
            self.state_ub_idxs.append(np.where(su < np.inf)[0])
            self.state_lb_idxs.append(np.where(sl > -np.inf)[0])
            self.input_ub_idxs.append(np.where(iu < np.inf)[0])
            self.input_lb_idxs.append(np.where(il > -np.inf)[0])

        self.n_cs = [0 for _ in range(self.N+1)]
        self.n_ca = [[0 for _ in range(self.N+1)] for _ in range(self.M)]
        self.n_cbr = [[0 for _ in range(self.N+1)] for _ in range(self.M)]
        self.n_c = [0 for _ in range(self.N+1)]
        
        self.state_input_predictions = [VehiclePrediction() for _ in range(self.M)]

        self.n_u = self.joint_dynamics.n_u
        self.n_q = self.joint_dynamics.n_q
            
        self.q_pred = np.zeros((self.N+1, self.n_q))
        self.u_pred = np.zeros((self.N, self.n_u))

        self.q_new = np.zeros((self.N+1, self.n_q))
        self.u_new = np.zeros((self.N+1, self.n_u))

        self.num_qa_d = [int(self.joint_dynamics.dynamics_models[a].n_q) for a in range(self.M)]
        self.num_ua_d = [int(self.joint_dynamics.dynamics_models[a].n_u) for a in range(self.M)]
        self.num_ua_el = [int(self.N*self.joint_dynamics.dynamics_models[a].n_u) for a in range(self.M)]

        self.ua_idxs = [np.concatenate([np.arange(int(self.n_u*k+np.sum(self.num_ua_d[:a])), int(self.n_u*k+np.sum(self.num_ua_d[:a+1]))) for k in range(self.N)]) for a in range(self.M)]

        if params.solver_dir:
            self._load_solver()
        else:
            self._build_solver()
        
        self.u_prev = np.zeros(self.n_u)
        self.l_pred = np.zeros(np.sum(self.n_c))
        self.u_ws = np.zeros((self.N, self.n_u)).ravel()
        self.l_ws = None

        # Construct QP solver
        if self.conv_approx:
            if self.qp_interface == 'casadi':
                if self.qp_solver == 'osqp':
                    solver_opts = dict(error_on_fail=False, osqp=dict(polish=True, verbose=self.verbose))
                elif self.qp_solver == 'qrqp':
                    solver_opts = dict(error_on_fail=False)
                elif self.qp_solver == 'superscs':
                    solver_opts = dict(error_on_fail=False)
                elif self.qp_solver == 'qpoases':
                    solver_opts = dict(error_on_fail=False, sparse=False, printLevel='tabular' if self.verbose else 'none')
                elif self.qp_solver == 'cplex':
                    # Change this to match version of libcplex<CPLEX_VERSION>.so 
                    os.environ['CPLEX_VERSION'] = '2210'
                    solver_opts = dict(error_on_fail=False, cplex=dict(CPXPARAM_OptimalityTarget=1, CPXPARAM_ScreenOutput=self.verbose))

                if self.hessian_approximation == 'none':
                    prob = {'h': self.f_Q.sparsity_out(0), 'a': self.f_Du_C.sparsity_out(0)}
                else:
                    prob = {'h': ca.DM.zeros(self.f_Q.size_out(0)).sparsity(), 'a': self.f_Du_C.sparsity_out(0)}
                self.solver = ca.conic('qp', self.qp_solver, prob, solver_opts)
                self.dual_name = 'lam_a'
            # elif self.qp_interface == 'cvxpygen':
            #     Q_size = self.f_Q.size_out(0)
            #     L = cp.Parameter(Q_size, name='L')
            #     q = cp.Parameter(Q_size[1], name='q')

            #     G_sparsity = self.f_Du_C.sparsity_out(0)
            #     G_ij_idxs = list(zip(*G_sparsity.get_triplet()))
            #     G = cp.Parameter(G_sparsity.size(), name='G', sparsity=G_ij_idxs)
            #     g = cp.Parameter(G_sparsity.size1(), name='g')

            #     x = cp.Variable(Q_size[1], name='x')
            #     prob = cp.Problem(cp.Minimize(cp.sum_squares(L @ x)/2 + cp.scalar_product(q, x)), [G @ x + g <= 0])
            #     cpg.generate_code(prob, code_dir='inner_qp', solver='OSQP')

            #     from inner_qp.cpg_solver import cpg_solve
            #     prob.register_solve('cpg', cpg_solve)

            #     self.solver = dict(prob=prob, x=x, L=L, q=q, G=G, g=g)
            else:
                raise(ValueError(f'Unsupported QP interface {self.qp_interface}'))
        else:
            du_ph = ca.SX.sym('du', self.N*self.n_u)
            Q_ph = ca.SX.sym('Q', self.N*self.n_u, self.N*self.n_u)
            q_ph = ca.SX.sym('q', self.N*self.n_u)
            G_ph = ca.SX.sym('G', np.sum(self.n_c), self.N*self.n_u)

            f = ca.bilin(Q_ph, du_ph, du_ph)/2 + ca.dot(q_ph, du_ph)
            g = G_ph @ du_ph
            p = ca.vertcat(*ca.horzsplit(Q_ph), q_ph, *ca.horzsplit(G_ph))
            prob = dict(x=du_ph, f=f, g=g, p=p)

            # Build NLP solver for Newton step
            ipopt_opts = dict(max_iter=500,
                            linear_solver='ma27',
                            warm_start_init_point='yes',
                            mu_strategy='adaptive',
                            mu_init=1e-5,
                            mu_min=1e-15,
                            barrier_tol_factor=1,
                            print_level=0)
            solver_opts = dict(error_on_fail=False, 
                                verbose_init=self.verbose, 
                                ipopt=ipopt_opts)
            self.solver = ca.nlpsol('nlp', 'ipopt', prob, solver_opts)
            self.dual_name = 'lam_g'

        self.initialized = True

    def _solve_qp(self, Q, q, G, g, x0=None):
        t = time.time()
        if self.conv_approx:
            Q = self._nearestPD(Q)
        else:
            Q = (Q + Q.T)/2
        if self.reg > 0:
            Q += self.reg*np.eye(Q.shape[0])
        if x0 is None:
            x0 = np.zeros(self.N*self.n_u)

        try:
            if self.conv_approx:
                if self.qp_interface == 'casadi':
                    sol = self.solver(h=Q, g=q, a=G, uba=-g, x0=x0)
                    du = sol['x'].toarray().squeeze()
                    l_hat = sol['lam_a'].toarray().squeeze()
                    status = self.solver.stats()['return_status']
                # elif self.qp_interface == 'cvxpygen':
                #     L = np.linalg.cholesky(Q).T
                #     self.solver['L'].value = L
                #     self.solver['q'].value = q
                #     self.solver['G'].value = G.full()
                #     self.solver['g'].value = g
                #     sol = self.solver['prob'].solve(method='cpg')
                #     du = self.solver['x'].value
                #     l_hat = self.solver['prob'].constraints[0].dual_value
                #     status = self.solver['prob'].status
                else:
                    raise(ValueError(f'Unsupported QP interface {self.qp_interface}'))
            else:
                sol = self.solver(ubg=-g, x0=x0,
                                    p=np.concatenate((np.ravel(Q, 'F'), q, np.ravel(G, 'F'))))
                du = sol['x'].toarray().squeeze()
                l_hat = sol['lam_g'].toarray().squeeze()
                status = self.solver.stats()['return_status']
            
            if self.verbose:
                self.print_method(f'Status: {status}')
                self.print_method(f'Inner solve time: {time.time()-t}')
        except:
            du = [None]
            l_hat = [None]
        
        return du, l_hat

    def initialize(self):
        pass

    def set_warm_start(self, u_ws: np.ndarray, l_ws: np.ndarray = None):
        if u_ws.shape[0] != self.N or u_ws.shape[1] != self.n_u:
            raise(RuntimeError('Warm start state sequence of shape (%i,%i) is incompatible with required shape (%i,%i)' % (u_ws.shape[0],u_ws.shape[1],self.N,self.n_u)))
        
        u = []
        for a in range(self.M):
            si = int(np.sum(self.num_ua_d[:a]))
            ei = int(np.sum(self.num_ua_d[:a])+self.num_ua_d[a])
            u.append(u_ws[:,si:ei].ravel())
        self.u_ws = np.concatenate(u)
        self.l_ws = l_ws

    def step(self, states: List[VehicleState],  parameters: np.ndarray = np.array([])):
        info = self.solve(states, parameters)

        self.joint_dynamics.qu2state(states, None, self.u_pred[0])
        self.joint_dynamics.qu2prediction(self.state_input_predictions, self.q_pred, self.u_pred)
        for q in self.state_input_predictions:
            q.t = states[0].t

        self.u_prev = self.u_pred[0]

        if info['msg'] not in ['diverged', 'qp_fail']:
            u_ws = np.vstack((self.u_pred[1:], self.u_pred[-1]))
            self.set_warm_start(u_ws)

        return info

    def get_prediction(self) -> List[VehiclePrediction]:
        return self.state_input_predictions

    def solve(self, states: List[VehicleState], parameters: np.ndarray = np.array([])):
        solve_info = {}
        solve_start = time.time()
        self.u_prev = np.zeros(self.n_u)

        u = copy.copy(self.u_ws)
        up = copy.copy(self.u_prev)

        x0 = self.joint_dynamics.state2q(states)

        # Warm start dual variables
        # if self.l_ws is None:
        #     # Least squares approx
        #     q, G, _, _ = self._evaluate(u, None, x0, up, hessian=False)
        #     G = sp.sparse.csc_matrix(G)
        #     l = np.maximum(0, -sp.sparse.linalg.lsqr(G @ G.T, G @ q)[0])
        # else:
        #     l = copy.copy(self.l_ws)
        q, G, _, _ = self._evaluate(u, None, x0, up, P=parameters, hessian=False)
        # G = G.full()
        # l = np.maximum(0, -np.linalg.lstsq(G @ G.T, G @ q, rcond=None)[0])
        G = G.sparse()
        l = np.maximum(0, -sp.sparse.linalg.lsqr(G @ G.T, G @ q)[0])
        if l is None:
            l = np.zeros(np.sum(self.n_c))
        init = dict(u=u, l=l)

        if self.debug_plot:
            self._update_debug_plot(copy.copy(u), copy.copy(x0), copy.copy(up), copy.copy(parameters))
            if self.pause_on_plot:
                pdb.set_trace()

        sqp_converged = False
        rel_tol_its = 0
        sqp_it = 0
        iter_data = []
        # self.print_method(self.solver_name)
        while True:
            if self.verbose:
                self.print_method('===================================================')
                self.print_method(f'DGSQP iteration: {sqp_it}')

            if self.debug_plot:
                self._update_debug_plot(copy.copy(u), copy.copy(x0), copy.copy(up), copy.copy(parameters))
                if self.pause_on_plot:
                    pdb.set_trace()
            
            sqp_it_start = time.time()
            qp_solves = 0

            # Evaluate SQP approximation
            if sqp_it == 0 or self.hessian_approximation == 'none':
                # Exact Hessian
                Q_i, q_i, G_i, g_i, _ = self._evaluate(u, l, x0, up, P=parameters)
                # if self.hessian_approximation:
                #     B = list(map(lambda x: self._nearestPD(x.full()), self.f_Duu_L(u, l, x0, up)))
            else:
                q_i, G_i, g_i, _ = self._evaluate(u, l, x0, up, P=parameters, hessian=False)
                def jac_eval(u, l):
                    q, G, _, _ = self._evaluate(u, l, x0, up, P=parameters, hessian=False)
                    return [q + G.T @ l]
                # Approximate Hessian
                Q_i = self._hessian_approximation(u, l, u_im1, l_im1, [self._nearestPD(Q_i)], jac_eval, method=self.hessian_approximation)[0]
                # B = self._bfgs(u, l, u_im1, l_im1, B, lambda u, l: self.f_Du_L(u, l, x0, up))
                # Q_i = np.vstack([_B[int(np.sum(self.num_ua_el[:a])):int(np.sum(self.num_ua_el[:a+1])),:] for a, _B in enumerate(B)])

            d_i = q_i + G_i.T @ l

            u_im1 = copy.copy(u)
            l_im1 = copy.copy(l)

            # Convergence test
            xtol = self.p_tol
            ltol = self.d_tol
            p_feas = max(0, np.amax(g_i))
            comp = np.linalg.norm(g_i * l, ord=np.inf)
            stat = np.linalg.norm(d_i, ord=np.inf)
            cond = {'p_feas': p_feas, 'comp': comp, 'stat': stat}
            if self.verbose:
                self.print_method(f'SQP iteration {sqp_it}')
                self.print_method(f'p feas: {p_feas:.4e} | comp: {comp:.4e} | stat: {stat:.4e}')
            if stat > 1e5:
                sqp_it_dur = time.time() - sqp_it_start
                if self.save_iter_data:
                    iter_data.append(dict(cond=cond, u_sol=u, l_sol=l, qp_solves=qp_solves, it_time=sqp_it_dur))
                if self.verbose: self.print_method('SQP diverged')
                msg = 'diverged'
                sqp_converged = False
                break
            if p_feas < xtol and comp < ltol and stat < ltol:
                sqp_it_dur = time.time() - sqp_it_start
                if self.save_iter_data:
                    iter_data.append(dict(cond=cond, u_sol=u, l_sol=l, qp_solves=qp_solves, it_time=sqp_it_dur))
                sqp_converged = True
                msg = 'conv_abs_tol'
                if self.verbose: self.print_method('SQP converged via optimality conditions')
                break
            
            # Compute SQP primal dual step
            du, l_hat = self._solve_qp(Q_i, q_i, G_i, g_i)
            qp_solves += 1
            if None in du:
                sqp_it_dur = time.time() - sqp_it_start
                if self.save_iter_data:
                    iter_data.append(dict(cond=cond, u_sol=u, l_sol=l, qp_solves=qp_solves, it_time=sqp_it_dur))
                sqp_converged = False
                msg = 'qp_fail'
                if self.verbose: self.print_method('QP solution failed')
                break
            dl = l_hat - l

            ls = True
            s = np.minimum(0, g_i)
            ds = g_i + G_i @ du - s
            mu = self._get_mu(u, du, l, dl, s, ds, Q_i, q_i, G_i, g_i)

            # Do line search
            if ls:
                if self.nonmono_ls:
                    # u, l, n_qp = self._watchdog_line_search_3(u, du, l, dl, s, ds, Q_i, q_i, G_i, g_i, 
                    #                 lambda u, l, hessian: self._evaluate(u, l, x0, up, hessian),
                    #                 lambda u, l, s, q, G, g: float(self.f_phi(l, s, q, G, g, mu)),
                    #                 lambda u, du, l, dl, s, ds, Q, q, G, g: float(self.f_dphi(du, l, dl, s, Q, q, G, g, mu)))
                    u, l, n_qp = self._watchdog_line_search_4(u, du, l, dl, s, ds, Q_i, q_i, G_i, g_i, 
                                    lambda u, l, hessian: self._evaluate(u, l, x0, up, P=parameters, hessian=hessian),
                                    lambda u, l, s, q, G, g: float(self.f_phi(l, s, q, G, g, mu)),
                                    lambda u, du, l, dl, s, ds, Q, q, G, g: float(self.f_dphi(du, l, dl, s, Q, q, G, g, mu)))
                    # u, l, n_qp = self._watchdog_line_search_4(u, du, l, dl, Q_i, q_i, G_i, g_i, 
                    #                 lambda u, l, hessian: self._evaluate(u, l, x0, up, hessian),
                    #                 lambda u, l, s, q, G, g, mu: float(self.f_phi(l, s, q, G, g, mu)),
                    #                 lambda u, du, l, dl, s, ds, Q, q, G, g, mu: float(self.f_dphi(du, l, dl, s, Q, q, G, g, mu)))
                    qp_solves += n_qp
                else:
                    u, l, _ = self._line_search_3(u, du, l, dl, s, ds, Q_i, q_i, G_i, g_i, 
                                    lambda u, l, hessian: self._evaluate(u, l, x0, up, P=parameters, hessian=hessian),
                                    lambda u, l, s, q, G, g: float(self.f_phi(l, s, q, G, g, mu)),
                                    lambda u, du, l, dl, s, ds, Q, q, G, g: float(self.f_dphi(du, l, dl, s, Q, q, G, g, mu)))
            else:
                u += 0.01*du
                l += 0.01*dl

            sqp_it_dur = time.time() - sqp_it_start
            if self.verbose:
                J = self.f_J(u, x0, up, parameters)
                self.print_method(str(J))
                self.print_method(f'SQP iteration {sqp_it} time: {sqp_it_dur}')
                self.print_method('===================================================')
            
            if self.save_iter_data:
                iter_data.append(dict(cond=cond, u_sol=u, l_sol=l, qp_solves=qp_solves, it_time=sqp_it_dur))

            # Convergence via relative tolerance
            # if np.linalg.norm(u-u_im1) < xtol/2 and np.linalg.norm(l-l_im1) < ltol/2:
            if np.linalg.norm(u-u_im1) < xtol and np.linalg.norm(l-l_im1) < ltol:
                rel_tol_its += 1
                if rel_tol_its >= self.rel_tol_req and p_feas < xtol:
                    sqp_converged = True
                    msg = 'conv_rel_tol'
                    if self.verbose: self.print_method('SQP converged via relative tolerance')
                    break
            else:
                rel_tol_its = 0

            sqp_it += 1
            if sqp_it >= self.sqp_iters:
                msg = 'max_it'
                sqp_converged = False
                if self.verbose: self.print_method('Max SQP iterations reached')
                break
            if time.time() - solve_start > self.time_limit:
                msg = 'time_limit'
                sqp_converged = False
                if self.verbose: self.print_method('Time limit reached')
                break
        
        x_bar = np.array(self.evaluate_dynamics(u, x0)).squeeze()
        ua_bar = []
        for a in range(self.M):
            si = int(np.sum(self.num_ua_el[:a]))
            ei = int(np.sum(self.num_ua_el[:a])+self.num_ua_el[a])
            ua_bar.append(u[si:ei].reshape((self.N, self.num_ua_d[a])))
        u_bar = np.hstack(ua_bar)

        self.q_pred = x_bar
        self.u_pred = u_bar
        self.l_pred = l

        solve_dur = time.time() - solve_start
        self.print_method(f'p feas: {p_feas:.4e} | comp: {comp:.4e} | stat: {stat:.4e}')
        self.print_method(f'Solve status: {msg}')
        p_feas, comp, stat = cond['p_feas'], cond['comp'], cond['stat']
        self.print_method(f'Solve stats: p feas: {p_feas:.4e} | comp: {comp:.4e} | stat: {stat:.4e}')
        self.print_method(f'Solve iters: {sqp_it}')
        self.print_method(f'Solve time: {solve_dur:.2f}')
        J = self.f_J(u, x0, up, parameters)
        self.print_method(str(np.array(J).squeeze()))

        # active_idxs = [np.where(l[self.Cbr_v_idxs[a]] > 1e-7)[0] for a in range(self.M)]
        # A = [self.f_Du_Cbr[a](u, x0, up).toarray()[active_idxs[a]] for a in range(self.M)]
        # N = [sp.linalg.null_space(A[a]) for a in range(self.M)]
        # Luu = [self.f_Duu_L[a](u, l, x0, up)[a] for a in range(self.M)]
        # pdb.set_trace()

        solve_info['time'] = solve_dur
        solve_info['num_iters'] = sqp_it
        solve_info['status'] = sqp_converged
        solve_info['cost'] = J
        solve_info['cond'] = cond
        solve_info['iter_data'] = iter_data
        solve_info['msg'] = msg
        solve_info['init'] = init

        if self.debug_plot:
            plt.ioff()

        return solve_info

    def _evaluate(self, u, l, x0, up, P: np.ndarray = np.array([]), hessian: bool = True):
        eval_start = time.time()
        x = ca.vertcat(*self.evaluate_dynamics(u, x0))
        A = self.evaluate_jacobian_A(x, u)
        B = self.evaluate_jacobian_B(x, u)
        Du_x = self.f_Du_x(*A, *B)

        g = ca.vertcat(*self.f_Cxu(x, u, up, P)).full().squeeze()
        H = self.f_Du_C(x, u, up, Du_x, P)
        q = self.f_q(x, u, up, Du_x, P).full().squeeze()

        if hessian:
            E = self.evaluate_hessian_E(x, u)
            F = self.evaluate_hessian_F(x, u)
            G = self.evaluate_hessian_G(x, u)
            Q = self.f_Q(x, u, l, up, *A, *B, *E, *F, *G, P)
            eval_time = time.time() - eval_start
            if self.verbose:
                self.print_method(f'Jacobian and Hessian evaluation time: {eval_time}')
            return Q, q, H, g, x
        else:
            eval_time = time.time() - eval_start
            if self.verbose:
                self.print_method(f'Jacobian evaluation time: {eval_time}')
            return q, H, g, x

    def _hessian_approximation(self, u, l, um, lm, Bm, df, method='bfgs'):
        eval_start = time.time()
        if method == 'bfgs':
            # Damped BFGS updating from Nocedal 1999 Procedure 18.2
            s = u - um
            d = df(u, l)
            dm = df(um, l)
            y = [_d.full().squeeze() - _dm.full().squeeze() for _d, _dm in zip(d, dm)]

            Bs = [_Bm @ s for _Bm in Bm]
            sBs = [np.dot(s, _Bs) for _Bs in Bs]
            sy = [np.dot(s, _y) for _y in y]
            t = [1 if _sy >= 0.2*_sBs else 0.8*_sBs/(_sBs-_sy) for _sy, _sBs in zip(sy, sBs)]
            
            r = [_t*_y + (1-_t)*_Bs for _t, _y, _Bs in zip(t, y, Bs)]
            B = [_Bm - np.outer(_Bs, _Bs)/_sBs + np.outer(_r, _r)/np.dot(s, _r) for _Bm, _Bs, _sBs, _r in zip(Bm, Bs, sBs, r)]
        else:
            raise(ValueError(f'Hessian approximation method {method} not implmented'))
        eval_time = time.time() - eval_start
        if self.verbose:
            self.print_method(f'Approximate Hessian evaluation time: {eval_time}')

        return B

    def _get_mu(self, u, du, l, dl, s, ds, Q, q, G, g):
        thresh = 0
        if self.merit_function == 'stat_l1':
            constr_vio = g - s
            d_stat_norm = float(self.f_dstat_norm(du, l, dl, s, Q, q, G, g, 0))
            rho = 0.5
            
            if d_stat_norm < 0 and np.sum(constr_vio) > thresh:
                if self.verbose:
                    self.print_method('Case 1: negative directional derivative with constraint violation')
                mu = -d_stat_norm / ((1-rho)*np.sum(constr_vio))  
            elif d_stat_norm < 0 and np.sum(constr_vio) <= thresh:
                if self.verbose:
                    self.print_method('Case 2: negative directional derivative no constraint violation')
                mu = 0
            elif d_stat_norm >= 0 and np.sum(constr_vio) > thresh:
                if self.verbose:
                    self.print_method('Case 3: positive directional derivative with constraint violation')
                mu = d_stat_norm / ((1-rho)*np.sum(constr_vio))  
            elif d_stat_norm >= 0 and np.sum(constr_vio) <= thresh:
                if self.verbose:
                    self.print_method('Case 4: positive directional derivative no constraint violation')
                mu = 0
        elif self.merit_function == 'stat':
            mu = 0
        
        return mu
        
    def _build_solver(self):
        # u_0, ..., u_N-1, u_-1
        u_ph = [[self.ca_sym(f'u_{a}_ph_{k}', self.joint_dynamics.dynamics_models[a].n_u) for k in range(self.N+1)] for a in range(self.M)] # Agent inputs
        ua_ph = [ca.vertcat(*u_ph[a][:-1]) for a in range(self.M)] # [u_0^1, ..., u_{N-1}^1, u_0^2, ..., u_{N-1}^2]
        uk_ph = [ca.vertcat(*[u_ph[a][k] for a in range(self.M)]) for k in range(self.N+1)] # [[u_0^1, u_0^2], ..., [u_{N-1}^1, u_{N-1}^2]]

        agent_cost_params = [[] for _ in range(self.M)]
        agent_constraint_params = [[] for _ in range(self.M)]
        shared_constraint_params = []

        # Function for evaluating the dynamics function given an input sequence
        xr_ph = [self.ca_sym('xr_ph_0', self.n_q)] # Initial state
        for k in range(self.N):
            xr_ph.append(self.joint_dynamics.fd(xr_ph[k], uk_ph[k]))
        self.evaluate_dynamics = ca.Function('evaluate_dynamics', [ca.vertcat(*ua_ph), xr_ph[0]], xr_ph, self.options)

        # State sequence placeholders
        x_ph = [self.ca_sym(f'x_ph_{k}', self.n_q) for k in range(self.N+1)]

        # Function for evaluating the dynamics Jacobians given a state and input sequence
        A, B = [], []
        for k in range(self.N):
            A.append(self.joint_dynamics.fAd(x_ph[k], uk_ph[k]))
            B.append(self.joint_dynamics.fBd(x_ph[k], uk_ph[k]))
        self.evaluate_jacobian_A = ca.Function('evaluate_jacobian_A', [ca.vertcat(*x_ph), ca.vertcat(*ua_ph)], A, self.options)
        self.evaluate_jacobian_B = ca.Function('evaluate_jacobian_B', [ca.vertcat(*x_ph), ca.vertcat(*ua_ph)], B, self.options)

        # Placeholders for dynamics Jacobians
        # [Dx0_x1, Dx1_x2, ..., DxN-1_xN]
        A_ph = [self.ca_sym(f'A_ph_{k}', self.joint_dynamics.sym_Ad.sparsity()) for k in range(self.N)]
        # [Du0_x1, Du1_x2, ..., DuN-1_xN]
        B_ph = [self.ca_sym(f'B_ph_{k}', self.joint_dynamics.sym_Bd.sparsity()) for k in range(self.N)]

        # Function for evaluating the dynamics Hessians given a state and input sequence
        E, F, G = [], [], []
        for k in range(self.N):
            E += self.joint_dynamics.fEd(x_ph[k], uk_ph[k])
            F += self.joint_dynamics.fFd(x_ph[k], uk_ph[k])
            G += self.joint_dynamics.fGd(x_ph[k], uk_ph[k])
        self.evaluate_hessian_E = ca.Function('evaluate_hessian_E', [ca.vertcat(*x_ph), ca.vertcat(*ua_ph)], E)
        self.evaluate_hessian_F = ca.Function('evaluate_hessian_F', [ca.vertcat(*x_ph), ca.vertcat(*ua_ph)], F)
        self.evaluate_hessian_G = ca.Function('evaluate_hessian_G', [ca.vertcat(*x_ph), ca.vertcat(*ua_ph)], G)

        # Placeholders for dynamics Hessians
        E_ph, F_ph, G_ph = [], [], []
        for k in range(self.N):
            Ek, Fk, Gk = [], [], []
            for i in range(self.n_q):
                Ek.append(self.ca_sym(f'E{k}_ph_{i}', self.joint_dynamics.sym_Ed[i].sparsity()))
                Fk.append(self.ca_sym(f'F{k}_ph_{i}', self.joint_dynamics.sym_Fd[i].sparsity()))
                Gk.append(self.ca_sym(f'G{k}_ph_{i}', self.joint_dynamics.sym_Gd[i].sparsity()))
            E_ph.append(Ek)
            F_ph.append(Fk)
            G_ph.append(Gk)

        Du_x = []
        for k in range(self.N):
            Duk_x = [self.ca_sym(f'Du{k}_x', ca.Sparsity(self.n_q*(k+1), self.n_u)), B_ph[k]]
            for t in range(k+1, self.N):
                Duk_x.append(A_ph[t] @ Duk_x[-1])
            Du_x.append(ca.vertcat(*Duk_x))
        Du_x = ca.horzcat(*Du_x)
        Du_x = ca.horzcat(*[Du_x[:,self.ua_idxs[a]] for a in range(self.M)])
        self.f_Du_x = ca.Function('f_Du_x', A_ph + B_ph, [Du_x], self.options)

        Du_x_ph = self.ca_sym('Du_x', Du_x.sparsity())

        # Agent cost functions
        # J = [ca.DM.zeros(1) for _ in range(self.M)]
        J = [[] for _ in range(self.M)]
        for a in range(self.M):
            for k in range(self.N):
                if self.costs_sym[a][k].n_in == 4:
                    pJa_k = self.ca_sym(f'pJ{a}_{k}', self.costs_sym[a][k].numel_in(3))
                    J[a].append(self.costs_sym[a][k](x_ph[k], u_ph[a][k], u_ph[a][k-1], pJa_k))
                    agent_cost_params[a].append(pJa_k)
                else:
                    J[a].append(self.costs_sym[a][k](x_ph[k], u_ph[a][k], u_ph[a][k-1]))
            if self.costs_sym[a][-1].n_in == 2:
                pJa_k = self.ca_sym(f'pJ{a}_{self.N}', self.costs_sym[a][k].numel_in(1))
                J[a].append(self.costs_sym[a][-1](x_ph[-1], pJa_k))
                agent_cost_params[a].append(pJa_k)
            else:
                J[a].append(self.costs_sym[a][-1](x_ph[-1]))
        
        # First derivatives of cost function w.r.t. input sequence        
        Dx_Jxu = [ca.jacobian(ca.sum1(ca.vertcat(*J[a])), ca.vertcat(*x_ph)) for a in range(self.M)]
        Du_Jxu = [ca.jacobian(ca.sum1(ca.vertcat(*J[a])), ca.vertcat(*ua_ph)) for a in range(self.M)]
        Du_J = [(Du_Jxu[a] + Dx_Jxu[a] @ Du_x_ph).T for a in range(self.M)]
        Du_J = [[Du_J[a][int(np.sum(self.num_ua_el[:b])):int(np.sum(self.num_ua_el[:b+1]))] for b in range(self.M)] for a in range(self.M)]

        # Second derivatves of cost function w.r.t. input sequence using dynamic programming
        Duu_J = []
        for a in range(self.M):
            Duu_Q = []
            Dxu_Q = []
            Dx_Q = [ca.jacobian(J[a][-1], x_ph[-1])]
            Dxx_Q = [ca.jacobian(ca.jacobian(J[a][-1], x_ph[-1]), x_ph[-1])]
            for k in range(self.N-1, -1, -1):
                if k == self.N-1:
                    Jk = J[a][k]
                else:
                    Jk = J[a][k] + J[a][k+1]
                Dx_Jk = ca.jacobian(Jk, x_ph[k])
                Dxx_Jk = ca.jacobian(ca.jacobian(Jk, x_ph[k]), x_ph[k])
                Duu_Jk = ca.jacobian(ca.jacobian(Jk, uk_ph[k]), uk_ph[k])
                Dxu_Jk = ca.jacobian(ca.jacobian(Jk, uk_ph[k]), x_ph[k])
                Duu_Jk2 = ca.jacobian(ca.jacobian(Jk, uk_ph[k+1]), uk_ph[k])

                Dx_Qk = Dx_Jk + Dx_Q[-1] @ A_ph[k]

                A1 = Duu_Jk + B_ph[k].T @ Dxx_Q[-1] @ B_ph[k]
                for i in range(self.n_q):
                    A1 += Dx_Q[-1][i] * F_ph[k][i]
                if len(Dxu_Q) == 0:
                    Duu_Qk = A1
                else:
                    B1 = Dxu_Q[-1] @ B_ph[k]
                    B1[:Duu_Jk2.size1(),:] += Duu_Jk2
                    Duu_Qk = ca.blockcat([[A1, B1.T], [B1, Duu_Q[-1]]])

                A2 = Dxu_Jk + B_ph[k].T @ Dxx_Q[-1] @ A_ph[k]
                for i in range(self.n_q):
                    A2 += Dx_Q[-1][i] * G_ph[k][i]
                if len(Dxu_Q) == 0:
                    Dxu_Qk = A2
                else:
                    B2 = Dxu_Q[-1] @ A_ph[k]
                    Dxu_Qk = ca.vertcat(A2, B2)

                Dxx_Qk = Dxx_Jk + A_ph[k].T @ Dxx_Q[-1] @ A_ph[k]
                for i in range(self.n_q):
                    Dxx_Qk += Dx_Q[-1][i] * E_ph[k][i]
                
                Dx_Q.append(Dx_Qk)
                Dxx_Q.append(Dxx_Qk)
                Duu_Q.append(Duu_Qk)
                Dxu_Q.append(Dxu_Qk)
            Duu_Ja = ca.horzcat(*[Duu_Q[-1][:,self.ua_idxs[a]] for a in range(self.M)])
            Duu_Ja = ca.vertcat(*[Duu_Ja[self.ua_idxs[a],:] for a in range(self.M)])
            Duu_J.append(Duu_Ja)

        # Duu_J2 = [ca.jacobian(ca.jacobian(Ju[a], ca.vertcat(*ua_ph)), ca.vertcat(*ua_ph)) for a in range(self.M)]
        # self.f_Duu_J2 = ca.Function('f_Duu_J2', [ca.vertcat(*ua_ph), xr_ph[0], uk_ph[-1]], Duu_J2)

        # Placeholders for gradient of dynamics w.r.t. state and input
        Cs = [[] for _ in range(self.N+1)] # Shared constraints
        Ca = [[[] for _ in range(self.N+1)] for _ in range(self.M)] # Agent specific constraints
        for k in range(self.N):
            # Add shared constraints
            if self.shared_constraints_sym[k] is not None:
                if self.shared_constraints_sym[k].n_in() == 4:
                    pCs_k = self.ca_sym(f'pCs_{k}', self.shared_constraints_sym[k].numel_in(3))
                    Cs[k].append(self.shared_constraints_sym[k](x_ph[k], uk_ph[k], uk_ph[k-1], pCs_k))
                    shared_constraint_params.append(pCs_k)
                else:
                    Cs[k].append(self.shared_constraints_sym[k](x_ph[k], uk_ph[k], uk_ph[k-1]))
            if len(Cs[k]) > 0:
                Cs[k] = ca.vertcat(*Cs[k])
                self.n_cs[k] = Cs[k].shape[0]
            else:
                Cs[k] = ca.DM()
            # Add agent constraints
            for a in range(self.M):
                if self.constraints_sym[a][k] is not None:
                    if self.constraints_sym[a][k].n_in() == 4:
                        pCa_k = self.ca_sym(f'pC{a}_{k}', self.constraints_sym[a][k].numel_in(3))
                        Ca[a][k].append(self.constraints_sym[a][k](x_ph[k], u_ph[a][k], u_ph[a][k-1], pCa_k))
                        agent_constraint_params[a].append(pCa_k)
                    else:
                        Ca[a][k].append(self.constraints_sym[a][k](x_ph[k], u_ph[a][k], u_ph[a][k-1]))
                # Add agent box constraints
                if len(self.input_ub_idxs[a]) > 0:
                    Ca[a][k].append(u_ph[a][k][self.input_ub_idxs[a]] - self.input_ub[a][self.input_ub_idxs[a]])
                if len(self.input_lb_idxs[a]) > 0:
                    Ca[a][k].append(self.input_lb[a][self.input_lb_idxs[a]] - u_ph[a][k][self.input_lb_idxs[a]])
                if k > 0:
                    if len(self.state_ub_idxs[a]) > 0:
                        Ca[a][k].append(x_ph[k][self.state_ub_idxs[a]+int(np.sum(self.num_qa_d[:a]))] - self.state_ub[a][self.state_ub_idxs[a]])
                    if len(self.state_lb_idxs[a]) > 0:
                        Ca[a][k].append(self.state_lb[a][self.state_lb_idxs[a]] - x_ph[k][self.state_lb_idxs[a]+int(np.sum(self.num_qa_d[:a]))])
                if len(Ca[a][k]) > 0:
                    Ca[a][k] = ca.vertcat(*Ca[a][k])
                    self.n_ca[a][k] = Ca[a][k].shape[0]
                else:
                    Ca[a][k] = ca.DM()
        # Add shared constraints
        if self.shared_constraints_sym[-1] is not None:
            if self.shared_constraints_sym[-1].n_in() == 2:
                pCs_k = self.ca_sym(f'pCs_{self.N}', self.shared_constraints_sym[-1].numel_in(1))
                Cs[-1].append(self.shared_constraints_sym[-1](x_ph[-1], pCs_k))
                shared_constraint_params.append(pCs_k)
            else:
                Cs[-1].append(self.shared_constraints_sym[-1](x_ph[-1]))
        if len(Cs[-1]) > 0:
            Cs[-1] = ca.vertcat(*Cs[-1])
            self.n_cs[-1] = Cs[-1].shape[0]
        else:
            Cs[-1] = ca.DM()
        # Add agent constraints
        for a in range(self.M):
            if self.constraints_sym[a][-1] is not None:
                if self.constraints_sym[a][-1].n_in() == 2:
                    pCa_k = self.ca_sym(f'pC{a}_{self.N}', self.constraints_sym[a][-1].numel_in(1))
                    Ca[a][-1].append(self.constraints_sym[a][-1](x_ph[-1], pCa_k))
                    agent_constraint_params[a].append(pCa_k)
                else:
                    Ca[a][-1].append(self.constraints_sym[a][-1](x_ph[-1]))
            # Add agent box constraints
            if len(self.state_ub_idxs[a]) > 0:
                Ca[a][-1].append(x_ph[-1][self.state_ub_idxs[a]+int(np.sum(self.num_qa_d[:a]))] - self.state_ub[a][self.state_ub_idxs[a]])
            if len(self.state_lb_idxs[a]) > 0:
                Ca[a][-1].append(self.state_lb[a][self.state_lb_idxs[a]] - x_ph[-1][self.state_lb_idxs[a]+int(np.sum(self.num_qa_d[:a]))])
            if len(Ca[a][-1]) > 0:
                Ca[a][-1] = ca.vertcat(*Ca[a][-1])
                self.n_ca[a][-1] = Ca[a][-1].shape[0]
            else:
                Ca[a][-1] = ca.DM()
        # self.f_Cs = ca.Function('f_Cs', [ca.vertcat(*ua_ph), ca.vertcat(*x_ph), uk_ph[-1]], Cs)
        # self.f_Ca = [ca.Function(f'f_Ca{a}', [ca.vertcat(*ua_ph), ca.vertcat(*x_ph), uk_ph[-1]], Ca[a]) for a in range(self.M)]

        # Joint constraint functions for both agents: C(x, u) <= 0
        C = [[] for _ in range(self.N+1)]
        # Constraint indexes specific to each best response problem at each time step
        self.Cbr_k_idxs = [[[] for _ in range(self.N+1)] for _ in range(self.M)] 
        # Constraint indexes specific to each best response problem in batch vector form
        self.Cbr_v_idxs = [[] for _ in range(self.M)]
        for k in range(self.N+1):
            C[k].append(Cs[k])
            n = self.n_cs[k]
            for a in range(self.M):
                self.Cbr_k_idxs[a][k].append(np.arange(self.n_cs[k]))
                C[k].append(Ca[a][k])
                self.Cbr_k_idxs[a][k].append(np.arange(self.n_ca[a][k]) + n)
                n += self.n_ca[a][k]
                self.Cbr_k_idxs[a][k] = np.concatenate(self.Cbr_k_idxs[a][k]).astype(int)
                self.Cbr_v_idxs[a].append((self.Cbr_k_idxs[a][k] + np.sum(self.n_c[:k])).astype(int))
            C[k] = ca.vertcat(*C[k])    
            self.n_c[k] = C[k].shape[0]
        for a in range(self.M): self.Cbr_v_idxs[a] = np.concatenate(self.Cbr_v_idxs[a])
        # First derivatives of constraints w.r.t. input sequence
        Dx_Cxu = ca.jacobian(ca.vertcat(*C), ca.vertcat(*x_ph))
        Du_Cxu = ca.jacobian(ca.vertcat(*C), ca.vertcat(*ua_ph))
        Du_C = Du_Cxu + Dx_Cxu @ Du_x_ph
        
        # Hessian of constraints using dynamic programming
        Duu_C = []
        for k in range(self.N+1):
            for j in range(self.n_c[k]):
                Dx_Cj = [ca.jacobian(C[k][j], x_ph[k])]
                Dxx_Cj = [ca.jacobian(ca.jacobian(C[k][j], x_ph[k]), x_ph[k])]
                if k == self.N:
                    Duu_Cj, Dxu_Cj = [], []
                else:
                    Duu_Cj = [ca.jacobian(ca.jacobian(C[k][j], uk_ph[k]), uk_ph[k])]
                    Dxu_Cj = [ca.jacobian(ca.jacobian(C[k][j], uk_ph[k]), x_ph[k])]
                for t in range(k-1, -1, -1):
                    Dx_Cjt = ca.jacobian(C[k][j], x_ph[t])
                    Dxx_Cjt = ca.jacobian(ca.jacobian(C[k][j], x_ph[t]), x_ph[t])
                    Duu_Cjt = ca.jacobian(ca.jacobian(C[k][j], uk_ph[t]), uk_ph[t])
                    Dxu_Cjt = ca.jacobian(ca.jacobian(C[k][j], uk_ph[t]), x_ph[t])

                    Dx = Dx_Cjt + Dx_Cj[-1] @ A_ph[t]

                    A1 = Duu_Cjt + B_ph[t].T @ Dxx_Cj[-1] @ B_ph[t]
                    for i in range(self.n_q):
                        A1 += Dx_Cj[-1][i] * F_ph[t][i]
                    if len(Dxu_Cj) == 0:
                        Duu = A1
                    else:
                        B1 = Dxu_Cj[-1] @ B_ph[t]
                        Duu = ca.blockcat([[A1, B1.T], [B1, Duu_Cj[-1]]])

                    A2 = Dxu_Cjt + B_ph[t].T @ Dxx_Cj[-1] @ A_ph[t]
                    for i in range(self.n_q):
                        A2 += Dx_Cj[-1][i] * G_ph[t][i]
                    if len(Dxu_Cj) == 0:
                        Dxu = A2
                    else:
                        B2 = Dxu_Cj[-1] @ A_ph[t]
                        Dxu = ca.vertcat(A2, B2)

                    Dxx = Dxx_Cjt + A_ph[t].T @ Dxx_Cj[-1] @ A_ph[t]
                    for i in range(self.n_q):
                        Dxx += Dx_Cj[-1][i] * E_ph[t][i]
                    
                    Dx_Cj.append(Dx)
                    Dxx_Cj.append(Dxx)
                    Duu_Cj.append(Duu)
                    Dxu_Cj.append(Dxu)
                Duu = self.ca_sym(f'Duu_C{k}_{j}', ca.Sparsity(self.n_u*self.N, self.n_u*self.N))
                Duu[:Duu_Cj[-1].size1(),:Duu_Cj[-1].size2()] = Duu_Cj[-1]
                Duu = ca.horzcat(*[Duu[:,self.ua_idxs[a]] for a in range(self.M)])
                Duu = ca.vertcat(*[Duu[self.ua_idxs[a],:] for a in range(self.M)])
                Duu_C.append(Duu)
        
        # Paramter vector
        P = []
        for a in range(self.M):
            P += agent_cost_params[a]
        for a in range(self.M):
            P += agent_constraint_params[a]
        P += shared_constraint_params
        P = ca.vertcat(*P)

        # Cost function in sparse form
        self.f_Jxu = ca.Function('f_Jxu', [ca.vertcat(*x_ph), ca.vertcat(*ua_ph), uk_ph[-1], P], [ca.sum1(ca.vertcat(*J[a])) for a in range(self.M)])
        
        # Cost function in batch form
        Ju = self.f_Jxu(ca.vertcat(*xr_ph), ca.vertcat(*ua_ph), uk_ph[-1], P)
        self.f_J = ca.Function('f_J', [ca.vertcat(*ua_ph), xr_ph[0], uk_ph[-1], P], Ju, self.options)
        
        # First derivatives of cost function w.r.t. input sequence
        self.f_Du_J = [ca.Function(f'f_Du_J{a}', [ca.vertcat(*x_ph), ca.vertcat(*ua_ph), uk_ph[-1], Du_x_ph, P], Du_J[a]) for a in range(self.M)]
        
        q = ca.vertcat(*[Du_J[a][a] for a in range(self.M)])
        self.f_q = ca.Function('f_q', [ca.vertcat(*x_ph), ca.vertcat(*ua_ph), uk_ph[-1], Du_x_ph, P], [q], self.options)

        # Second derivatives of cost function w.r.t. input sequence
        in_args = [ca.vertcat(*x_ph), ca.vertcat(*ua_ph), uk_ph[-1]] \
                    + A_ph + B_ph \
                    + list(itertools.chain(*E_ph)) \
                    + list(itertools.chain(*F_ph)) \
                    + list(itertools.chain(*G_ph)) \
                    + [P]
        self.f_Duu_J = ca.Function('f_Duu_J', in_args, Duu_J)

        # Constraint function in sparse form
        self.f_Cxu = ca.Function('f_Cxu', [ca.vertcat(*x_ph), ca.vertcat(*ua_ph), uk_ph[-1], P], C, self.options)

        # Constraint function in batch form
        Cu = self.f_Cxu(ca.vertcat(*xr_ph), ca.vertcat(*ua_ph), uk_ph[-1], P)
        self.f_C = ca.Function('f_C', [ca.vertcat(*ua_ph), xr_ph[0], uk_ph[-1], P], Cu, self.options)

        # First derivatives of constraints w.r.t. input sequence
        self.f_Du_C = ca.Function('f_Du_C', [ca.vertcat(*x_ph), ca.vertcat(*ua_ph), uk_ph[-1], Du_x_ph, P], [Du_C], self.options)

        l_ph = self.ca_sym(f'l_ph', np.sum(self.n_c))
        lDuu_C = 0
        for j in range(np.sum(self.n_c)):
            lDuu_C += l_ph[j] * Duu_C[j]
        in_args = [ca.vertcat(*x_ph), ca.vertcat(*ua_ph), l_ph, uk_ph[-1]] \
                    + A_ph + B_ph \
                    + list(itertools.chain(*E_ph)) \
                    + list(itertools.chain(*F_ph)) \
                    + list(itertools.chain(*G_ph)) \
                    + [P]
        self.f_lDuu_C = ca.Function('f_lDuu_C', in_args, [lDuu_C])

        # Hessian of the Lagrangian
        Q = ca.vertcat(*[Duu_J[a][int(np.sum(self.num_ua_el[:a])):int(np.sum(self.num_ua_el[:a+1])),:] for a in range(self.M)]) + lDuu_C
        self.f_Q = ca.Function('f_Q', in_args, [Q])

        # Best response specific constraint functions
        # Cbr = [[[] for _ in range(self.N+1)] for _ in range(self.M)]
        # for k in range(self.N+1):
        #     for a in range(self.M):
        #         Cbr[a][k] = Cu[k][self.Cbr_k_idxs[a][k]]
        #         self.n_cbr[a][k] = Cbr[a][k].shape[0]
        # self.f_Cbr = [ca.Function(f'f_C{a}', [ca.vertcat(*ua_ph), x_ph[0], uk_ph[-1]], Cbr[a]) for a in range(self.M)]

        # Duk_C = [ca.jacobian(ca.vertcat(*Cu), ca.vertcat(*uk_ph[:-1]))]
        # Du_Cbr = [ca.jacobian(ca.vertcat(*Cbr[a]), ua_ph[a]) for a in range(self.M)]
        # self.f_Duk_C = ca.Function('f_Duk_C', [ca.vertcat(*uk_ph[:-1]), xr_ph[0], uk_ph[-1]], Duk_C)
        # self.f_Du_Cbr = [ca.Function(f'f_Du_Cbr{a}', [ca.vertcat(*ua_ph), xr_ph[0], uk_ph[-1]], [Du_Cbr[a]]) for a in range(self.M)]

        # Symbolic Hessian of Lagrangian
        L = [Ju[a] + ca.dot(l_ph, ca.vertcat(*Cu)) for a in range(self.M)]
        Du_L = [ca.jacobian(L[a], ca.vertcat(*ua_ph)).T for a in range(self.M)]
        self.f_Du_L = ca.Function('f_Du_L', [ca.vertcat(*ua_ph), l_ph, xr_ph[0], uk_ph[-1], P], Du_L)
        Duu_L = [ca.jacobian(Du_L[a], ca.vertcat(*ua_ph)) for a in range(self.M)]
        self.f_Duu_L = ca.Function('f_Duu_L', [ca.vertcat(*ua_ph), l_ph, xr_ph[0], uk_ph[-1], P], Duu_L)

        DuL = [ca.jacobian(L[a], ca.vertcat(*u_ph[a][:-1])).T for a in range(self.M)]
        F = ca.vertcat(*DuL, *Cu)
        self.F = ca.Function('F', [ca.vertcat(*ua_ph, l_ph), xr_ph[0], uk_ph[-1], P], [F])
        J = ca.jacobian(F, ca.vertcat(*ua_ph, l_ph))
        self.J = ca.Function('J', [ca.vertcat(*ua_ph, l_ph), xr_ph[0], uk_ph[-1], P], [J])
        # Q2 = ca.vertcat(*[Duu_L[a][a] for a in range(self.M)]) 
        # self.f_Q2 = ca.Function('f_Q', [ca.vertcat(*ua_ph), l_ph, xr_ph[0], uk_ph[-1]], [Q2])

        # Merit function
        du_ph = [[self.ca_sym(f'du_{a}_ph_{k}', self.joint_dynamics.dynamics_models[a].n_u) for k in range(self.N)] for a in range(self.M)] # Agent inputs
        dua_ph = [ca.vertcat(*du_ph[a]) for a in range(self.M)] # Stack input sequences by agent
        dl_ph = self.ca_sym(f'dl_ph', np.sum(self.n_c))
        s_ph = self.ca_sym(f's_ph', np.sum(self.n_c))
        ds_ph = self.ca_sym(f'ds_ph', np.sum(self.n_c))
        mu_ph = self.ca_sym('mu_ph', 1)

        q_ph = self.ca_sym('q_ph', self.n_u*self.N)
        Q_ph = self.ca_sym('Q_ph', Q.sparsity())
        g_ph = self.ca_sym('g_ph', np.sum(self.n_c))
        H_ph = self.ca_sym('H_ph', Du_C.sparsity())

        stat = ca.vertcat(q_ph + H_ph.T @ l_ph, ca.dot(l_ph, g_ph))
        stat_norm = (1/2)*ca.sumsqr(stat)
        dstat_norm = (q_ph + H_ph.T @ l_ph).T @ ca.horzcat(Q_ph, H_ph.T) @ ca.vertcat(*dua_ph, dl_ph) \
                        + ca.dot(l_ph, g_ph)*(l_ph.T @ H_ph @ ca.vertcat(*dua_ph) + ca.dot(dl_ph, g_ph))
        vio = mu_ph*ca.sum1(g_ph-s_ph)
        dvio = -mu_ph*ca.sum1(g_ph-s_ph)

        phi_args = [l_ph, s_ph, q_ph, H_ph, g_ph, mu_ph]
        dphi_args = [ca.vertcat(*dua_ph), l_ph, dl_ph, s_ph, Q_ph, q_ph, H_ph, g_ph, mu_ph]
        if self.merit_function == 'stat_l1':
            self.f_phi = ca.Function('f_phi', phi_args, [stat_norm + vio], self.options)
            self.f_dphi = ca.Function('f_dphi', dphi_args, [dstat_norm + dvio], self.options)
        elif self.merit_function == 'stat':
            self.f_phi = ca.Function('f_phi', phi_args, [stat_norm], self.options)
            self.f_dphi = ca.Function('f_dphi', dphi_args, [dstat_norm], self.options)
        else:
            raise(ValueError(f'Merit function option {self.merit_function} not recognized'))
        self.f_dstat_norm = ca.Function('f_dstat_norm', dphi_args, [dstat_norm], self.options)

        # Casadi C code generation
        if self.code_gen and not self.jit:
            generator = ca.CodeGenerator(self.c_file_name)
            generator.add(self.evaluate_dynamics)
            generator.add(self.evaluate_jacobian_A)
            generator.add(self.evaluate_jacobian_B)
            generator.add(self.evaluate_hessian_E)
            generator.add(self.evaluate_hessian_F)
            generator.add(self.evaluate_hessian_G)
            generator.add(self.f_Du_x)

            generator.add(self.f_J)

            generator.add(self.f_Cxu)
            generator.add(self.f_Du_C)
            
            generator.add(self.f_q)
            generator.add(self.f_Q)

            generator.add(self.f_phi)
            generator.add(self.f_dphi)
            generator.add(self.f_dstat_norm)

            # Set up paths
            cur_dir = pathlib.Path.cwd()
            gen_path = cur_dir.joinpath(self.solver_name)
            c_path = gen_path.joinpath(self.c_file_name)
            if gen_path.exists():
                shutil.rmtree(gen_path)
            gen_path.mkdir(parents=True)

            os.chdir(gen_path)
            if self.verbose:
                self.print_method(f'- Generating C code for solver {self.solver_name} at {str(gen_path)}')
            generator.generate()
            # Compile into shared object
            so_path = gen_path.joinpath(self.so_file_name)
            command = f'gcc -fPIC -shared -{self.opt_flag} {c_path} -o {so_path}'
            if self.verbose:
                self.print_method(f'- Compiling shared object {so_path} from {c_path}')
                self.print_method(f'- Executing "{command}"')
            # pdb.set_trace()
            os.system(command)
            # pdb.set_trace()
            # Swtich back to working directory
            os.chdir(cur_dir)
            install_dir = self.install()

            # Load solver
            self._load_solver(str(install_dir.joinpath(self.so_file_name)))

    def _load_solver(self, solver_path=None):
        if solver_path is None:
            solver_path = str(pathlib.Path(self.solver_dir, self.so_file_name).expanduser())
        if self.verbose:
            self.print_method(f'- Loading solver from {solver_path}')
        self.evaluate_dynamics = ca.external('evaluate_dynamics', solver_path)
        self.evaluate_jacobian_A = ca.external('evaluate_jacobian_A', solver_path)
        self.evaluate_jacobian_B = ca.external('evaluate_jacobian_B', solver_path)
        self.evaluate_hessian_E = ca.external('evaluate_hessian_E', solver_path)
        self.evaluate_hessian_F = ca.external('evaluate_hessian_F', solver_path)
        self.evaluate_hessian_G = ca.external('evaluate_hessian_G', solver_path)
        self.f_Du_x = ca.external('f_Du_x', solver_path)

        self.f_J = ca.external('f_J', solver_path)
        
        self.f_Cxu = ca.external('f_Cxu', solver_path)
        self.f_Du_C = ca.external('f_Du_C', solver_path)

        self.f_q = ca.external('f_q', solver_path)
        self.f_Q = ca.external('f_Q', solver_path)
        
        self.f_phi = ca.external('f_phi', solver_path)
        self.f_dphi = ca.external('f_dphi', solver_path)
        self.f_dstat_norm = ca.external('f_dstat_norm', solver_path)

    # def _line_search(self, u, du, l, dl, merit, d_merit):
    #     phi = merit(u, l)
    #     dphi = d_merit(u, du, l, dl)
    #     if dphi > 0:
    #         if self.verbose:
    #             self.print_method(f'- Line search directional derivative is positive: {dphi}')
    #     alpha, conv = 1.0, False
    #     for i in range(self.line_search_iters):
    #         u_trial = u + alpha*du
    #         l_trial = l + alpha*dl
    #         phi_trial = merit(u_trial, l_trial)
    #         if self.verbose:
    #             self.print_method(f'- Line search iteration: {i} | merit gap: {phi_trial-(phi + self.beta*alpha*dphi):.4e} | a: {alpha:.4e}')
    #         if phi_trial <= phi + self.beta*alpha*dphi:
    #             conv = True
    #             break
    #         else:
    #             alpha *= self.tau
    #     if not conv:
    #         if self.verbose:
    #             self.print_method('- Max iterations reached, line search did not succeed')
    #         # pdb.set_trace()
    #     return u_trial, l_trial, phi_trial

    # def _watchdog_line_search(self, u_k, du_k, l_k, dl_k, x0, up, merit, d_merit):
    #     if self.verbose:
    #         self.print_method('===================================================')
    #         self.print_method('Watchdog step acceptance routine')
    #     t_hat = 7 # Number of steps where we search for sufficient merit decrease
    #     phi_log = []

    #     phi_k = merit(u_k, l_k)
    #     phi_log.append(phi_k)
    #     dphi_k = d_merit(u_k, du_k, l_k, dl_k)
    #     if dphi_k > 0:
    #         if self.verbose:
    #             self.print_method(f'Time k: Directional derivative is positive: {dphi_k}')

    #     # Take relaxed (full) step
    #     u_kp1 = u_k + du_k
    #     l_kp1 = l_k + dl_k
    #     phi_kp1 = merit(u_kp1, l_kp1)
    #     phi_log.append(phi_kp1)

    #     # Check for sufficient decrease w.r.t. time k
    #     if self.verbose:
    #         self.print_method(f'Time k+1:')
    #     if phi_kp1 <= phi_k + self.beta*dphi_k:
    #         if self.verbose:
    #             self.print_method(f'Sufficient decrease achieved')
    #         return u_kp1, l_kp1
    #     if self.verbose:
    #         self.print_method(f'Insufficient decrease in merit')

    #     # Check for sufficient decrease in the next t_hat steps
    #     u_t, l_t, phi_t = u_kp1, l_kp1, phi_kp1
    #     for t in range(t_hat):
    #         if self.verbose:
    #             self.print_method(f'Time k+{t+2}:')
    #         # Compute step at time t
    #         Q_t, q_t, G_t, g_t, _ = self._evaluate(u_t, l_t, x0, up)
    #         du_t, l_hat = self._solve_qp(Q_t, q_t, G_t, g_t)
    #         dl_t = l_hat - l_t

    #         # Do line search
    #         u_tp1, l_tp1, phi_tp1 = self._line_search(u_t, du_t, l_t, dl_t, merit, d_merit)
    #         phi_log.append(phi_tp1)
    #         if self.verbose:
    #             self.print_method(phi_log)
            
    #         # Check for sufficient decrease w.r.t. time 0
    #         if phi_t <= phi_k or phi_tp1 <= phi_k + self.beta*dphi_k:
    #             if self.verbose:
    #                 self.print_method(f'Sufficient decrease achieved')
    #             return u_tp1, l_tp1
            
    #         # Update for next time step
    #         u_t, l_t, phi_t = u_tp1, l_tp1, phi_tp1
        
    #     if phi_tp1 > phi_k:
    #         if self.verbose:
    #             self.print_method(f'No decrease in merit, returning to search along step at time k')
    #         u_kp1, l_kp1, phi_kp1 = self._line_search(u_k, du_k, l_k, dl_k, merit, d_merit)
    #         return u_kp1, l_kp1
    #     else:
    #         if self.verbose:
    #             self.print_method(f'Insufficient decrease in merit')
    #         Q_tp1, q_tp1, G_tp1, g_tp1, _ = self._evaluate(u_tp1, l_tp1, x0, up)
    #         du_tp1, l_hat = self._solve_qp(Q_tp1, q_tp1, G_tp1, g_tp1)
    #         dl_tp1 = l_hat - l_tp1

    #         u_tp2, l_tp2, phi_tp2 = self._line_search(u_tp1, du_tp1, l_tp1, dl_tp1, merit, d_merit)
    #         return u_tp2, l_tp2

    # def _line_search_2(self, u, du, l, dl, s, ds, merit, d_merit):
    #     phi = merit(u, l, s)
    #     dphi = d_merit(u, du, l, dl, s, ds)
    #     if dphi > 0:
    #         if self.verbose:
    #             self.print_method(f'- Line search directional derivative is positive: {dphi}')
    #     alpha, conv = 1.0, False
    #     for i in range(self.line_search_iters):
    #         u_trial = u + alpha*du
    #         l_trial = l + alpha*dl
    #         s_trial = s + alpha*ds
    #         phi_trial = merit(u_trial, l_trial, s_trial)
    #         # dphi_trial = d_merit(u_trial, du, l_trial, dl, s_trial, ds)
    #         if self.verbose:
    #             self.print_method(f'- Line search iteration: {i} | merit gap: {phi_trial-(phi + self.beta*alpha*dphi):.4e} | a: {alpha:.4e}')
    #         # if phi_trial <= phi + self.beta*alpha*dphi and dphi_trial >= 2*self.beta*dphi:
    #         if phi_trial <= phi + self.beta*alpha*dphi:
    #             conv = True
    #             break
    #         else:
    #             alpha *= self.tau
    #     if not conv:
    #         if self.verbose:
    #             self.print_method('- Max iterations reached, line search did not succeed')
    #         # pdb.set_trace()
    #     return u_trial, l_trial, phi_trial

    # def _watchdog_line_search_2(self, u_k, du_k, l_k, dl_k, s_k, ds_k, x0, up, merit, d_merit):
    #     if self.verbose:
    #         self.print_method('===================================================')
    #         self.print_method('Watchdog step acceptance routine')
    #     qp_solves = 0
    #     t_hat = 7 # Number of steps where we search for sufficient merit decrease
    #     phi_log = []

    #     phi_k = merit(u_k, l_k, s_k)
    #     phi_log.append(phi_k)
    #     dphi_k = d_merit(u_k, du_k, l_k, dl_k, s_k, ds_k)
    #     if dphi_k > 0:
    #         if self.verbose:
    #             self.print_method(f'Time k: Directional derivative is positive: {dphi_k}')

    #     # Take relaxed (full) step
    #     u_kp1 = u_k + du_k
    #     l_kp1 = l_k + dl_k
    #     s_kp1 = s_k + ds_k
    #     phi_kp1 = merit(u_kp1, l_kp1, s_kp1)
    #     phi_log.append(phi_kp1)

    #     # Check for sufficient decrease w.r.t. time k
    #     if self.verbose:
    #         self.print_method(f'Time k+1:')
    #     if phi_kp1 <= phi_k + self.beta*dphi_k:
    #         if self.verbose:
    #             self.print_method(f'Sufficient decrease achieved')
    #         return u_kp1, l_kp1, qp_solves
    #     if self.verbose:
    #         self.print_method(f'Insufficient decrease in merit')

    #     # Check for sufficient decrease in the next t_hat steps
    #     u_t, l_t, phi_t = u_kp1, l_kp1, phi_kp1
    #     for t in range(t_hat):
    #         if self.verbose:
    #             self.print_method(f'Time k+{t+2}:')
    #         # Compute step at time t
    #         Q_t, q_t, G_t, g_t, _ = self._evaluate(u_t, l_t, x0, up)
    #         du_t, l_hat = self._solve_qp(Q_t, q_t, G_t, g_t)
    #         qp_solves += 1
    #         if None in du_t:
    #             if self.verbose:
    #                 self.print_method(f'QP failed, returning to search along step at time k')
    #             u_kp1, l_kp1, phi_kp1 = self._line_search_2(u_k, du_k, l_k, dl_k, s_k, ds_k, merit, d_merit)
    #             return u_kp1, l_kp1, qp_solves
    #         dl_t = l_hat - l_t

    #         s_t = np.minimum(1e-7, g_t)
    #         ds_t = g_t + G_t @ du_t - s_t

    #         # Do line search
    #         u_tp1, l_tp1, phi_tp1 = self._line_search_2(u_t, du_t, l_t, dl_t, s_t, ds_t, merit, d_merit)
    #         phi_log.append(phi_tp1)
    #         if self.verbose:
    #             self.print_method(phi_log)
            
    #         # Check for sufficient decrease w.r.t. time 0
    #         if phi_t <= phi_k or phi_tp1 <= phi_k + self.beta*dphi_k:
    #             if self.verbose:
    #                 self.print_method(f'Sufficient decrease achieved')
    #             return u_tp1, l_tp1, qp_solves
            
    #         # Update for next time step
    #         u_t, l_t, phi_t = u_tp1, l_tp1, phi_tp1
        
    #     if phi_tp1 > phi_k:
    #         if self.verbose:
    #             self.print_method(f'No decrease in merit, returning to search along step at time k')
    #         u_kp1, l_kp1, phi_kp1 = self._line_search_2(u_k, du_k, l_k, dl_k, s_k, ds_k, merit, d_merit)
    #         return u_kp1, l_kp1, qp_solves
    #     else:
    #         if self.verbose:
    #             self.print_method(f'Insufficient decrease in merit')
    #         Q_tp1, q_tp1, G_tp1, g_tp1, _ = self._evaluate(u_tp1, l_tp1, x0, up)
    #         du_tp1, l_hat = self._solve_qp(Q_tp1, q_tp1, G_tp1, g_tp1)
    #         if None in du_t:
    #             if self.verbose:
    #                 self.print_method(f'QP failed, returning to search along step at time k')
    #             u_kp1, l_kp1, phi_kp1 = self._line_search_2(u_k, du_k, l_k, dl_k, s_k, ds_k, merit, d_merit)
    #             return u_kp1, l_kp1, qp_solves
    #         qp_solves += 1
    #         dl_tp1 = l_hat - l_tp1

    #         s_tp1 = np.minimum(1e-7, g_tp1)
    #         ds_tp1 = g_tp1 + G_tp1 @ du_tp1 - s_tp1

    #         u_tp2, l_tp2, phi_tp2 = self._line_search_2(u_tp1, du_tp1, l_tp1, dl_tp1, s_tp1, ds_tp1, merit, d_merit)
    #         return u_tp2, l_tp2, qp_solves

    def _line_search_3(self, u, du, l, dl, s, ds, Q, q, G, g, evaluate, merit, d_merit):
        phi = merit(u, l, s, q, G, g)
        dphi = d_merit(u, du, l, dl, s, ds, Q, q, G, g)
        if dphi > 0:
            if self.verbose:
                self.print_method(f'- Line search directional derivative is positive: {dphi}')
        alpha, conv = 1.0, False
        for i in range(self.line_search_iters):
            u_trial = u + alpha*du
            l_trial = l + alpha*dl
            s_trial = s + alpha*ds
            q_trial, G_trial, g_trial, _ = evaluate(u_trial, l_trial, False)
            phi_trial = merit(u_trial, l_trial, s_trial, q_trial, G_trial, g_trial)
            if self.verbose:
                self.print_method(f'- Line search iteration: {i} | merit gap: {phi_trial-(phi + self.beta*alpha*dphi):.4e} | a: {alpha:.4e}')
            if phi_trial <= phi + self.beta*alpha*dphi:
                conv = True
                break
            else:
                alpha *= self.tau
        if not conv:
            if self.verbose:
                self.print_method('- Max iterations reached, line search did not succeed')
            # pdb.set_trace()
        return u_trial, l_trial, phi_trial

    # def _watchdog_line_search_3(self, u_k, du_k, l_k, dl_k, s_k, ds_k, Q_k, q_k, G_k, g_k, evaluate, merit, d_merit):
    #     if self.verbose:
    #         self.print_method('===================================================')
    #         self.print_method('Watchdog step acceptance routine')
    #     qp_solves = 0
    #     t_hat = 7 # Number of steps where we search for sufficient merit decrease
    #     phi_log = []

    #     phi_k = merit(u_k, l_k, s_k, q_k, G_k, g_k)
    #     phi_log.append(phi_k)
    #     dphi_k = d_merit(u_k, du_k, l_k, dl_k, s_k, ds_k, Q_k, q_k, G_k, g_k)
    #     if dphi_k > 0:
    #         if self.verbose:
    #             self.print_method(f'Time k: Directional derivative is positive: {dphi_k}')

    #     # Take relaxed (full) step
    #     u_kp1 = u_k + du_k
    #     l_kp1 = l_k + dl_k
    #     s_kp1 = s_k + ds_k
    #     q_kp1, G_kp1, g_kp1, _ = evaluate(u_kp1, l_kp1, False)
    #     phi_kp1 = merit(u_kp1, l_kp1, s_kp1, q_kp1, G_kp1, g_kp1)
    #     phi_log.append(phi_kp1)

    #     # Check for sufficient decrease w.r.t. time k
    #     if self.verbose:
    #         self.print_method(f'Time k+1:')
    #     if phi_kp1 <= phi_k + self.beta*dphi_k:
    #         if self.verbose:
    #             self.print_method(f'Sufficient decrease achieved')
    #         return u_kp1, l_kp1, qp_solves
    #     if self.verbose:
    #         self.print_method(f'Insufficient decrease in merit')

    #     # Check for sufficient decrease in the next t_hat iterations
    #     u_t, l_t, phi_t = u_kp1, l_kp1, phi_kp1
    #     for t in range(t_hat):
    #         if self.verbose:
    #             self.print_method(f'Time k+{t+2}:')
    #         # Compute step at iteration t
    #         Q_t, q_t, G_t, g_t, _ = evaluate(u_t, l_t, True)
    #         du_t, l_hat = self._solve_qp(Q_t, q_t, G_t, g_t)
    #         qp_solves += 1
    #         if None in du_t:
    #             if self.verbose:
    #                 self.print_method(f'QP failed, returning to search along step at iteration k')
    #             u_kp1, l_kp1, phi_kp1 = self._line_search_3(u_k, du_k, l_k, dl_k, s_k, ds_k, Q_k, q_k, G_k, g_k, evaluate, merit, d_merit)
    #             return u_kp1, l_kp1, qp_solves
    #         dl_t = l_hat - l_t

    #         s_t = np.minimum(0, g_t)
    #         ds_t = g_t + G_t @ du_t - s_t

    #         # Do line search
    #         u_tp1, l_tp1, phi_tp1 = self._line_search_3(u_t, du_t, l_t, dl_t, s_t, ds_t, Q_t, q_t, G_t, g_t, evaluate, merit, d_merit)
    #         phi_log.append(phi_tp1)
    #         if self.verbose:
    #             self.print_method(phi_log)
            
    #         # Check for sufficient decrease w.r.t. base iteration
    #         if phi_t <= phi_k or phi_tp1 <= phi_k + self.beta*dphi_k:
    #             if self.verbose:
    #                 self.print_method(f'Sufficient decrease achieved')
    #             return u_tp1, l_tp1, qp_solves
            
    #         # Update for next iteration
    #         u_t, l_t, phi_t = u_tp1, l_tp1, phi_tp1
        
    #     if phi_tp1 > phi_k:
    #         if self.verbose:
    #             self.print_method(f'No decrease in merit, returning to search along step at iteration k')
    #         u_kp1, l_kp1, phi_kp1 = self._line_search_3(u_k, du_k, l_k, dl_k, s_k, ds_k, Q_k, q_k, G_k, g_k, evaluate, merit, d_merit)
    #         return u_kp1, l_kp1, qp_solves
    #     else:
    #         if self.verbose:
    #             self.print_method(f'Insufficient decrease in merit')
    #         Q_tp1, q_tp1, G_tp1, g_tp1, _ = evaluate(u_tp1, l_tp1, True)
    #         du_tp1, l_hat = self._solve_qp(Q_tp1, q_tp1, G_tp1, g_tp1)
    #         if None in du_tp1:
    #             if self.verbose:
    #                 self.print_method(f'QP failed, returning to search along step at time k')
    #             u_kp1, l_kp1, phi_kp1 = self._line_search_3(u_k, du_k, l_k, dl_k, s_k, ds_k, Q_k, q_k, G_k, g_k, evaluate, merit, d_merit)
    #             return u_kp1, l_kp1, qp_solves
    #         qp_solves += 1
    #         dl_tp1 = l_hat - l_tp1

    #         s_tp1 = np.minimum(0, g_tp1)
    #         ds_tp1 = g_tp1 + G_tp1 @ du_tp1 - s_tp1

    #         u_tp2, l_tp2, phi_tp2 = self._line_search_3(u_tp1, du_tp1, l_tp1, dl_tp1, s_tp1, ds_tp1, Q_tp1, q_tp1, G_tp1, g_tp1, evaluate, merit, d_merit)
    #         return u_tp2, l_tp2, qp_solves

    def _watchdog_line_search_4(self, u_k, du_k, l_k, dl_k, s_k, ds_k, Q_k, q_k, G_k, g_k, evaluate, merit, d_merit, merit_max=1e6):
        if self.verbose:
            self.print_method('===================================================')
            self.print_method('Watchdog step acceptance routine')
        qp_solves = 0
        t_hat = 5 # Number of steps where we search for sufficient merit decrease
        phi_log = []

        phi_k = merit(u_k, l_k, s_k, q_k, G_k, g_k)
        phi_log.append(phi_k)
        dphi_k = d_merit(u_k, du_k, l_k, dl_k, s_k, ds_k, Q_k, q_k, G_k, g_k)
        if dphi_k > 0:
            if self.verbose:
                self.print_method(f'Time k: Directional derivative is positive: {dphi_k}')

        # Take relaxed (full) step
        u_kp1 = u_k + du_k
        l_kp1 = l_k + dl_k
        s_kp1 = s_k + ds_k
        q_kp1, G_kp1, g_kp1, _ = evaluate(u_kp1, l_kp1, False)
        phi_kp1 = merit(u_kp1, l_kp1, s_kp1, q_kp1, G_kp1, g_kp1)
        phi_log.append(phi_kp1)

        # Check for sufficient decrease w.r.t. time k
        if self.verbose:
            self.print_method(f'Time k+1:')
        if phi_kp1 <= phi_k + self.beta*dphi_k:
            if self.verbose:
                self.print_method(f'Sufficient decrease achieved')
            return u_kp1, l_kp1, qp_solves
        # if self.verbose:
        #     self.print_method(f'Insufficient decrease in merit')

        start_time = time.time()
        fail = False
        # Check for sufficient decrease in the next t_hat iterations
        u_t, l_t = u_kp1, l_kp1
        for t in range(t_hat):
            if self.verbose:
                self.print_method(f'Time k+{t+2}:')
            # Compute step at iteration t
            Q_t, q_t, G_t, g_t, _ = evaluate(u_t, l_t, True)
            du_t, l_hat = self._solve_qp(Q_t, q_t, G_t, g_t)
            qp_solves += 1
            if None in du_t:
                fail = True
                break
            dl_t = l_hat - l_t
            s_t = np.minimum(0, g_t)
            ds_t = g_t + G_t @ du_t - s_t

            # Take full step
            u_tp1 = u_t + du_t
            l_tp1 = l_hat
            s_tp1 = s_t + ds_t
            q_tp1, G_tp1, g_tp1, _ = evaluate(u_tp1, l_tp1, False)
            phi_tp1 = merit(u_tp1, l_tp1, s_tp1, q_tp1, G_tp1, g_tp1)
            phi_log.append(phi_tp1)
            if self.verbose:
                self.print_method(phi_log)
            if phi_tp1 > merit_max: break
            
            # Check for sufficient decrease w.r.t. base iteration
            if phi_tp1 <= phi_k + self.beta*dphi_k:
                if self.verbose:
                    self.print_method(f'Sufficient decrease achieved')
                return u_tp1, l_tp1, qp_solves
            
            # Update for next iteration
            u_t, l_t = u_tp1, l_tp1

            if time.time() - start_time > self.time_limit:
                fail = True
                break
        
        # Insist on merit function decrease
        Q_t, q_t, G_t, g_t, _ = evaluate(u_t, l_t, True)
        du_t, l_hat = self._solve_qp(Q_t, q_t, G_t, g_t)
        qp_solves += 1
        if None in du_t:
            fail = True
        else:
            dl_t = l_hat - l_t
            s_t = np.minimum(0, g_t)
            ds_t = g_t + G_t @ du_t - s_t
            u_tp1, l_tp1, phi_tp1 = self._line_search_3(u_t, du_t, l_t, dl_t, s_t, ds_t, Q_t, q_t, G_t, g_t, evaluate, merit, d_merit)

        if not fail:
            if phi_tp1 <= phi_k + self.beta*dphi_k:
                return u_tp1, l_tp1, qp_solves
            elif phi_tp1 > phi_k:
                fail = True
            else:
                if self.verbose:
                    self.print_method(f'Insufficient decrease in merit')
                Q_tp1, q_tp1, G_tp1, g_tp1, _ = evaluate(u_tp1, l_tp1, True)
                du_tp1, l_hat = self._solve_qp(Q_tp1, q_tp1, G_tp1, g_tp1)
                if None in du_tp1:
                    if self.verbose:
                        self.print_method(f'QP failed, returning to search along step at time k')
                    u_kp1, l_kp1, phi_kp1 = self._line_search_3(u_k, du_k, l_k, dl_k, s_k, ds_k, Q_k, q_k, G_k, g_k, evaluate, merit, d_merit)
                    return u_kp1, l_kp1, qp_solves
                qp_solves += 1
                dl_tp1 = l_hat - l_tp1
                s_tp1 = np.minimum(0, g_tp1)
                ds_tp1 = g_tp1 + G_tp1 @ du_tp1 - s_tp1

                u_tp2, l_tp2, _ = self._line_search_3(u_tp1, du_tp1, l_tp1, dl_tp1, s_tp1, ds_tp1, Q_tp1, q_tp1, G_tp1, g_tp1, evaluate, merit, d_merit)
                return u_tp2, l_tp2, qp_solves

        if fail:
            if self.verbose:
                self.print_method(f'No decrease in merit, returning to search along step at iteration k')
            u_kp1, l_kp1, phi_kp1 = self._line_search_3(u_k, du_k, l_k, dl_k, s_k, ds_k, Q_k, q_k, G_k, g_k, evaluate, merit, d_merit)
            return u_kp1, l_kp1, qp_solves

    # def _watchdog_line_search_4(self, u_k, du_k, l_k, dl_k, Q_k, q_k, G_k, g_k, evaluate, merit, d_merit):
    #     if self.verbose:
    #         self.print_method('===================================================')
    #         self.print_method('Watchdog step acceptance routine')
    #     qp_solves = 0
    #     t_hat = 7 # Number of steps where we search for sufficient merit decrease
    #     phi_log = []

    #     s_k = np.minimum(0, g_k)
    #     ds_k = g_k + G_k @ du_k - s_k
    #     mu_k = self._get_mu(u_k, du_k, l_k, dl_k, s_k, ds_k, Q_k, q_k, G_k, g_k)
    #     m_k = lambda u, l, s, q, G, g: merit(u, l, s, q, G, g, mu_k)
    #     dm_k = lambda u, du, l, dl, s, ds, Q, q, G, g: d_merit(u, du, l, dl, s, ds, Q, q, G, g, mu_k)

    #     phi_k = m_k(u_k, l_k, s_k, q_k, G_k, g_k)
    #     phi_log.append(phi_k)
    #     dphi_k = dm_k(u_k, du_k, l_k, dl_k, s_k, ds_k, Q_k, q_k, G_k, g_k)
    #     if dphi_k > 0:
    #         if self.verbose:
    #             self.print_method(f'Time k: Directional derivative is positive: {dphi_k}')

    #     # Take relaxed (full) step
    #     u_kp1 = u_k + du_k
    #     l_kp1 = l_k + dl_k
    #     s_kp1 = s_k + ds_k
    #     q_kp1, G_kp1, g_kp1, _ = evaluate(u_kp1, l_kp1, False)
    #     phi_kp1 = m_k(u_kp1, l_kp1, s_kp1, q_kp1, G_kp1, g_kp1)
    #     phi_log.append(phi_kp1)

    #     # Check for sufficient decrease w.r.t. time k
    #     if self.verbose:
    #         self.print_method(f'Time k+1:')
    #     if phi_kp1 <= phi_k + self.beta*dphi_k:
    #         if self.verbose:
    #             self.print_method(f'Sufficient decrease achieved')
    #         return u_kp1, l_kp1, qp_solves
    #     if self.verbose:
    #         self.print_method(f'Insufficient decrease in merit')

    #     # Check for sufficient decrease in the next t_hat iterations
    #     u_t, l_t, phi_t = u_kp1, l_kp1, phi_kp1
    #     for t in range(t_hat):
    #         if self.verbose:
    #             self.print_method(f'Time k+{t+2}:')
    #         # Compute step at iteration t
    #         Q_t, q_t, G_t, g_t, _ = evaluate(u_t, l_t, True)
    #         du_t, l_hat = self._solve_qp(Q_t, q_t, G_t, g_t)
    #         qp_solves += 1
    #         if du_t is None:
    #             if self.verbose:
    #                 self.print_method(f'QP failed, returning to search along step at iteration k')
    #             u_kp1, l_kp1, phi_kp1 = self._line_search_3(u_k, du_k, l_k, dl_k, s_k, ds_k, Q_k, q_k, G_k, g_k, evaluate, m_k, dm_k)
    #             return u_kp1, l_kp1, qp_solves
    #         dl_t = l_hat - l_t

    #         s_t = np.minimum(0, g_t)
    #         ds_t = g_t + G_t @ du_t - s_t
    #         mu_t = self._get_mu(u_t, du_t, l_t, dl_t, s_t, ds_t, Q_t, q_t, G_t, g_t)
    #         m_t = lambda u, l, s, q, G, g: merit(u, l, s, q, G, g, mu_t)
    #         dm_t = lambda u, du, l, dl, s, ds, Q, q, G, g: d_merit(u, du, l, dl, s, ds, Q, q, G, g, mu_t)

    #         # Do line search
    #         u_tp1, l_tp1, phi_tp1 = self._line_search_3(u_t, du_t, l_t, dl_t, s_t, ds_t, Q_t, q_t, G_t, g_t, evaluate, m_t, dm_t)
    #         phi_log.append(phi_tp1)
    #         if self.verbose:
    #             self.print_method(phi_log)
            
    #         # Check for sufficient decrease w.r.t. base iteration
    #         if phi_t <= phi_k or phi_tp1 <= phi_k + self.beta*dphi_k:
    #             if self.verbose:
    #                 self.print_method(f'Sufficient decrease achieved')
    #             return u_tp1, l_tp1, qp_solves
            
    #         # Update for next iteration
    #         u_t, l_t, phi_t = u_tp1, l_tp1, phi_tp1
        
    #     if phi_tp1 > phi_k:
    #         if self.verbose:
    #             self.print_method(f'No decrease in merit, returning to search along step at iteration k')
    #         u_kp1, l_kp1, phi_kp1 = self._line_search_3(u_k, du_k, l_k, dl_k, s_k, ds_k, Q_k, q_k, G_k, g_k, evaluate, m_k, dm_k)
    #         return u_kp1, l_kp1, qp_solves
    #     else:
    #         if self.verbose:
    #             self.print_method(f'Insufficient decrease in merit')
    #         Q_tp1, q_tp1, G_tp1, g_tp1, _ = evaluate(u_tp1, l_tp1, True)
    #         du_tp1, l_hat = self._solve_qp(Q_tp1, q_tp1, G_tp1, g_tp1)
    #         qp_solves += 1
    #         if du_tp1 is None:
    #             if self.verbose:
    #                 self.print_method(f'QP failed, returning to search along step at time k')
    #             u_kp1, l_kp1, phi_kp1 = self._line_search_3(u_k, du_k, l_k, dl_k, s_k, ds_k, Q_k, q_k, G_k, g_k, evaluate, m_k, dm_k)
    #             return u_kp1, l_kp1, qp_solves
    #         dl_tp1 = l_hat - l_tp1

    #         s_tp1 = np.minimum(0, g_tp1)
    #         ds_tp1 = g_tp1 + G_tp1 @ du_tp1 - s_tp1
    #         mu_tp1 = self._get_mu(u_tp1, du_tp1, l_tp1, dl_tp1, s_tp1, ds_tp1, Q_tp1, q_tp1, G_tp1, g_tp1)
    #         m_tp1 = lambda u, l, s, q, G, g: merit(u, l, s, q, G, g, mu_tp1)
    #         dm_tp1 = lambda u, du, l, dl, s, ds, Q, q, G, g: d_merit(u, du, l, dl, s, ds, Q, q, G, g, mu_tp1)

    #         u_tp2, l_tp2, phi_tp2 = self._line_search_3(u_tp1, du_tp1, l_tp1, dl_tp1, s_tp1, ds_tp1, Q_tp1, q_tp1, G_tp1, g_tp1, evaluate, m_tp1, dm_tp1)
    #         return u_tp2, l_tp2, qp_solves

    def _nearestPD(self, A):
        B = (A + A.T)/2
        s, U = np.linalg.eigh(B)
        # s[np.where(s < 0)[0]] = 0
        s[np.where(s < 0)[0]] = 1e-10
        C = U @ np.diag(s) @ U.T
        return (C + C.T)/2

    def _update_debug_plot(self, u, x0, up, P=np.array([])):
        q_bar = np.array(self.evaluate_dynamics(u, x0)).squeeze()
        ua_bar = []
        for a in range(self.M):
            si = int(np.sum(self.num_ua_el[:a]))
            ei = int(np.sum(self.num_ua_el[:a])+self.num_ua_el[a])
            ua_bar.append(u[si:ei].reshape((self.N, self.num_ua_d[a])))
        u_bar = np.hstack(ua_bar)
        if not self.local_pos:
            for i in range(self.M):
                self.l_xy[i].set_data(q_bar[:,0+int(np.sum(self.num_qa_d[:i]))], q_bar[:,1+int(np.sum(self.num_qa_d[:i]))])
        else:
            for i in range(self.M):
                if i < self.M-1:
                    idx = int(np.sum(self.num_qa_d[i+1:]))
                else:
                    idx = 0
                s = q_bar[:,-2-idx]
                ey = q_bar[:,-1-idx]
                x, y = [], []
                for _s, _ey in zip(s, ey):
                    _x, _y, _ = self.joint_dynamics.dynamics_models[0].track.local_to_global((_s, _ey, 0))
                    x.append(_x)
                    y.append(_y)
                self.l_xy[i].set_data(x, y)
        self.ax_xy.set_aspect('equal')
        self.ax_xy.relim()
        self.ax_xy.autoscale_view()
        J = self.f_J(u, x0, up, P)
        self.ax_xy.set_title(self.solver_name + ': ' + str(J))
        for i in range(self.M):
            self.l_a[i].set_data(np.arange(self.N), u_bar[:,0+int(np.sum(self.num_ua_d[:i]))])
            self.l_s[i].set_data(np.arange(self.N), u_bar[:,1+int(np.sum(self.num_ua_d[:i]))])
        self.ax_a.relim()
        self.ax_a.autoscale_view()
        self.ax_s.relim()
        self.ax_s.autoscale_view()
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    # def _filter_line_search(self, u, du, l, dl, F, cost_fn, constr_fn):
    #     line_search_converged = False
    #     beta = 1e-5
    #     C_min = 1e-6
        
    #     C = np.mean(np.maximum(constr_fn(u), 0))

    #     alpha = 1.0
    #     for k in range(self.line_search_iters):
    #         u_trial = u + alpha*du
    #         l_trial = l + alpha*dl

    #         J_trial = list(map(float, cost_fn(u_trial))) # Agent costs
    #         C_trial = np.mean(np.maximum(constr_fn(u_trial), 0)) # Constraint violation
    #         f_trial = np.array(J_trial + [C_trial])

    #         if self.verbose:
    #                 print(f'- Line search iteration: {k} | a: {alpha:.4e}')
    #                 print(f_trial)
    #                 print(F)

    #         trial_accepted = False
    #         for f in F:
    #             if C <= C_min:
    #                 if np.any(f_trial[:-1] <= f[:-1] - beta*C):
    #                     trial_accepted = True
    #                     break
    #             else:
    #                 if np.any(f_trial <= f - beta*C):
    #                     trial_accepted = True
    #                     break
            
    #         if trial_accepted:
    #             # Remove dominated points
    #             idx = 0
    #             while idx < len(F):
    #                 if np.all(f_trial < F[idx]):
    #                     F.pop(idx)
    #                 else:
    #                     idx += 1
    #             F.append(f_trial)
    #             line_search_converged = True
    #             break 
    #         else:
    #             alpha *= self.tau
            
    #     if not line_search_converged:
    #         if self.verbose:
    #             print('- Max line search iterations reached, did not converge')

    #     return u_trial, l_trial, F

    # def _hybrid_line_search(self, u, du, l, dl, F, merit_fn, d_merit_fn, cost_fn, constr_fn):
    #     trial_accepted = False
        
    #     C_min = 1e-6
    #     g = 1e-5
    #     s_mer = 2.3
    #     s_con = 1.1
    #     delta = 1
        
    #     C = np.mean(np.maximum(constr_fn(u), 0))
    #     phi = merit_fn(u, l)
    #     dphi = d_merit_fn(u, du, l, dl)
        
    #     alpha = 1.0
    #     for k in range(self.line_search_iters):
    #         u_trial = u + alpha*du
    #         l_trial = l + alpha*dl

    #         J_trial = list(map(float, cost_fn(u_trial))) # Agent costs
    #         C_trial = np.mean(np.maximum(constr_fn(u_trial), 0)) # Constraint violation
    #         f_trial = np.array(J_trial + [C_trial])

    #         print(f_trial)
    #         print(F)
    #         filter_accepted = True
    #         for f in F:
    #             if np.all(f_trial >= f - g*C):
    #                 if self.verbose:
    #                     print('Rejected by filter')
    #                 filter_accepted = False
    #                 break
    #         if filter_accepted and self.verbose:
    #                 print('Accepted by filter')
            
    #         if filter_accepted:
    #             if self.verbose:
    #                 print(C)
    #                 print(dphi)
    #                 print(alpha*np.power(-dphi, s_mer))
    #                 print(delta*np.power(C, s_con))
    #             if C <= C_min and dphi < 0 and alpha*np.power(-dphi, s_mer) > delta*np.power(C, s_con):
    #                 if self.verbose:
    #                     print('Case 1')
    #                 phi_trial = merit_fn(u_trial, l_trial)
    #                 if phi_trial <= phi + self.beta*alpha*dphi:
    #                     if self.verbose:
    #                         print('Trial accepted via line search')
    #                     trial_accepted = True
    #             else:
    #                 if self.verbose:
    #                     print('Case 2')
    #                 for f in F:
    #                     if np.any(f_trial <= f - g*C):
    #                         if self.verbose:
    #                             print('Trial accepted via filter')
    #                         trial_accepted = True
    #                         break

    #         if filter_accepted and trial_accepted:
    #             # Remove dominated points
    #             idx = 0
    #             while idx < len(F):
    #                 if np.all(f_trial <= F[idx]):
    #                     F.pop(idx)
    #                 else:
    #                     idx += 1
    #             F.append(f_trial)
            
    #         if trial_accepted:
    #             break
    #         else:
    #             alpha *= self.tau
            
    #     if not trial_accepted:
    #         if self.verbose:
    #             print('- Max line search iterations reached, did not converge')

    #     return u_trial, l_trial, F

if __name__ == '__main__':
    pass
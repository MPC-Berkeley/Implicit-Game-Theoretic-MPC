#!/usr/bin python3

import time, copy
from typing import List, Dict
from collections import deque
import pdb
import os

import numpy as np
import scipy as sp

import casadi as ca
import cvxpy as cp

from mpclab_common.models.dynamics_models import CasadiDynamicsModel
from mpclab_common.pytypes import VehicleState, VehiclePrediction

from mpclab_controllers.abstract_controller import AbstractController
from mpclab_controllers.utils.controllerTypes import CALMPCParams

# import matplotlib
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

class CA_LMPC(AbstractController):
    def __init__(self, dynamics: CasadiDynamicsModel, 
                       costs: Dict[str, List[ca.Function]], 
                       constraints: Dict[str, ca.Function],
                       bounds: Dict[str, VehicleState], 
                       control_params=CALMPCParams(),
                       print_method=print):
        self.dynamics       = dynamics
        self.track          = dynamics.track
        self.costs          = costs
        self.constraints    = constraints
        self.print_method   = print_method

        self.n_u            = self.dynamics.n_u
        self.n_q            = self.dynamics.n_q
        self.n_z            = self.n_q + self.n_u

        self.verbose        = control_params.verbose

        self.qp_interface   = control_params.qp_interface
        self.qp_solver      = control_params.qp_solver

        self.options        = dict()
        self.code_gen       = control_params.code_gen
        self.jit            = control_params.jit
        self.opt_flag       = control_params.opt_flag
        if self.code_gen and self.jit:
            self.options = dict(jit=True, jit_name='LMPC', compiler='shell', jit_options=dict(compiler='gcc', flags=['-%s' % self.opt_flag], verbose=self.verbose))

        self.dt             = control_params.dt
        self.N              = control_params.N

        self.damping        = control_params.damping
        self.qp_iters       = control_params.qp_iters

        self.keep_init_safe_set = control_params.keep_init_safe_set

        self.delay          = control_params.delay
        self.delay_buffer   = []
        if self.delay is None:
            self.delay = np.zeros(self.dynamics.n_u)
            self.delay_buffer = None

        self.n_ss_pts       = control_params.n_ss_pts
        self.n_ss_its       = control_params.n_ss_its

        self.terminal_cost_scaling  = control_params.terminal_cost_scaling
        self.adaptive_scaling       = control_params.adaptive_scaling

        self.reg_e                  = control_params.regression_regularization
        self.reg_state_in_idxs      = control_params.regression_state_in_idxs
        self.reg_input_in_idxs      = control_params.regression_input_in_idxs
        self.reg_state_out_idxs     = control_params.regression_state_out_idxs

        self.nn_w                   = control_params.nearest_neighbor_weights
        self.nn_h                   = control_params.nearest_neighbor_bw
        self.nn_max                 = control_params.nearest_neighbor_max_points

        self.convex_hull_slack_quad     = np.array(control_params.convex_hull_slack_quad)
        self.convex_hull_slack_lin      = np.array(control_params.convex_hull_slack_lin)
        
        self.soft_state_bound_idxs      = control_params.soft_state_bound_idxs
        if self.soft_state_bound_idxs is not None:
            self.soft_state_bound_quad      = np.array(control_params.soft_state_bound_quad)
            self.soft_state_bound_lin       = np.array(control_params.soft_state_bound_lin)

        self.wrapped_state_idxs     = control_params.wrapped_state_idxs
        self.wrapped_state_periods  = control_params.wrapped_state_periods

        self.warm_start_with_nonlinear_rollout = False

        if control_params.state_scaling:
            self.q_scaling = 1/np.array(control_params.state_scaling)
        else:
            self.q_scaling = np.ones(self.n_q)
        self.q_scaling_inv = 1/self.q_scaling
        if control_params.input_scaling:
            self.u_scaling  = 1/np.array(control_params.input_scaling)
        else:
            self.u_scaling = np.ones(self.n_u)
        self.u_scaling_inv = 1/self.u_scaling

        # Process box constraints
        self.state_ub, self.input_ub = self.dynamics.state2qu(bounds['qu_ub'])
        self.state_lb, self.input_lb = self.dynamics.state2qu(bounds['qu_lb'])
        _, self.input_rate_ub = self.dynamics.state2qu(bounds['du_ub'])
        _, self.input_rate_lb = self.dynamics.state2qu(bounds['du_lb'])

        # if self.soft_state_bound_idxs is not None:
        #     self.state_ub[self.soft_state_bound_idxs] = np.inf
        #     self.state_lb[self.soft_state_bound_idxs] = -np.inf

        # Construct normalization matrix
        self.qu_scaling     = np.concatenate((self.q_scaling, self.u_scaling))
        self.qu_scaling_inv = 1/self.qu_scaling

        self.qu_ub = np.concatenate((self.state_ub, self.input_ub))
        self.qu_lb = np.concatenate((self.state_lb, self.input_lb))
        self.D_lb = np.concatenate((np.tile(self.qu_lb, self.N+1), np.tile(self.input_rate_lb, self.N)))
        self.D_ub = np.concatenate((np.tile(self.qu_ub, self.N+1), np.tile(self.input_rate_ub, self.N)))

        self.D_scaling = np.concatenate((np.tile(self.qu_scaling, self.N+1), np.ones(self.N*self.n_u)))
        self.D_scaling_inv = 1/self.D_scaling

        self.iter_data = []
        self.SS_data = []

        self.q_pred = np.zeros((self.N+1, self.n_q))
        self.u_pred = np.zeros((self.N, self.n_u))
        self.du_pred = np.zeros((self.N, self.n_u))

        self.SS_q_sel = np.zeros((self.n_ss_pts, self.n_q))
        self.SS_u_sel = np.zeros((self.n_ss_pts, self.n_u))
        self.SS_Q_sel = np.zeros(self.n_ss_pts)

        self.A = None
        self.B = None
        self.g = None

        self.q_ws = None
        self.u_ws = np.zeros((self.N+1, self.n_u))
        self.du_ws = np.zeros((self.N, self.n_u))
        self.a_ws = None

        self.n_c = [0 for _ in range(self.N+1)]

        self.first_solve = True
        self.init_safe_set_iters = []

        self._build_solver()

        self.debug_plot = control_params.debug_plot
        if self.debug_plot:
            plt.ion()
            self.fig = plt.figure(figsize=(10,5))
            self.ax_xy = self.fig.add_subplot(1,2,1)
            self.ax_a = self.fig.add_subplot(2,2,2)
            self.ax_d = self.fig.add_subplot(2,2,4)
            self.dynamics.track.plot_map(self.ax_xy, close_loop=False)
            self.l_xy = self.ax_xy.plot([], [], 'bo', markersize=4)[0]
            self.l_ss = self.ax_xy.plot([], [], 'rs', markersize=4, markerfacecolor='None')[0]
            self.l_a = self.ax_a.plot([], [], '-bo')[0]
            self.l_d = self.ax_d.plot([], [], '-bo')[0]
            self.ax_a.set_ylabel('accel')
            self.ax_d.set_ylabel('steering')
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()

        self.state_input_prediction = None
        self.safe_set = None
    
    def _update_debug_plot(self, q, u, ss):
        x, y, ss_x, ss_y = [], [], [], []
        for i in range(q.shape[0]):
            xt, yt, _ = self.track.local_to_global((q[i,4], q[i,5], 0))
            x.append(xt); y.append(yt)
        for i in range(ss.shape[0]):
            xt, yt, _ = self.track.local_to_global((ss[i,4], ss[i,5], 0))
            ss_x.append(xt); ss_y.append(yt)
        self.l_xy.set_data(x, y)
        self.l_ss.set_data(ss_x, ss_y)
        self.ax_xy.set_aspect('equal')
        self.l_a.set_data(np.arange(self.N), u[:,0])
        self.l_d.set_data(np.arange(self.N), u[:,1])
        self.ax_a.relim()
        self.ax_a.autoscale_view()
        self.ax_d.relim()
        self.ax_d.autoscale_view()
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        pdb.set_trace()

    def initialize(self):
        pass

    def add_iter_data(self, q_iter, u_iter):
        self.iter_data.append(dict(t=len(q_iter), state=q_iter, input=u_iter))
        
    def add_safe_set_data(self, state_ss, input_ss, cost_to_go_ss, iter_idx=None):
        if iter_idx is not None:
            self.SS_data[iter_idx] = dict(state=state_ss, input=input_ss, cost_to_go=cost_to_go_ss)
        else:
            self.SS_data.append(dict(state=state_ss, input=input_ss, cost_to_go=cost_to_go_ss))

    def _add_point_to_safe_set(self, vehicle_state: VehicleState):
        q, u = self.dynamics.state2qu(vehicle_state)
        self.SS_data[-1]['state'] = np.vstack((self.SS_data[-1]['state'], q.reshape((1,-1))))
        self.SS_data[-1]['input'] = np.vstack((self.SS_data[-1]['input'], u.reshape((1,-1))))
        self.SS_data[-1]['cost_to_go'] = np.append(self.SS_data[-1]['cost_to_go'], self.SS_data[-1]['cost_to_go'][-1]-1)

    def set_warm_start(self, u_ws: np.ndarray, du_ws: np.ndarray, 
                            q_ws: np.ndarray = None, 
                            a_ws: np.ndarray = None, 
                            l_ws: np.ndarray = None):
        self.q_ws = q_ws
        self.u_ws = u_ws
        self.u_prev = u_ws[0]
        self.du_ws = du_ws

        if a_ws is not None:
            self.a_ws = a_ws
        else:
            self.a_ws = np.zeros(self.n_ss_pts)
        if l_ws is not None:
            self.l_ws = l_ws
        
        if self.delay_buffer is not None:
            for i in range(self.dynamics.n_u):
                self.delay_buffer.append(deque(self.u_ws[1:1+self.delay[i],i], maxlen=self.delay[i]))

    def step(self, vehicle_state: VehicleState, env_state=None):
        self.solve(vehicle_state)

        u = self.u_pred[0]
        self.dynamics.qu2state(vehicle_state, None, u)

        if self.state_input_prediction is None:
            self.state_input_prediction = VehiclePrediction()
        if self.safe_set is None:
            self.safe_set = VehiclePrediction()
        self.dynamics.qu2prediction(self.state_input_prediction, self.q_pred, self.u_pred)
        self.dynamics.qu2prediction(self.safe_set, self.SS_q_sel, self.SS_u_sel)
        self.state_input_prediction.t = vehicle_state.t
        self.safe_set.t = vehicle_state.t

        # Update delay buffer
        if self.delay_buffer is not None:
            for i in range(self.dynamics.n_u):
                self.delay_buffer[i].append(u[i])

        # Construct initial guess for next iteration
        u_ws = np.vstack((self.u_pred, self.u_pred[-1]))
        du_ws = np.vstack((self.du_pred[1:], self.du_pred[-1]))
        q_ws = np.vstack((self.q_pred[1:], self.q_pred[-1]))
        self.set_warm_start(u_ws, du_ws, q_ws=q_ws)

        # This method adds the state and input to the safe sets at the previous iteration.
        # The "s" component will be greater than the track length and the cost-to-go will be negative
        state = VehicleState()
        self.dynamics.qu2state(state, self.q_pred[0], self.u_pred[0])
        state.p.s += self.track.track_length
        self._add_point_to_safe_set(state)

        return

    def solve(self, state: VehicleState, params: np.ndarray = None):
        if self.first_solve:
            self.init_safe_set_iters = np.arange(len(self.SS_data))
            self.first_solve = False

        state.e.psi = np.mod(state.e.psi, 2*np.pi)

        q0, _ = self.dynamics.state2qu(state)
        um1 = self.u_prev

        if self.delay_buffer is not None:
            delay_steps = int(np.amin(self.delay))
            u_delay = np.hstack([np.array(self.delay_buffer[i])[:delay_steps].reshape((-1,1)) for i in range(self.dynamics.n_u)])
            q_bar = self._evaluate_dynamics(q0, u_delay)
            q0 = q_bar[-1]
            um1 = u_delay[-1]

        if self.q_ws is not None:
            q_ws = self.q_ws
            q_ws[0] = q0
            if self.wrapped_state_idxs is not None:
                for i, p in zip(self.wrapped_state_idxs, self.wrapped_state_periods):
                    q_ws[:,i] = np.unwrap(q_ws[:,i], period=p)
        else:
            q_ws = self._evaluate_dynamics(q0, self.u_ws[1:])

        if self.verbose:
            self.print_method('Current state q: ' + str(q0))
        SS_q, SS_u, SS_Q = self._select_safe_set(q_ws[-1], self.u_ws[-1], mode='state')
        SS_q = np.vstack(SS_q)
        SS_u = np.vstack(SS_u)
        SS_Q = np.concatenate(SS_Q)

        D = np.concatenate((np.hstack((q_ws, self.u_ws)).ravel(), 
                            self.du_ws.ravel()))
        P = np.concatenate((q0, um1))

        for _ in range(self.qp_iters):
            if self.qp_interface == 'casadi':
                D_bar, success, status = self._solve_casadi(D, P, SS_q, SS_u, SS_Q)
            elif self.qp_interface == 'cvxpy':
                D_bar, success, status = self._solve_cvxpy(D, P, SS_q, SS_u, SS_Q)
            if not success:
                self.print_method('Warning: QP returned ' + str(status))
                break
            if self.qp_iters > 1:
                D = self.damping*D + (1-self.damping)*D_bar
                D[(self.n_q+self.n_u)*(self.N+1):] = 0
            else:
                D = D_bar

        if success:
            # Unpack solution
            qu_sol = D[:(self.n_q+self.n_u)*(self.N+1)].reshape((self.N+1, self.n_q+self.n_u))
            du_sol = D[(self.n_q+self.n_u)*(self.N+1):(self.n_q+self.n_u)*(self.N+1)+self.n_u*self.N].reshape((self.N, self.n_u))

            u_sol = qu_sol[1:,self.n_q:self.n_q+self.n_u]
            if self.warm_start_with_nonlinear_rollout:
                q_sol = self._evaluate_dynamics(q0, u_sol)
            else:
                q_sol = qu_sol[:,:self.n_q]
        else:
            q_sol = self.q_ws
            u_sol = self.u_ws[1:]
            du_sol = self.du_ws

        if self.debug_plot:
            self._update_debug_plot(q_sol, u_sol, SS_q)

        self.q_pred = q_sol
        self.u_pred = u_sol
        self.du_pred = du_sol
        self.SS_q_sel = SS_q
        self.SS_u_sel = SS_u
        self.SS_Q_sel = SS_Q
    
    def get_prediction(self):
        return self.state_input_prediction

    def get_safe_set(self):
        return self.safe_set

    def _evaluate_dynamics(self, q0, U):
        t = time.time()
        Q = [q0]
        for k in range(U.shape[0]):
            Q.append(self.dynamics.fd(Q[k], U[k]).toarray().squeeze())
        if self.verbose:
            dt = (time.time()-t)*1000
            self.print_method(f'Dynamics evalution time: {dt:.2f} ms')
        return np.array(Q)

    def _select_safe_set(self, q, u, mode='state'):
        t = time.time()

        if self.verbose:
            if mode == 'state':
                self.print_method('Finding safe set about q: ' + str(q))
            elif mode == 'state_input':
                self.print_method('Finding safe set about q: ' + str(q) + ' u: ' + str(u))
        n_ss = int(self.n_ss_pts/self.n_ss_its)
        iter_costs = []
        for d in self.SS_data:
            iter_costs.append(d['cost_to_go'][0])
        iter_idxs = np.argsort(iter_costs)[:self.n_ss_its]

        SS_q, SS_u, SS_Q = [], [], []
        for i in iter_idxs:
            n_data = self.SS_data[i]['state'].shape[0]
            if mode == 'state':
                z = q
                z_data = self.SS_data[i]['state']
            elif mode == 'state_input':
                z = np.concatenate((q, u))
                z_data = np.hstack((self.SS_data[i]['state'], self.SS_data[i]['input']))
            dist = np.linalg.norm(z_data - np.tile(z, (z_data.shape[0], 1)), ord=1, axis=1)
            min_idx = np.argmin(dist)
            
            if min_idx - int(n_ss/2) < 0:
                SS_idxs = np.arange(n_ss)
            elif min_idx + int(n_ss/2) > n_data:
                SS_idxs = np.arange(n_data-n_ss, n_data)
            else:
                SS_idxs = np.arange(min_idx-int(n_ss/2), min_idx+int(n_ss/2))

            SS_q.append(self.SS_data[i]['state'][SS_idxs])
            SS_u.append(self.SS_data[i]['input'][SS_idxs])
            SS_Q.append(self.SS_data[i]['cost_to_go'][SS_idxs])

        if self.verbose:
            dt = (time.time()-t)*1000
            self.print_method(f'Safe set selection time: {dt:.2f} ms')

        return SS_q, SS_u, SS_Q

    def _regress_model(self, D, P):
        t = time.time()
        iter_costs = []
        for d in self.SS_data:
            iter_costs.append(d['cost_to_go'][0])
        # iter_idxs = np.argsort(iter_costs)[:self.n_ss_its] # Fastest laps
        iter_idxs = np.arange(len(self.iter_data)-self.n_ss_its, len(self.iter_data)) # Most recent laps

        if self.keep_init_safe_set:
            for i in self.init_safe_set_iters:
                if i not in iter_idxs:
                    iter_idxs = np.append(iter_idxs, i)
        
        qk, uk, qkp1 = [], [], []
        for j in iter_idxs:
            qk.append(self.iter_data[j]['state'][:-1])
            uk.append(self.iter_data[j]['input'][:-1])
            qkp1.append(self.iter_data[j]['state'][1:])
        qk = np.vstack(qk)
        uk = np.vstack(uk)
        qkp1 = np.vstack(qkp1)

        data_quality = []
        A, B, g = [], [], []
        for k in range(self.N):
            q = D[k*self.n_z:k*self.n_z+self.n_q]
            u = D[(k+1)*self.n_z+self.n_q:(k+2)*self.n_z]

            # Linearized model
            _A = self.f_Ad(q, u).full()
            _B = self.f_Bd(q, u).full()
            _g = self.f_gd(q, u).full().squeeze()

            if self.reg_state_out_idxs:
                for i in range(len(self.reg_state_out_idxs)):
                    _A[self.reg_state_out_idxs[i]] = 0
                    _B[self.reg_state_out_idxs[i]] = 0
                    z = np.concatenate((q[self.reg_state_in_idxs[i]], u))
                    
                    z_data = np.hstack((qk[:,self.reg_state_in_idxs[i]], uk))
                    ii, K = self._compute_nn_indices(z, z_data, self.nn_w[i], self.nn_h, self.nn_max*len(iter_idxs)) 
                    reg_X_data = np.hstack((qk[:,self.reg_state_in_idxs[i]], uk[:,self.reg_input_in_idxs[i]]))[ii]
                    reg_Y_data = qkp1[:,self.reg_state_out_idxs[i]][ii]

                    data_quality.append(np.amax(K)/0.75)

                    # reg_X_data, reg_Y_data, K = [], [], []
                    # for j in iter_idxs:
                    #     qd, ud = self.iter_data[j]['state'], self.iter_data[j]['input']
                    #     z_data = np.hstack((qd[:-1,self.reg_state_in_idxs[i]], ud[:-1]))
                    #     ii, kk = self._compute_nn_indices(z, z_data, self.nn_w[i], self.nn_h, self.nn_max)
                    #     if ii is not None:
                    #         X = np.hstack((qd[:,self.reg_state_in_idxs[i]], ud[:,self.reg_input_in_idxs[i]]))
                    #         reg_X_data.append(X[ii])
                    #         Y = qd[:,self.reg_state_out_idxs[i]]
                    #         reg_Y_data.append(Y[ii+1])
                    #         K = np.append(K, kk)
                    # reg_X_data = np.vstack(reg_X_data)
                    # reg_Y_data = np.vstack(reg_Y_data)

                    M = np.hstack((reg_X_data, np.ones((reg_X_data.shape[0], 1))))
                    Q = M.T @ np.diag(K) @ M + self.reg_e*np.eye(M.shape[1])
                    b = -M.T @ np.diag(K) @ reg_Y_data

                    R = (-np.linalg.pinv(Q, hermitian=True) @ b).squeeze()
                    # R = -np.linalg.solve(Q, b).squeeze()

                    _A[self.reg_state_out_idxs[i],self.reg_state_in_idxs[i]] = R[:len(self.reg_state_in_idxs[i])]
                    _B[self.reg_state_out_idxs[i],self.reg_input_in_idxs[i]] = R[len(self.reg_state_in_idxs[i]):len(self.reg_state_in_idxs[i])+len(self.reg_input_in_idxs[i])]
                    _g[self.reg_state_out_idxs[i]] = R[-1]

            A.append(_A)
            B.append(_B)
            g.append(_g)

        if self.verbose:
            dt = (time.time()-t)*1000
            self.print_method(f'Model regression time: {dt:.2f} ms')
        
        max_quality = np.max(data_quality)
        if np.max(data_quality) <= 0.5:
            self.print_method(f'Warning: sparse data (quality = {max_quality:.2f}) w.r.t. h = {self.nn_h}')

        return A, B, g, data_quality

    def _compute_nn_indices(self, z, data, weights, h, max_pts=25):
        Z = np.tile(z, (data.shape[0], 1))
        # Compute distances
        _dist = np.linalg.norm((data - Z) @ np.diag(weights), ord=1, axis=1)
        
        # Sort by distance
        _idxs = np.argsort(_dist)
        dist = _dist[_idxs]
        
        # Filter by distance threshold
        idxs = _idxs[np.where(dist < h)[0]]

        if len(idxs) == 0:
            idxs_ret = _idxs
        else:
            idxs_ret = idxs
            
        if len(idxs_ret) > max_pts:
            idxs_ret = idxs_ret[:max_pts]

        # if idxs_ret is not None:
        #     K  = (1-(dist[idxs_ret]/h)**2)*3/4
        # else:
        #     K = None
        
        K  = np.maximum((1-(dist[idxs_ret]/h)**2)*3/4, 0)
        # K  = (1-(np.minimum(dist[idxs_ret],h)/h)**2)*3/4

        return idxs_ret, K

    def _build_solver(self):
        # Dynamcis augmented with arc length dynamics
        q_sym = ca.MX.sym('q', self.n_q)
        u_sym = ca.MX.sym('u', self.n_u)
        
        # Use default discretization scheme
        self.f_Ad = self.dynamics.fAd
        self.f_Bd = self.dynamics.fBd
        self.f_gd = ca.Function('gd', [q_sym, u_sym], [self.dynamics.fd(q_sym, u_sym) - self.f_Ad(q_sym, u_sym) @ q_sym - self.f_Bd(q_sym, u_sym) @ u_sym])

        # q_0, ..., q_N
        q_ph = [ca.MX.sym(f'q_ph_{k}', self.n_q) for k in range(self.N+1)] # State
        # u_-1, u_0, ..., u_N-1
        u_ph = [ca.MX.sym(f'u_ph_{k}', self.n_u) for k in range(self.N+1)] # Inputs
        # du_0, ..., du_N-1
        du_ph = [ca.MX.sym(f'du_ph_{k}', self.n_u) for k in range(self.N)] # Input rates

        qu0_ph = ca.MX.sym('qu0', self.n_z) # Initial state

        # Scaling matricies
        T_q = ca.DM(sp.sparse.diags(self.q_scaling))
        T_q_inv = ca.DM(sp.sparse.diags(self.q_scaling_inv))
        T_u = ca.DM(sp.sparse.diags(self.u_scaling))
        T_u_inv = ca.DM(sp.sparse.diags(self.u_scaling_inv))
        T_qu = ca.DM(sp.sparse.diags(self.qu_scaling))
        T_qu_inv = ca.DM(sp.sparse.diags(self.qu_scaling_inv))

        A, B, g = [], [], []
        Q_qu, q_qu, Q_du, q_du = [], [], [], []
        C_qu, C_qu_lb, C_qu_ub, C_du, C_du_lb, C_du_ub = [], [], [], [], [], []
        for k in range(self.N+1):
            _Q_qu = ca.MX.sym(f'_Q_qu_{k}', ca.Sparsity(self.n_z, self.n_z))
            _q_qu = ca.MX.sym(f'_q_qu_{k}', ca.Sparsity(self.n_z, 1))

            # Quadratic approximation of state costs
            if self.costs['state'][k]:
                Jq_k = self.costs['state'][k](q_ph[k])
            else:
                Jq_k = ca.DM.zeros(1)
            M_q = ca.jacobian(ca.jacobian(Jq_k, q_ph[k]), q_ph[k])
            m_q = ca.jacobian(Jq_k, q_ph[k]).T
            _Q_qu[:self.n_q,:self.n_q] = 2 * T_q_inv @ M_q @ T_q_inv
            _q_qu[:self.n_q] = T_q_inv @ (m_q - M_q @ q_ph[k])

            # Quadratic approximation of input costs
            if self.costs['input'][k]:
                Ju_k = self.costs['input'][k](u_ph[k])
            else:
                Ju_k = ca.DM.zeros(1)
            M_u = ca.jacobian(ca.jacobian(Ju_k, u_ph[k]), u_ph[k])
            m_u = ca.jacobian(Ju_k, u_ph[k]).T
            _Q_qu[self.n_q:,self.n_q:] = 2 * T_u_inv @ M_u @ T_u_inv
            _q_qu[self.n_q:] = T_u_inv @ (m_u - M_u @ u_ph[k])

            Q_qu.append((_Q_qu + _Q_qu.T)/2 + 1e-10*ca.DM.eye(self.n_q+self.n_u))
            q_qu.append(_q_qu)

            _C_qu, _C_qu_ub, _C_qu_lb = ca.MX.sym(f'_C_qu_{k}', 0, self.n_z), ca.MX.sym(f'_C_qu_ub_{k}', 0), ca.MX.sym(f'_C_qu_lb_{k}', 0)
            # Linear approximation of constraints on states and inputs
            if self.constraints['state_input'][k]:
                _C = ca.jacobian(self.constraints['state_input'][k](q_ph[k], u_ph[k]), ca.vertcat(q_ph[k], u_ph[k]))
                _C_ub = -self.constraints['state_input'][k](q_ph[k], u_ph[k]) + _C @ ca.vertcat(q_ph[k], u_ph[k])
                _C_lb = -1e10*np.ones(_C_ub.size1())
                
                _C_qu = ca.vertcat(_C_qu, _C)
                _C_qu_ub = ca.vertcat(_C_qu_ub, _C_ub)
                _C_qu_lb = ca.vertcat(_C_qu_lb, _C_lb)

            self.n_c[k] += _C_qu.size1()
            C_qu.append(_C_qu @ T_qu_inv)
            C_qu_ub.append(_C_qu_ub)
            C_qu_lb.append(_C_qu_lb)

            if k < self.N:
                # Linearized dynamics
                _A = ca.MX.sym(f'_A_{k}', ca.Sparsity(self.n_z, self.n_z))
                _A[:self.n_q,:self.n_q] = T_q @ self.f_Ad(q_ph[k], u_ph[k]) @ T_q_inv
                _A[:self.n_q,self.n_q:] = T_q @ self.f_Bd(q_ph[k], u_ph[k]) @ T_u_inv
                _A[self.n_q:,self.n_q:] = ca.DM.eye(self.n_u)
                A.append(_A)

                _B = ca.MX.sym(f'_B_{k}', ca.Sparsity(self.n_z, self.n_u))
                _B[:self.n_q,:] = self.dt*T_q @ self.f_Bd(q_ph[k], u_ph[k]) @ T_u_inv
                _B[self.n_q:,:] = self.dt*ca.DM.eye(self.n_u)
                B.append(_B)

                _g = ca.MX.sym(f'_g_{k}', ca.Sparsity(self.n_z, 1))
                _g[:self.n_q] = T_q @ self.f_gd(q_ph[k], u_ph[k])
                _g[self.n_q:] = ca.DM.zeros(self.n_u)
                g.append(_g)

                # Quadratic approximation of input rate costs
                if self.costs['rate'][k]:
                    Jdu_k = self.costs['rate'][k](du_ph[k])
                else:
                    Jdu_k = ca.DM.zeros(1)
                M_du = ca.jacobian(ca.jacobian(Jdu_k, du_ph[k]), du_ph[k])
                m_du = ca.jacobian(Jdu_k, du_ph[k]).T
                Q_du.append(2*M_du)
                q_du.append(m_du - M_du @ du_ph[k])

                # Linear approximation of constraints on input rates
                _C_du, _C_du_ub, _C_du_lb = ca.MX.sym(f'_C_du_{k}', 0, self.n_u), ca.MX.sym(f'_C_du_ub_{k}', 0), ca.MX.sym(f'_C_du_lb_{k}', 0)
                if self.constraints['rate'][k]:
                    _C_du = ca.jacobian(self.constraints['rate'][k](du_ph[k]), du_ph[k])
                    _C_du_ub = -self.constraints['rate'][k](du_ph[k]) + _C_du @ du_ph[k]
                    _C_du_lb = -1e10*np.ones(_C_du.size1())
                
                self.n_c[k] += _C_du.size1()
                C_du.append(_C_du)
                C_du_ub.append(_C_du_ub)
                C_du_lb.append(_C_du_lb)
        
        # Form decision vector using augmented states (q_k, u_k) and inputs du_k
        # D = [(q_0, u_-1), ..., (q_N, u_N-1), du_0, ..., du_N-1, SS multipliers]
        D = []
        for x, w in zip(q_ph, u_ph):
            D.extend([x, w])
        D += du_ph

        # Parameters
        P = [qu0_ph]
        
        if self.qp_interface == 'casadi':
            ss_q_ph = [ca.MX.sym(f'ssq_ph_{k}', self.n_q) for k in range(self.n_ss_pts)] # Safe set points
            ss_u_ph = [ca.MX.sym(f'ssu_ph_{k}', self.n_u) for k in range(self.n_ss_pts)] # Safe set points
            ss_Q_ph = ca.MX.sym('ssQ_ph', self.n_ss_pts)
            a_ph  = ca.MX.sym('a_ph', self.n_ss_pts) # Safe set multipliers
            convex_hull_slack_ph = ca.MX.sym('convex_hull_slack_ph', self.n_q) # Convex hull slack
            D += [a_ph, convex_hull_slack_ph]

            if self.soft_state_bound_idxs is not None:
                state_ub_slack_ph = ca.MX.sym('state_ub_slack_ph', len(self.soft_state_bound_idxs))
                state_lb_slack_ph = ca.MX.sym('state_ub_slack_ph', len(self.soft_state_bound_idxs))
                D += [state_ub_slack_ph, state_lb_slack_ph]
            P += [ca.vertcat(*ss_q_ph), ca.vertcat(*ss_u_ph), ss_Q_ph]
        
        D = ca.vertcat(*D)
        P = ca.vertcat(*P)

        n_D = D.size1()

        if self.qp_interface == 'casadi':
            # Construct batch QP cost matrix and vector
            H = ca.MX.sym('H', ca.Sparsity(n_D, n_D))
            h = ca.MX.sym('h', ca.Sparsity(n_D, 1))
            for k in range(self.N+1):
                H[k*self.n_z:(k+1)*self.n_z,k*self.n_z:(k+1)*self.n_z]  = Q_qu[k]
                h[k*self.n_z:(k+1)*self.n_z]                            = q_qu[k]
                if k < self.N:
                    s_idx, e_idx = (self.N+1)*self.n_z+k*self.n_u, (self.N+1)*self.n_z+(k+1)*self.n_u
                    H[s_idx:e_idx,s_idx:e_idx]  = Q_du[k]
                    h[s_idx:e_idx]              = q_du[k]
            
            # Cost-to-go
            s_idx = (self.N+1)*self.n_z + self.N*self.n_u
            h[s_idx:s_idx+self.n_ss_pts] = self.terminal_cost_scaling*ss_Q_ph
            # Convex hull slack cost
            s_idx = (self.N+1)*self.n_z + self.N*self.n_u + self.n_ss_pts
            H[s_idx:s_idx+self.n_q,s_idx:s_idx+self.n_q] = 2*ca.DM(sp.sparse.diags(self.convex_hull_slack_quad))
            h[s_idx:s_idx+self.n_q] = self.convex_hull_slack_lin
            # Soft state bounds slack cost
            if self.soft_state_bound_idxs is not None:
                n_s = 2*len(self.soft_state_bound_idxs)
                s_idx = (self.N+1)*self.n_z + self.N*self.n_u + self.n_ss_pts + self.n_q
                H[s_idx:s_idx+n_s,s_idx:s_idx+n_s] = 2*ca.DM(sp.sparse.diags(np.concatenate((self.soft_state_bound_quad, self.soft_state_bound_quad))))
                h[s_idx:s_idx+n_s] = np.concatenate((self.soft_state_bound_lin, self.soft_state_bound_lin))

            H = (H + H.T)/2 + 1e-9*ca.DM.eye(n_D)
            self.f_H = ca.Function('H', [D, P], [H])
            self.f_h = ca.Function('h', [D, P], [h])
            
            # Construct equality constraint matrix and vector
            A_ph = [ca.MX.sym(f'A_{k}', self.n_q, self.n_q) for k in range(self.N)]
            B_ph = [ca.MX.sym(f'B_{k}', self.n_q, self.n_u) for k in range(self.N)]
            g_ph = [ca.MX.sym(f'g_{k}', self.n_q) for k in range(self.N)]

            A_eq = ca.MX.sym('A_eq', ca.Sparsity((self.N+1)*self.n_z + self.n_q + 1, n_D))
            b_eq = ca.MX.sym('b_eq', ca.Sparsity((self.N+1)*self.n_z + self.n_q + 1, 1))
            A_eq[:self.n_z,:self.n_z] = ca.DM.eye(self.n_z)
            b_eq[:self.n_z] = T_qu @ qu0_ph
            for k in range(self.N):
                A_tmp = ca.MX.sym('A_tmp', ca.Sparsity(self.n_z, self.n_z))
                A_tmp[:self.n_q,:self.n_q] = T_q @ A_ph[k] @ T_q_inv
                A_tmp[:self.n_q,self.n_q:] = T_q @ B_ph[k] @ T_u_inv
                A_tmp[self.n_q:,self.n_q:] = ca.DM.eye(self.n_u)

                B_tmp = ca.MX.sym('B_tmp', ca.Sparsity(self.n_z, self.n_u))
                B_tmp[:self.n_q,:] = self.dt*T_q @ B_ph[k] @ T_u_inv
                B_tmp[self.n_q:,:] = self.dt*ca.DM.eye(self.n_u)

                g_tmp = ca.MX.sym('g_tmp', ca.Sparsity(self.n_z, 1))
                g_tmp[:self.n_q] = T_q @ g_ph[k]

                A_eq[(k+1)*self.n_z:(k+2)*self.n_z,k*self.n_z:(k+1)*self.n_z] = -A_tmp
                A_eq[(k+1)*self.n_z:(k+2)*self.n_z,(k+1)*self.n_z:(k+2)*self.n_z] = ca.DM.eye(self.n_z)
                A_eq[(k+1)*self.n_z:(k+2)*self.n_z,(self.N+1)*self.n_z+k*self.n_u:(self.N+1)*self.n_z+(k+1)*self.n_u] = -B_tmp
                b_eq[(k+1)*self.n_z:(k+2)*self.n_z] = g_tmp

            # Convex hull constraint
            rs_idx = (self.N+1)*self.n_z
            re_idx = rs_idx + self.n_q
            cs_idx = self.N*self.n_z
            ce_idx = cs_idx + self.n_q
            A_eq[rs_idx:re_idx,cs_idx:ce_idx] = -ca.DM.eye(self.n_q) @ T_q_inv
            cs_idx = (self.N+1)*self.n_z + self.N*self.n_u
            for i in range(self.n_ss_pts):
                A_eq[rs_idx:re_idx,cs_idx+i] = ss_q_ph[i]
            cs_idx = (self.N+1)*self.n_z + self.N*self.n_u + self.n_ss_pts
            ce_idx = cs_idx + self.n_q
            A_eq[rs_idx:re_idx,cs_idx:ce_idx] = -ca.DM.eye(self.n_q)
            b_eq[rs_idx:re_idx] = ca.DM.zeros(self.n_q)

            # Multiplier sum constraint
            r_idx = (self.N+1)*self.n_z + self.n_q
            cs_idx = (self.N+1)*self.n_z + self.N*self.n_u
            ce_idx = cs_idx + self.n_ss_pts
            A_eq[r_idx,cs_idx:ce_idx] = np.ones((1, self.n_ss_pts))
            b_eq[r_idx] = 1
            
            self.f_A_eq = ca.Function('A_eq', [D, P] + A_ph + B_ph + g_ph, [A_eq])
            self.f_b_eq = ca.Function('b_eq', [D, P] + A_ph + B_ph + g_ph, [b_eq])

            # Construct inequality constraint matrix and vectors
            n_Cqu = int(np.sum([c.size1() for c in C_qu]))
            n_Cdu = int(np.sum([c.size1() for c in C_du]))
            n_C = n_Cqu + n_Cdu
            if self.soft_state_bound_idxs is not None:
                n_C += 2*self.N*len(self.soft_state_bound_idxs)

            A_in = ca.MX.sym('A_in', ca.Sparsity(n_C, n_D))
            ub_in = ca.MX.sym('ub_in', ca.Sparsity(n_C, 1))
            lb_in = ca.MX.sym('lb_in', ca.Sparsity(n_C, 1))
            s1_idx, s2_idx = 0, n_Cqu
            for k in range(self.N+1):
                n_c = C_qu[k].size1()
                A_in[s1_idx:s1_idx+n_c,k*self.n_z:(k+1)*self.n_z] = C_qu[k]
                ub_in[s1_idx:s1_idx+n_c] = C_qu_ub[k]
                lb_in[s1_idx:s1_idx+n_c] = C_qu_lb[k]
                s1_idx += n_c
                if k < self.N:
                    n_c = C_du[k].size1()
                    A_in[s2_idx:s2_idx+n_c,(self.N+1)*self.n_z+k*self.n_u:(self.N+1)*self.n_z+(k+1)*self.n_u] = C_du[k]
                    ub_in[s2_idx:s2_idx+n_c] = C_du_ub[k]
                    lb_in[s2_idx:s2_idx+n_c] = C_du_lb[k]
                    s2_idx += n_c

            if self.soft_state_bound_idxs is not None:
                rs_idx = n_Cqu + n_Cdu
                cs_idx = (self.N+1)*self.n_z + self.N*self.n_u + self.n_ss_pts + self.n_q
                for i, j in enumerate(self.soft_state_bound_idxs):
                    for k in range(self.N):
                        # q[j] <= ub[j] + s[j]
                        A_in[rs_idx+2*k,(k+1)*self.n_z+j] = 1 * T_q_inv[j,j]
                        A_in[rs_idx+2*k,cs_idx+i] = -1
                        ub_in[rs_idx+2*k] = self.state_ub[j]
                        lb_in[rs_idx+2*k] = -ca.DM.inf(1)
                        # q[j] >= lb[j] - s[j]
                        A_in[rs_idx+2*k+1,(k+1)*self.n_z+j] = 1 * T_q_inv[j,j]
                        A_in[rs_idx+2*k+1,cs_idx+len(self.soft_state_bound_idxs)+i] = 1
                        lb_in[rs_idx+2*k+1] = self.state_lb[j]
                        ub_in[rs_idx+2*k+1] = ca.DM.inf(1)

            self.f_A_in = ca.Function('A_in', [D, P], [A_in])
            self.f_ub_in = ca.Function('ub_in', [D, P], [ub_in])
            self.f_lb_in = ca.Function('lb_in', [D, P], [lb_in])

        # Functions which return the QP components for each stage
        self.f_Qqu = ca.Function('Qqu', [D, P], Q_qu, self.options)
        self.f_qqu = ca.Function('qqu', [D, P], q_qu, self.options)
        self.f_Qdu = ca.Function('Qdu', [D, P], Q_du, self.options)
        self.f_qdu = ca.Function('qdu', [D, P], q_du, self.options)
        self.f_Cqu = ca.Function('Cqu', [D, P], C_qu, self.options)
        self.f_Cquub = ca.Function('Cqu_ub', [D, P], C_qu_ub, self.options)
        self.f_Cqulb = ca.Function('Cqu_lb', [D, P], C_qu_lb, self.options)
        self.f_Cdu = ca.Function('Cdu', [D, P], C_du, self.options)
        self.f_Cduub = ca.Function('Cdu_ub', [D, P], C_du_ub, self.options)
        self.f_Cdulb = ca.Function('Cdu_lb', [D, P], C_du_lb, self.options)
        self.f_A = ca.Function('A', [D, P], A, self.options)
        self.f_B = ca.Function('B', [D, P], B, self.options)
        self.f_g = ca.Function('g', [D, P], g, self.options)

        if self.qp_interface == 'casadi':
            prob = dict(h=H.sparsity(), a=ca.vertcat(A_eq.sparsity(), A_in.sparsity()))
            if self.qp_solver == 'qpoases':
                solver_opts = dict(error_on_fail=False, sparse=False, printLevel='tabular' if self.verbose else 'none')
            elif self.qp_solver == 'cplex':
                os.environ['CPLEX_VERSION'] = '2210'
                solver_opts = dict(error_on_fail=False, cplex=dict(CPXPARAM_OptimalityTarget=2, CPXPARAM_ScreenOutput=self.verbose))
            elif self.qp_solver == 'osqp':
                solver_opts = dict(error_on_fail=False, osqp={'polish': True, 'verbose': False})
            else:
                raise(ValueError(f'QP solver {self.qp_solver} not supported'))
            self.solver = ca.conic('qp', self.qp_solver, prob, solver_opts)
        elif self.qp_interface == 'cvxpy':
            pass
        else:
            raise(ValueError('QP interface name not recognized'))

    def _solve_casadi(self, D, P, SS_q, SS_u, SS_Q):
        if self.verbose:
            self.print_method('============ Sovling using CaSAdi ============')
        t = time.time()
        D = np.concatenate((D, np.zeros(self.n_ss_pts), np.zeros(self.n_q)))
        P = np.concatenate((P, SS_q.ravel(), SS_u.ravel(), SS_Q))

        D_scaling = np.concatenate((self.D_scaling, np.ones(self.n_ss_pts + self.n_q)))
        state_ub = copy.copy(self.state_ub)
        state_lb = copy.copy(self.state_lb)
        if self.soft_state_bound_idxs is not None:
            state_ub[self.soft_state_bound_idxs] = np.inf
            state_lb[self.soft_state_bound_idxs] = -np.inf
            D = np.concatenate((D, np.zeros(2*len(self.soft_state_bound_idxs))))
            D_scaling = np.concatenate((D_scaling, np.ones(2*len(self.soft_state_bound_idxs))))
        D_scaling_inv = 1/D_scaling

        qu_ub = np.concatenate((state_ub, self.input_ub))
        qu_lb = np.concatenate((state_lb, self.input_lb))
        D_lb = np.concatenate((-np.inf*np.ones(self.n_q), self.input_lb, np.tile(qu_lb, self.N), np.tile(self.input_rate_lb, self.N), np.zeros(self.n_ss_pts), -np.inf*np.ones(self.n_q)))
        D_ub = np.concatenate((np.inf*np.ones(self.n_q), self.input_ub, np.tile(qu_ub, self.N), np.tile(self.input_rate_ub, self.N), np.ones(self.n_ss_pts), np.inf*np.ones(self.n_q)))
        if self.soft_state_bound_idxs is not None:
            D_lb = np.concatenate((D_lb, np.zeros(2*len(self.soft_state_bound_idxs))))
            D_ub = np.concatenate((D_ub, np.inf*np.ones(2*len(self.soft_state_bound_idxs))))
        lb = D_scaling * D_lb
        ub = D_scaling * D_ub

        # Evaluate QP approximation
        self.A, self.B, self.g, self.data_quality = self._regress_model(D, P)
        if self.adaptive_scaling:
            P[-len(SS_Q):] = np.amax(self.data_quality)*P[-len(SS_Q):]

        h       = self.f_h(D, P)
        H       = self.f_H(D, P)
        A_in    = self.f_A_in(D, P)
        ub_in   = self.f_ub_in(D, P)
        lb_in   = self.f_lb_in(D, P)

        A_eq = self.f_A_eq(D, P, *self.A, *self.B, *self.g)
        b_eq = self.f_b_eq(D, P, *self.A, *self.B, *self.g)
        if self.verbose:
            dt = (time.time()-t)*1000
            self.print_method(f'Evaluation time: {dt:.2f} ms')

        t = time.time()
        sol = self.solver(h=H, g=h, a=ca.vertcat(A_eq, A_in), lba=ca.vertcat(b_eq, lb_in), uba=ca.vertcat(b_eq, ub_in), lbx=lb, ubx=ub)
        if self.verbose:
            dt = (time.time()-t)*1000
            self.print_method(f'Solve time: {dt:.2f} ms')

        t = time.time()
        D_bar = None
        success = self.solver.stats()['success']
        status = self.solver.stats()['return_status']

        if success:
            t = time.time()
            D_bar = (D_scaling_inv*np.array(sol['x']).squeeze())[:(self.N+1)*self.n_z+self.N*self.n_u]

            # a_val = sol['x'].toarray().squeeze()[(self.N+1)*self.n_z+self.N*self.n_u:(self.N+1)*self.n_z+self.N*self.n_u+self.n_ss_pts]
            # if self.soft_state_bound_idxs is not None:
            #     if np.amax(D_bar[-2*len(self.soft_state_bound_idxs):]) > 1e-9:
            #         self.print_method(f'Track constraint violation: {np.amax(D_bar[-2*len(self.soft_state_bound_idxs):])}')
        if self.verbose:
            dt = (time.time()-t)*1000
            self.print_method(f'Unpack time: {dt:.2f} ms')
            self.print_method('==============================================')

        return D_bar, success, status

    def _solve_cvxpy(self, D, P, SS_q, SS_u, SS_Q):
        if self.verbose:
            self.print_method('============ Sovling using CVXPY ============')
        t = time.time()
        Q = self.f_Qqu(D, P)
        q = self.f_qqu(D, P)
        R = self.f_Qdu(D, P)
        r = self.f_qdu(D, P)
        Cqu = self.f_Cqu(D, P)
        lbCqu = self.f_Cqulb(D, P)
        ubCqu = self.f_Cquub(D, P)
        Cdu = self.f_Cdu(D, P)
        lbCdu = self.f_Cdulb(D, P)
        ubCdu = self.f_Cduub(D, P)

        T_q = np.diag(self.q_scaling)
        T_q_inv = np.diag(self.q_scaling_inv)
        T_u_inv = np.diag(self.u_scaling_inv)

        self.A, self.B, self.g, _ = self._regress_model(D, P)
        A, B, g = [], [], []
        for _A, _B, _g in zip(self.A, self.B, self.g):
            A_tmp = np.zeros((self.n_z, self.n_z))
            A_tmp[:self.n_q,:self.n_q] = T_q @ _A @ T_q_inv
            A_tmp[:self.n_q,self.n_q:] = T_q @ _B @ T_u_inv
            A_tmp[self.n_q:,self.n_q:] = np.eye(self.n_u)
            A.append(A_tmp)

            B_tmp = np.zeros((self.n_z, self.n_u))
            B_tmp[:self.n_q,:] = T_q @ _B @ T_u_inv
            B_tmp[self.n_q:,:] = np.eye(self.n_u)
            B.append(B_tmp)

            g_tmp = np.zeros(self.n_z)
            g_tmp[:self.n_q] = T_q @ _g
            g.append(g_tmp)
        if self.verbose:
            dt = (time.time()-t)*1000
            self.print_method(f'Evaluation time: {dt:.2f} ms')

        t = time.time()
        x = cp.Variable(shape=(self.n_z, self.N+1))
        u = cp.Variable(shape=(self.n_u, self.N))
        a = cp.Variable(shape=(self.n_ss_pts, 1))
        convex_hull_slack = cp.Variable(shape=(self.n_q, 1))
        if self.soft_state_bound_idxs is not None:
            soft_idxs = self.soft_state_bound_idxs
            state_ub_slack = cp.Variable(shape=(len(soft_idxs), 1))
            state_lb_slack = cp.Variable(shape=(len(soft_idxs), 1))
        
        if self.soft_state_bound_idxs is not None:
            hard_idxs = np.setdiff1d(np.arange(self.n_z), soft_idxs)
            qu_scaling = self.qu_scaling[hard_idxs]
            lb = qu_scaling * np.concatenate((self.state_lb, self.input_lb))[hard_idxs]
            ub = qu_scaling * np.concatenate((self.state_ub, self.input_ub))[hard_idxs]
            lb_soft = self.q_scaling[soft_idxs] * self.state_lb[soft_idxs]
            ub_soft = self.q_scaling[soft_idxs] * self.state_ub[soft_idxs]
        else:
            lb = self.qu_scaling * np.concatenate((self.state_lb, self.input_lb))
            ub = self.qu_scaling * np.concatenate((self.state_ub, self.input_ub))
        qu0 = P[:self.n_z]

        constraints, objective = [x[:,0] == self.qu_scaling*qu0], 0
        for k in range(self.N+1):
            objective += 0.5*cp.quad_form(x[:,k], Q[k].full()) + q[k].full().T @ x[:,k]
            if k < self.N:
                objective += 0.5*cp.quad_form(u[:,k], R[k].full()) + r[k].full().T @ u[:,k]
            if k > 0:
                if self.soft_state_bound_idxs is not None:
                    constraints += [x[soft_idxs,k] <= ub_soft + state_ub_slack, x[soft_idxs,k] >= lb_soft - state_lb_slack]
                    if len(hard_idxs) > 0:
                        constraints += [x[hard_idxs,k] <= ub, x[hard_idxs,k] >= lb]
                else:
                    constraints += [x[:,k] <= ub, x[:,k] >= lb]
                if Cqu[k].size1() > 0:
                    constraints += [Cqu[k].full() @ x[:,k] <= ubCqu[k].full().squeeze(), Cqu[k].full() @ x[:,k] >= lbCqu[k].full().squeeze()]
            if k < self.N:
                constraints += [x[:,k+1] == A[k] @ x[:,k] + B[k] @ u[:,k] + g[k]]
                constraints += [u[:,k] <= self.input_rate_ub, u[:,k] >= self.input_rate_lb]
                if Cdu[k].size1() > 0:
                    constraints += [Cdu[k].full() @ x[:,k] <= ubCdu[k].full().squeeze(), Cdu[k].full() @ x[:,k] >= lbCdu[k].full().squeeze()]

        constraints += [cp.reshape(x[:self.n_q,-1], (self.n_q, 1)) == SS_q.T @ a + convex_hull_slack, a >= 0, cp.sum(a) == 1]
        objective += self.terminal_cost_scaling * SS_Q @ a
        objective += 0.5*cp.quad_form(convex_hull_slack, np.diag(self.convex_hull_slack_quad)) + self.convex_hull_slack_lin @ convex_hull_slack
        
        if self.soft_state_bound_idxs is not None:
            constraints += [state_ub_slack >= 0, state_lb_slack >= 0]
            objective += 0.5*cp.quad_form(state_ub_slack, np.diag(self.soft_state_bound_quad)) + self.soft_state_bound_lin @ state_ub_slack
            objective += 0.5*cp.quad_form(state_lb_slack, np.diag(self.soft_state_bound_quad)) + self.soft_state_bound_lin @ state_lb_slack

        prob = cp.Problem(cp.Minimize(objective), constraints)
        if self.verbose:
            dt = (time.time()-t)*1000
            self.print_method(f'Construction time: {dt:.2f} ms')
        
        t = time.time()
        success = False
        D_bar = None
        prob.solve(solver='ECOS', verbose=self.verbose)
        if self.verbose:
            dt = (time.time()-t)*1000
            self.print_method(f'Solve time: {dt:.2f} ms')

        t = time.time()
        if prob.status == 'optimal':
            success = True
            qu_sol, du_sol = x.value.T @ np.diag(self.qu_scaling_inv), u.value.T
            D_bar = np.concatenate((qu_sol.ravel(), du_sol.ravel()))
            # self.print_method('Convex hull slack norm: ' + str(np.linalg.norm(convex_hull_slack.value)))
            # if self.soft_state_bound_idxs:
            #     self.print_method('State ub slack norm: ' + str(np.linalg.norm(state_ub_slack.value)))
            #     self.print_method('State lb slack norm: ' + str(np.linalg.norm(state_lb_slack.value)))
        if self.verbose:
            dt = (time.time()-t)*1000
            self.print_method(f'Unpack time: {dt:.2f} ms')
            self.print_method('==============================================')

        return D_bar, success, prob.status

if __name__ == "__main__":
    import pdb

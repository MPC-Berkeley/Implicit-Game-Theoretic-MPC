#!/usr/bin python3

import time, copy
from typing import List, Dict
from collections import deque
import pdb
import os

import numpy as np
import scipy as sp
import scipy.linalg as la

import casadi as ca
import cvxpy as cp

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

from tqdm import tqdm

from mpclab_common.models.dynamics_models import CasadiDynamicsModel
from mpclab_common.pytypes import VehicleState, VehiclePrediction

from mpclab_controllers.abstract_controller import AbstractController
from mpclab_controllers.utils.controllerTypes import CALMPCParams

# import matplotlib
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

class CA_LMPC_NN(AbstractController):
    def __init__(self, dynamics: CasadiDynamicsModel, 
                       costs: Dict[str, List[ca.Function]], 
                       constraints: Dict[str, ca.Function],
                       bounds: Dict[str, VehicleState], 
                       control_params=CALMPCParams(),
                       print_method=print,
                       activation_function='relu',
                       hidden_layer_size=[6]):
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

        self.rng = np.random.default_rng()

        self.hidden_layer_sizes = hidden_layer_size
        self.activation_function = activation_function

        self.epochs = 200

        self.weight_decay = 0

        # Early stopping parameters
        self.patience = 10

        self._initialize_nn_model()
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

    def _initialize_nn_model(self):
        # Instantiate model and send to GPU
        self.model = mlp(input_layer_size=self.n_q,
                    output_layer_size=1,
                    hidden_layer_sizes=self.hidden_layer_sizes, 
                    activation=self.activation_function, 
                    batch_norm=False)
        self.model = self.model.cuda()
        # Define optimization scheme
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3, weight_decay=self.weight_decay)
        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min')
        # Define loss function and send to GPU
        self.loss = nn.MSELoss().cuda()

    def train_model(self, train_percent=0.8):
        iter_costs = []
        for d in self.SS_data:
            iter_costs.append(d['cost_to_go'][0])
        iter_idxs = np.argsort(iter_costs)[:self.n_ss_its]

        self.print_method(f'Training cost-to-go function with data from iterations {iter_idxs}')
        SS_q, SS_u, SS_Q = [], [], []
        for i in iter_idxs:
            SS_q.append(self.SS_data[i]['state'])
            SS_u.append(self.SS_data[i]['input'])
            SS_Q.append(self.SS_data[i]['cost_to_go'])
        
        SS_q = np.vstack(SS_q)
        SS_Q = np.concatenate(SS_Q)
        shuf_idx = self.rng.permutation(len(SS_Q))
        SS_q = SS_q[shuf_idx]
        SS_Q = SS_Q.reshape((-1,1))[shuf_idx]

        train_idx = int(train_percent*SS_q.shape[0])
        self.train_dataset = dataset(SS_q[:train_idx], SS_Q[:train_idx])
        x_mean, x_cov, y_mean, y_cov = self.train_dataset.get_data_stats()
        self.test_dataset = dataset(SS_q[train_idx:], SS_Q[train_idx:], x_mean, x_cov, y_mean, y_cov)

        train_data_loader = DataLoader(self.train_dataset, batch_size=16, shuffle=True, num_workers=12)
        test_data_loader = DataLoader(self.test_dataset, batch_size=len(self.test_dataset), shuffle=True, num_workers=12)

        self.min_loss = np.inf
        self.counter = 0
        self.delta = 0

        for epoch in range(self.epochs):
            # Train model
            train_loss = 0
            self.model.train()
            for data in tqdm(train_data_loader):
                x_batch, y_batch = data
                x_batch = x_batch.cuda()
                y_batch = y_batch.cuda()
                self.model.forward(x_batch)

                self.optimizer.zero_grad()

                y_pred = self.model(x_batch)
                loss_batch = self.loss(y_pred, y_batch)

                loss_batch.backward()
                self.optimizer.step()

                train_loss += loss_batch.item()/len(train_data_loader)

            # Evaluate on test set
            test_loss = 0
            self.model.eval()
            for data in test_data_loader:
                x, y = data
                x = x.cuda()
                y = y.cuda()
                y_pred = self.model(x)
                test_loss += self.loss(y_pred, y).item()/len(test_data_loader)
            
            # scheduler.step(test_loss)

            # Save checkpoint if validation loss improves, otherwise increment patience counter
            if test_loss < self.min_loss - self.delta:
                self.counter = 0
                self.min_loss = test_loss
                # torch.save(self.model.state_dict(), out_path)
                # torch.save(model, out_path)
            else:
                self.counter += 1
            
            self.print_method(f'Epoch {epoch+1} | train loss: {train_loss} | test loss: {test_loss}')

            if self.counter >= self.patience:
                self.print_method(f'Validation loss has not decreased for {self.patience} epochs, stopping training...')
                break
        
        self.model.eval()

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

        W, b = self.model.get_vectorized_weights()
        q_mean, q_cov, Q_mean, Q_cov = self.train_dataset.get_data_stats() 
        q_prec = la.inv(la.sqrtm(q_cov)).ravel(order='F')
        Q_mean = np.array(Q_mean)
        Q_cov = np.sqrt(np.array([Q_cov]))

        D = np.concatenate((np.hstack((q_ws, self.u_ws)).ravel(), 
                            self.du_ws.ravel()))
        P = np.concatenate((q0, um1, W, b, q_mean, q_prec, Q_mean, Q_cov, SS_q.ravel(), SS_u.ravel()))

        for _ in range(self.qp_iters):
            if self.qp_interface == 'casadi':
                D_bar, success, status = self._solve_casadi(D, P)
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

        V_nn, W, b = self.model.get_casadi_symbolic_mlp()
        q_mean = ca.MX.sym('q_mean', self.n_q)
        q_prec = ca.MX.sym('q_cov', self.n_q, self.n_q)
        Q_mean = ca.MX.sym('Q_mean', 1)
        Q_cov = ca.MX.sym('Q_cov', 1)

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
            if k == self.N:
                _q = q_prec @ (q_ph[k] - q_mean)
                Jq_k += V_nn(_q, W, b)*Q_cov + Q_mean
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
        P = [qu0_ph, W, b, q_mean, ca.vec(q_prec), Q_mean, Q_cov]
        
        if self.qp_interface == 'casadi':
            ss_q_ph = [ca.MX.sym(f'ssq_ph_{k}', self.n_q) for k in range(self.n_ss_pts)] # Safe set points
            ss_u_ph = [ca.MX.sym(f'ssu_ph_{k}', self.n_u) for k in range(self.n_ss_pts)] # Safe set points
            a_ph  = ca.MX.sym('a_ph', self.n_ss_pts) # Safe set multipliers
            convex_hull_slack_ph = ca.MX.sym('convex_hull_slack_ph', self.n_q) # Convex hull slack
            D += [a_ph, convex_hull_slack_ph]

            if self.soft_state_bound_idxs is not None:
                state_ub_slack_ph = ca.MX.sym('state_ub_slack_ph', len(self.soft_state_bound_idxs))
                state_lb_slack_ph = ca.MX.sym('state_ub_slack_ph', len(self.soft_state_bound_idxs))
                D += [state_ub_slack_ph, state_lb_slack_ph]
            P += [ca.vertcat(*ss_q_ph), ca.vertcat(*ss_u_ph)]
        
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
            h[s_idx:s_idx+self.n_ss_pts] = 0
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
        else:
            raise(ValueError('QP interface name not recognized'))

    def _solve_casadi(self, D, P):
        if self.verbose:
            self.print_method('============ Sovling using CaSAdi ============')
        t = time.time()
        D = np.concatenate((D, np.zeros(self.n_ss_pts), np.zeros(self.n_q)))
        # P = np.concatenate((P, SS_q.ravel(), SS_u.ravel()))

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

        h       = self.f_h(D, P)
        # H       = self.f_H(D, P)
        H       = self._nearest_pd(self.f_H(D, P))
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

        if self.verbose:
            dt = (time.time()-t)*1000
            self.print_method(f'Unpack time: {dt:.2f} ms')
            self.print_method('==============================================')

        return D_bar, success, status

    def _nearest_pd(self, A):
        B = (A + A.T)/2
        s, U = np.linalg.eigh(B)
        # s[np.where(s < 0)[0]] = 0
        s[np.where(s < 0)[0]] = 1e-9
        C = U @ np.diag(s) @ U.T
        return (C + C.T)/2
    
# Sinusoidal activation
class Sin(nn.Module):
    def forward(self, x):
        return torch.sin(x)

class mlp(nn.Module):
    def __init__(self, input_layer_size=8, output_layer_size=1, hidden_layer_sizes=[100, 100, 100], activation='sin', batch_norm=False):
        super(mlp, self).__init__()
        layers = [nn.Linear(input_layer_size, hidden_layer_sizes[0], dtype=torch.double)]
        
        # Possibly add batch normalization
        if batch_norm:
            layers.append(nn.BatchNorm1d(num_features=hidden_layer_sizes[0], dtype=torch.double))
        
        # Apply activation function
        if activation == 'tanh':
            layers.append(nn.Tanh())
        elif activation == 'relu':
            layers.append(nn.ReLU(inplace=True))
        elif activation == 'leaky_relu':
            layers.append(nn.LeakyReLU(negative_slope=0.01, inplace=True))
        elif activation == 'sin':
            layers.append(Sin())
        
        for i in range(len(hidden_layer_sizes)-1):
            layers.append(nn.Linear(hidden_layer_sizes[i], hidden_layer_sizes[i+1], dtype=torch.double))
            
            # Possibly add batch normalization
            if batch_norm:
                layers.append(nn.BatchNorm1d(num_features=hidden_layer_sizes[0], dtype=torch.double))
            
            # Apply activation function
            if activation == 'tanh':
                layers.append(nn.Tanh())
            elif activation == 'relu':
                layers.append(nn.ReLU(inplace=True))
            elif activation == 'leaky_relu':
                layers.append(nn.LeakyReLU(negative_slope=0.01, inplace=True))
            elif activation == 'sin':
                layers.append(Sin())
        
        layers.append(nn.Linear(hidden_layer_sizes[-1], output_layer_size, dtype=torch.double))
        
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)

    def get_casadi_mlp(self):
        modules = list(self.modules())[2:]
        sym_in = ca.MX.sym('sym_in', modules[0].in_features)
        x = sym_in
        for m in modules:
            if m._get_name() == 'Linear':
                W = m.weight.cpu().detach().numpy()
                b = m.bias.cpu().detach().numpy()
                x = W @ x + b
            if m._get_name() == 'ReLU':
                x = ca.fmax(0, x)
            if m._get_name() == 'Sin':
                x = ca.sin(x)
            if m._get_name() == 'Tanh':
                x = ca.tanh(x)

        return ca.Function('mlp_ca', [sym_in], [x])
    
    def get_vectorized_weights(self):
        modules = list(self.modules())[2:]
        W = [];
        b = [];
        for i, m in enumerate(modules):
            if m._get_name() == 'Linear':
                _W = m.weight.cpu().detach().numpy().ravel(order='F')
                _b = m.bias.cpu().detach().numpy().ravel(order='F')
                W.append(_W)
                b.append(_b)
        return np.concatenate(W), np.concatenate(b)
    
    def get_casadi_symbolic_mlp(self):
        modules = list(self.modules())[2:]
        W = [];
        b = [];
        sym_in = ca.MX.sym('sym_in', modules[0].in_features)
        x = sym_in
        for i, m in enumerate(modules):
            if m._get_name() == 'Linear':
                _W = ca.MX.sym(f'W_{i}', m.weight.cpu().detach().numpy().shape)
                _b = ca.MX.sym(f'b_{i}', m.bias.cpu().detach().numpy().shape[0])
                x = _W @ x + _b
                W.append(ca.vec(_W))
                b.append(ca.vec(_b))
            if m._get_name() == 'ReLU':
                x = ca.fmax(x, 0)
            if m._get_name() == 'Sin':
                x = ca.sin(x)
            if m._get_name() == 'Tanh':
                x = ca.tanh(x)

        return ca.Function('mlp_ca', [sym_in, ca.vertcat(*W), ca.vertcat(*b)], [x]), ca.vertcat(*W), ca.vertcat(*b)

class dataset(Dataset):
    def __init__(self, features, targets, feature_mean=None, feature_cov=None, target_mean=None, target_cov=None):
        self.features = features
        self.targets = targets

        self.n = features.shape[1]
        self.d = targets.shape[1]

        if feature_mean is None:
            self.feature_mean = np.mean(self.features, axis=0)
        else:
            self.feature_mean = feature_mean
        if feature_cov is None:
            self.feature_cov = np.cov(self.features, rowvar=False)
        else:
            self.feature_cov = feature_cov

        if target_mean is None:
            self.target_mean = np.mean(self.targets, axis=0)
        else:
            self.target_mean = target_mean
        if target_cov is None:
            self.target_cov = np.cov(self.targets, rowvar=False)
        else:
            self.target_cov = target_cov

        self.features = la.solve(la.sqrtm(self.feature_cov), (self.features - self.feature_mean).T, assume_a='pos').T
        if self.d > 1:
            self.targets = la.solve(la.sqrtm(self.target_cov), (self.targets - self.target_mean).T, assume_a='pos').T
        else:
            self.targets = (self.targets - self.target_mean)/np.sqrt(self.target_cov)

    def __len__(self):
        return self.features.shape[0]

    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]

    def get_data_stats(self):
        return self.feature_mean, self.feature_cov, self.target_mean, self.target_cov
    
if __name__ == "__main__":
    import pdb

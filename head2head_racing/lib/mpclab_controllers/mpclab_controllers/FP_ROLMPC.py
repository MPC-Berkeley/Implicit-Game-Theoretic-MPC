#!/usr/bin/env python3

import casadi as ca
import numpy as np
import forcespro.nlp
import forcespro

import matplotlib.pyplot as plt

import time, copy
import pdb
from collections import deque
import pickle as pkl
import pathlib

from typing import Tuple, List, Dict

from mpclab_common.models.dynamics_models import CasadiDynamicsModel
from mpclab_common.pytypes import VehicleState, VehicleActuation, VehiclePrediction

from mpclab_common.models.dynamics_models import CasadiDynamicCLBicycle
from mpclab_common.models.model_types import DynamicBicycleConfig

from mpclab_controllers.abstract_controller import AbstractController
from mpclab_controllers.utils.controllerTypes import ROLMPCParams

class FPROLMPC(AbstractController):

    def __init__(self, dynamics: CasadiDynamicsModel,
                       costs: Dict[str, List[ca.Function]],
                       constraints: Dict[str, ca.Function],
                       bounds: Dict[str, VehicleState],
                       control_params=ROLMPCParams(),
                       ws=True,
                       use_Einv=True,
                       print_method=print):
        
        self.dynamics       = dynamics
        self.track          = dynamics.track

        self.track_L        = self.track.track_length
        self.costs          = costs
        self.constraints    = constraints
        self.print_method   = print_method

        self.solver_name    = control_params.solver_name
        self.opt_level      = control_params.opt_level
        self.solver_dir     = control_params.solver_dir
        self.rebuild        = control_params.rebuild
        self.max_iters      = control_params.max_iters

        self.n_u            = self.dynamics.n_u
        self.n_q            = self.dynamics.n_q
        self.n_z            = self.n_q + self.n_u

        self.R              = self.dynamics.R

        self.n_y            = self.R*self.n_u

        self.verbose        = control_params.verbose

        self.dt             = control_params.dt
        self.N              = control_params.N

        self.keep_init_safe_set = control_params.keep_init_safe_set

        self.delay          = control_params.delay
        self.delay_buffer   = []
        if self.delay is None:
            self.delay = np.zeros(self.dynamics.n_u)
            self.delay_buffer = None

        self.use_ws = ws
        self.use_E = use_Einv

        self.n_ss_pts       = control_params.n_ss_pts
        self.n_ss_its       = control_params.n_ss_its

        self.convex_hull_slack_quad = control_params.convex_hull_slack_quad
        self.reachability_slack_quad = control_params.reachability_slack_quad
        self.state_bound_slack_quad = control_params.state_bound_slack_quad

        self.soft_state_bound_idxs = control_params.soft_state_bound_idxs
        self.soft_state_bound_quad = control_params.soft_state_bound_quad
        self.soft_state_bound_lin = control_params.soft_state_bound_lin

        # self.soft_state_bounds = True
        self.soft_state_bounds = False

        self.curv = self.dynamics.get_curvature

        # self.segs=np.array([0.0, 0.83, -0.98])

        # # Change this using error invariant E
        # self.tight_q=ca.diag([0.9-0.09*i for i in range(self.n_q)])
        # self.tight_u=ca.diag([0.8 for i in range(self.n_u)])

        # self.F=err_fb_gain
        # self.n_inv=len(self.F)

        # Process box constraints
        self.state_ub, self.input_ub = self.dynamics.state2qu(bounds['qu_ub'])
        # self.state_ub= ca.DM(self.state_ub).T@self.tight_q
        # self.input_ub= ca.DM(self.input_ub).T@self.tight_u
        self.state_lb, self.input_lb = self.dynamics.state2qu(bounds['qu_lb'])
        # self.state_lb= ca.DM(self.state_lb).T@self.tight_q
        # self.input_lb= ca.DM(self.input_lb).T@self.tight_u

        _, self.input_rate_ub = self.dynamics.state2qu(bounds['du_ub'])
        _, self.input_rate_lb = self.dynamics.state2qu(bounds['du_lb'])

        # self.E_invs=E_invs
        # self.F_invs=F_invs

        path=pathlib.Path("~/barc_data/rpi.pkl").expanduser()
        with open(path, 'rb') as f:
            E_data=pkl.load(f)
        
        self.vel_rngs=E_data["velocity_brackets"]
        self.vel_rngs[0]=0.0
        self.vel_rngs[-1]+=1.
        self.s_rngs=self.track.key_pts[:,3]

        self.E_invs=E_data["chol_shape"]
        self.F_invs=E_data["gain"]

        path = pathlib.Path("~/barc_data/rpi_rate.pkl").expanduser()
        with open(path, 'rb') as f:
            rpi_rate = pkl.load(f)

        self.P_va = rpi_rate['P']
        self.K_da = rpi_rate['K']
        v_max = 0.4
        a_max = 0.5

        P_va_inv = np.linalg.inv(self.P_va)
        P_a_inv = P_va_inv[1,1]/(1-((v_max**2)*P_va_inv[0,0])-2*P_va_inv[0,1]*v_max*a_max)
        self.a_tightening = 1/np.sqrt(P_a_inv)

        self.P_perm=np.array([[0,1,0],[0,0,1],[1,0,0]])

        self._build_invs_constraints()

        self.u_rate_ub=ca.DM([bounds['du_ub'].u.u_a, bounds['du_ub'].u.u_steer ])
        self.u_rate_lb=ca.DM([bounds['du_lb'].u.u_a, bounds['du_lb'].u.u_steer ])
        
        self.SS_Y_sel = None
        self.SS_Q_sel = None
        self.iter_data = []
        self.SS_data = []

        self.q_pred = np.zeros((self.N+1, self.n_q))
        self.u_pred = np.zeros((self.N, self.n_u))
        self.lmbd_pred =np.zeros((1,self.n_ss_its*int(self.n_ss_pts/self.n_ss_its)))

        self.wrapped_state_idxs     = control_params.wrapped_state_idxs
        self.wrapped_state_periods  = control_params.wrapped_state_periods
        self.warm_start_with_nonlinear_rollout = False

        self.q_ws = np.zeros((self.N+1, self.n_q))
        self.u_ws = np.zeros((self.N+1, self.n_u))
        self.lmbd_ws= np.zeros((1, self.n_ss_its*int(self.n_ss_pts/self.n_ss_its)))
        self.Y_ws = np.zeros((1,self.R*self.n_u))

        self.first_solve = True
        self.init_safe_set_iters = []

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

        sim_dynamics_config = DynamicBicycleConfig(dt=self.dt,
                                        model_name='dynamic_bicycle_cl',
                                        noise=False,
                                        discretization_method='rk4',
                                        simple_slip=False,
                                        tire_model='pacejka',
                                        mass=2.2187,
                                        yaw_inertia=0.02723,
                                        wheel_friction=0.9,
                                        pacejka_b_front=5.0,
                                        pacejka_b_rear=5.0,
                                        pacejka_c_front=2.28,
                                        pacejka_c_rear=2.28)
        self.sim_dynamics = CasadiDynamicCLBicycle(0, sim_dynamics_config, self.track)

        if self.solver_dir and not self.rebuild:
            self.solver_dir = pathlib.Path(self.solver_dir).expanduser()  # allow the use of ~
            self._load_solver(self.solver_dir)
        else:
            if self.soft_state_bounds:
                self._build_solver_soft()
            else:
                self._build_solver()
            
    def _update_debug_plot(self, q, u, ss):
        x, y, ss_x, ss_y = [], [], [], []
        for i in range(q.shape[0]):
            xt, yt, _ = self.track.local_to_global((q[i,0], q[i,1], 0))
            x.append(xt); y.append(yt)
        for i in range(ss.shape[0]):
            xt, yt, _ = self.track.local_to_global((ss[i,0], ss[i,1], 0))
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
        
    def add_safe_set_data(self, state_ss, iter_idx=None):

        outputs_ss=self.dynamics.q2y(state_ss)
        lftd_outputs_ss=np.hstack([outputs_ss[i:-self.R+i,:] for i in range(self.R)])

        c2g_output_ss = np.array(self.costs['output'](lftd_outputs_ss.T)).squeeze()
        c2g_output_ss = np.cumsum(c2g_output_ss[::-1])[::-1]

        if iter_idx is not None:
            self.SS_data[iter_idx] = dict(output=lftd_outputs_ss, cost_to_go=c2g_output_ss)
        else:
            self.SS_data.append(dict(output=lftd_outputs_ss, cost_to_go=c2g_output_ss))

    def _add_point_to_safe_set(self, vehicle_state: VehicleState):
        q, u = self.dynamics.state2qu(vehicle_state)
    
        lftd_Y=np.hstack((self.SS_data[-1]['output'][-1,self.n_u:], self.dynamics.q2y(q) ))
        self.SS_data[-1]['output'] = np.vstack((self.SS_data[-1]['output'], lftd_Y.reshape((1,-1))))
        self.SS_data[-1]['cost_to_go'] += self.costs['output'](lftd_Y)
        self.SS_data[-1]['cost_to_go'] = np.append(self.SS_data[-1]['cost_to_go'], self.costs['output'](lftd_Y))
        
    def set_warm_start(self, u_ws: np.ndarray, 
                            q_ws: np.ndarray = None):
        self.q_ws = q_ws
        self.u_ws = u_ws
        self.u_prev = u_ws[1,:]
        self.qb_prev= self.q_pred[0,:]
        
        lmbd_temp=np.zeros((self.n_ss_its, int(self.n_ss_pts/self.n_ss_its)))
        lmbd_temp[:,1:]=self.lmbd_pred.reshape((self.n_ss_its, -1))[:,:-1]
        self.lmbd_ws=lmbd_temp.reshape((1,-1))
        # self.lmbd_ws=self.lmbd_pred

        if self.SS_Y_sel is not None:
            self.Y_ws=self.lmbd_ws@self.SS_Y_sel
        else:
            self.Y_ws=np.zeros((1,self.R*self.n_u))

        if q_ws is None:
            self.q_ws=self._evaluate_dynamics(self.qb_prev, self.u_ws[1:,:])
        
        
        if self.delay_buffer is not None:
            for i in range(self.dynamics.n_u):
                self.delay_buffer.append(deque(self.u_ws[1:1+self.delay[i],i], maxlen=self.delay[i]))

    
    def step(self, vehicle_state: VehicleState, env_state=None):
        self.solve(vehicle_state)

        u = self.u_pred[1,:]
        
        self.dynamics.qu2state(vehicle_state, None, u)

        if self.state_input_prediction is None:
            self.state_input_prediction = VehiclePrediction()
        if self.safe_set is None:
            self.safe_set = VehiclePrediction()
        self.dynamics.qu2prediction(self.state_input_prediction, self.q_pred, self.u_pred[1:,:])
        self.dynamics.Y2prediction(self.safe_set, self.SS_Y_sel)
        self.state_input_prediction.t = vehicle_state.t
        self.safe_set.t = vehicle_state.t
        
        # Update delay buffer
        if self.delay_buffer is not None:
            for i in range(self.dynamics.n_u):
                self.delay_buffer[i].append(u[i])

        # Construct initial guess for next iteration
        u_ws = np.vstack((self.u_pred[1:,:], self.u_pred[-1,:]))
        q_ws = np.vstack((self.q_pred[1:,:], self._evaluate_dynamics(self.q_pred[-1,:], self.u_pred[-1,:])[-1,:]))
        self.set_warm_start(u_ws, q_ws)

        # This method adds the state and input to the safe sets at the previous iteration.
        # The "s" component will be greater than the track length and the cost-to-go will be negative
        state = VehicleState()
        qt, _ = self.dynamics.state2qu(vehicle_state)
        self.dynamics.qu2state(state, qt, self.u_pred[1,:])
        state.p.s += self.track.track_length
        self._add_point_to_safe_set(state)

        return
    
    def solve(self, state: VehicleState, params: np.ndarray = None):
        if self.first_solve:
            self.init_safe_set_iters = np.arange(len(self.SS_data))
            self.first_solve = False

        state.e.psi = np.mod(state.e.psi, 2*np.pi)

        q0, u0 = self.dynamics.state2qu(state)
        # um1 = self.u_prev
        qb_prev=self.qb_prev
        if self.delay_buffer is not None:
            delay_steps = int(np.amin(self.delay))
            u_delay = np.hstack([np.array(self.delay_buffer[i])[:delay_steps].reshape((-1,1)) for i in range(self.dynamics.n_u)])
            beta = float(self.dynamics.fun_beta(u_delay[0,1]))
            _q0 = np.array([state.v.v_long*np.cos(beta), state.v.v_long*np.sin(beta), state.w.w_psi, state.p.e_psi, state.p.s, state.p.x_tran])
            q_bar = self._evaluate_high_fidelity_dynamics(_q0, u_delay)
            q0 = np.array([q_bar[-1,4], q_bar[-1,5], q_bar[-1,3], np.sqrt(q_bar[-1,0]**2+q_bar[-1,1]**2)])
            qb_prev = np.array([q_bar[-2,4], q_bar[-2,5], q_bar[-2,3], np.sqrt(q_bar[-2,0]**2+q_bar[-2,1]**2)])
            # q0 = q_bar[-1]
            # um1 = u_delay[-1]
            # qb_prev=q_bar[-2]

        if self.q_ws is not None:
            q_ws = self.q_ws
            # q_ws = self._evaluate_dynamics(q0, self.u_ws[1:,:])
            q_ws[0,:] = q0

            if self.wrapped_state_idxs is not None:
                for i, p in zip(self.wrapped_state_idxs, self.wrapped_state_periods):
                    q_ws[:,i] = np.unwrap(q_ws[:,i], period=p)
        else:
            q_ws = self._evaluate_dynamics(q0, self.u_ws[1:,:])

        SS_Y, SS_Q = self._select_safe_set(q_ws[-1,:], self.u_ws[-1,:])
        # SS_Y, SS_Q = self._select_safe_set(q_ws[-2,:], self.u_ws[-1,:])
    
        SS_Y = np.vstack(SS_Y)
        SS_Q = np.concatenate(SS_Q)

        self.SS_Y_sel = SS_Y
        self.SS_Q_sel = SS_Q

        # seg=np.argmin(np.abs(self.segs-self.curv(q0[0])))

        E_sched = self._error_inv_schedule(q_ws)
        
        if self.soft_state_bounds:
            qu_sol, success, status = self._solve_forces_soft(q0, u0, qb_prev, E_sched, q_ws, self.u_ws)
        else:
            qu_sol, success, status = self._solve_forces(q0, u0, qb_prev, E_sched, q_ws, self.u_ws)
        if not success:
            self.print_method('Warning: NLP returned ' + str(status))

        if success:
            # Unpack solution
            if self.verbose:
                self.print_method('Current state q: ' + str(q0))
                self.print_method(status)
        
            q_sol = qu_sol[0]
            u_sol = qu_sol[1]

            e = ca.vec(q_sol[0,:])-ca.vec(q0)
            
            vel_idx = int(E_sched[0]/len(self.F_invs[0]))
            arc_idx = int(E_sched[0])-vel_idx*len(self.F_invs[0])
            # pdb.set_trace()
            # print(vel_idx, arc_idx)
            # u_sol[1,1]=u_sol[1,1]-(self.F_invs[vel_idx][arc_idx]@self.P_perm.T@e[:-1])[1]

            # if self.warm_start_with_nonlinear_rollout:
            #     q_sol = self._evaluate_dynamics(q0, u_sol)
            # else:
                # q_sol = qu_sol[:,:self.n_q]
        
            lmbd_sol=qu_sol[2]
        else:
            u_sol = self.u_ws
            q_sol = self._evaluate_dynamics(q0, u_sol[1:,:])
            lmbd_sol=None
        
        # K = -9.01107551
        # K = -9.01107551/10
        # e = (np.array(q0).squeeze()-q_sol[0,:])
        # du_e = K * e[-1]
        # self.print_method(f'error feedback: {du_e:.2f}')
        # u_sol[1,0]=u_sol[1,0] + du_e

        if self.debug_plot:
            self._update_debug_plot(q_sol, u_sol, SS_Y)

        self.q_pred = np.array(q_sol)
        self.u_pred = np.array(u_sol)
        if lmbd_sol is not None:
            self.lmbd_pred=np.array(lmbd_sol)
        else:
            self.lmbd_pred=np.zeros((1,self.n_ss_its*int(self.n_ss_pts/self.n_ss_its)))

    
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

    def _evaluate_high_fidelity_dynamics(self, q0, U):
        t = time.time()
        Q = [q0]
        for k in range(U.shape[0]):
            Q.append(np.array(self.sim_dynamics.fd(Q[k], U[k])).squeeze())
        if self.verbose:
            dt = (time.time()-t)*1000
            self.print_method(f'Dynamics evalution time: {dt:.2f} ms')
        return np.array(Q)
    
    def _select_safe_set(self, q, u, mode='iters'):
        t = time.time()

        if self.verbose:
            self.print_method('Using top '+str(self.n_ss_its)+' iters as safe set')

        n_ss = int(self.n_ss_pts/self.n_ss_its)
        iter_costs = []
        for i in range(len(self.SS_data)):
            iter_costs.append(self.SS_data[i]['cost_to_go'][0])
        iter_idxs = np.argsort(iter_costs)[:self.n_ss_its]

        SS_Y, SS_Q = [], []
        for i in iter_idxs:
            n_data = self.SS_data[i]['output'].shape[0]

            z = q[:self.n_u]
            z_data = self.SS_data[i]['output'][:, :self.n_u]
           
            dist = np.linalg.norm(z_data - np.tile(z, (z_data.shape[0], 1)), ord=1, axis=1)
            min_idx = np.argmin(dist)
            
            if min_idx - int(n_ss/2) < 0:
                SS_idxs = np.arange(n_ss)
            elif min_idx + int(n_ss/2) > n_data:
                SS_idxs = np.arange(n_data-n_ss, n_data)
            else:
                SS_idxs = np.arange(min_idx-int(n_ss/2), min_idx+int(n_ss/2))

            SS_Y.append(self.SS_data[i]['output'][SS_idxs])
            SS_Q.append(self.SS_data[i]['cost_to_go'][SS_idxs])

        if self.verbose:
            dt = (time.time()-t)*1000
            self.print_method(f'Safe set selection time: {dt:.2f} ms')

        return SS_Y, SS_Q
    
    def _build_solver(self):
        def get_state_stage_cost(k):
            if self.costs['state'][k] is not None:
                def _J(z):
                    q_k = z[:self.n_q]
                    return self.costs['state'][k](q_k)
            else:
                self.print_method(f'No state cost for stage {k}')
                def _J(z):
                    return 0
            return _J
        
        def get_input_stage_cost(k):
            if self.costs['input'][k] is not None:
                def _J(z):
                    u_k = z[self.n_q:self.n_q+self.n_u]
                    return self.costs['input'][k](u_k)
            else:
                self.print_method(f'No input cost for stage {k}')
                def _J(z):
                    return 0
            return _J
        
        def get_rate_stage_cost(k):
            if self.costs['rate'][k] is not None:
                def _J(z):
                    u_k = z[self.n_q:self.n_q+self.n_u]
                    u_km1 = z[self.n_q+self.n_u:self.n_q+self.n_u+self.n_u]
                    return self.costs['rate'][k]((u_k - u_km1)/self.dt)
            else:
                self.print_method(f'No rate cost for stage {k}')
                def _J(z):
                    return 0
            return _J
        
        def safe_set_cost_to_go(z, p):
            l = z[self.n_q+self.n_y:self.n_q+self.n_y+self.n_ss_pts]
            SS_Q = p[self.n_ss_pts*self.n_y:self.n_ss_pts*self.n_y+self.n_ss_pts]
            return ca.dot(SS_Q, l)
        
        def safe_set_slack_penalty_cost(z, p):
            safe_set_slack = z[self.n_q+self.n_y+self.n_ss_pts:self.n_q+self.n_y+self.n_ss_pts+self.n_y]
            safe_set_slack_weight = p[self.n_ss_pts*self.n_y+self.n_ss_pts:self.n_ss_pts*self.n_y+self.n_ss_pts+self.n_y]
            return ca.bilin(ca.diag(safe_set_slack_weight), safe_set_slack, safe_set_slack)
        
        def reachability_slack_penalty_cost(z, p):
            reachability_slack = z[self.n_q+self.n_u+self.n_u+self.n_z:self.n_q+self.n_u+self.n_u+self.n_z+self.n_q]
            reachability_slack_weight = p[self.n_q+self.n_q*self.n_z:self.n_q+self.n_q*self.n_z+self.n_q]
            return ca.bilin(ca.diag(reachability_slack_weight), reachability_slack, reachability_slack)
        
        def state_bound_slack_penalty_cost(z, p):
            state_bound_slack = z[self.n_q+self.n_u+self.n_u]
            state_bound_slack_weight = p[self.n_q+self.n_q]
            return state_bound_slack_weight*state_bound_slack**2
        
        # Equality constraint functions
        def state_dynamics_constraint(z):
            q_k = z[:self.n_q]
            u_k = z[self.n_q:self.n_q+self.n_u]
            return self.dynamics.fd(q_k, u_k)
        
        def input_dynamics_constraint(z):
            u_k = z[self.n_q:self.n_q+self.n_u]
            return u_k
        
        def terminal_output_contraint(z):
            q = z[:self.n_q]
            y = z[self.n_q:self.n_q+self.n_y]
            return self.dynamics.fun_Fx_eq(q, y)
        
        def safe_set_multiplier_sum_constraint(z):
            l = z[self.n_q+self.n_y:self.n_q+self.n_y+self.n_ss_pts]
            return ca.sum1(l) - 1

        def safe_set_convex_hull_constraint(z, p):
            y = z[self.n_q:self.n_q+self.n_y]
            l = z[self.n_q+self.n_y:self.n_q+self.n_y+self.n_ss_pts]
            s = z[self.n_q+self.n_y+self.n_ss_pts:self.n_q+self.n_y+self.n_ss_pts+self.n_y]
            SS_Y = ca.reshape(p[:self.n_ss_pts*self.n_y], (self.n_y, self.n_ss_pts))
            return SS_Y @ l + s - y
        
        def rpi_eq_constraint(z, p):
            x = z[self.n_q+self.n_u+self.n_u:self.n_q+self.n_u+self.n_u+self.n_z]
            q_meas = p[:self.n_q]
            Q = ca.reshape(p[self.n_q:self.n_q+self.n_q*self.n_z], (self.n_q, self.n_z))
            return Q @ x + q_meas
        
        # Inequality constraint functions
        def rpi_norm_constraint(z):
            x = z[self.n_q+self.n_u+self.n_u:self.n_q+self.n_u+self.n_u+self.n_z]
            return ca.sumsqr(x) - 1
        
        def input_rate_constraint(z):
            u_k = z[self.n_q:self.n_q+self.n_u]
            u_km1 = z[self.n_q+self.n_u:self.n_q+self.n_u+self.n_u]
            return ca.vertcat((u_k - u_km1)/self.dt - self.input_rate_ub, self.input_rate_lb - (u_k - u_km1)/self.dt)
        
        def initial_state_soft_bounds(z, p):
            q = z[:self.n_q]
            state_bound_slack = z[self.n_q+self.n_u+self.n_u]
            state_ub = p[:self.n_q]
            state_lb = p[self.n_q:self.n_q+self.n_q]
            _ub = q + state_bound_slack*ca.DM.ones(self.n_q) - state_ub
            _lb = state_lb - (q + state_bound_slack*ca.DM.ones(self.n_q))
            return ca.vertcat(_ub, _lb)
        
        # Forces pro model
        self.model = forcespro.nlp.SymbolicModel(self.N+2)

        # Previous stage
        # Decision variables: z_0 = [q_prev, u_{-1}, u_prev, z, reachability_slack]
        self.model.nvar[0] = self.n_q + self.n_u + self.n_u + self.n_z + self.n_q
        self.model.xinitidx = np.concatenate((np.arange(0, self.n_q), np.arange(self.n_q+self.n_u, self.n_q+self.n_u+self.n_u)))
        # Parameters: p_0 = [q_meas, Q, reachability_slack_weight]
        self.model.npar[0] = self.n_q + self.n_q*self.n_z + self.n_q
        # State dynamics
        E_q_dyn = np.hstack((np.eye(self.n_q), np.zeros((self.n_q, self.n_u+self.n_u+1))))
        # Input dynamics
        E_u_dyn = np.hstack((np.zeros((self.n_u, self.n_q+self.n_u)), np.eye(self.n_u), np.zeros((self.n_u, 1))))
        # RPI constraint
        E_rpi = np.hstack((np.eye(self.n_q), np.zeros((self.n_q, self.n_u+self.n_u+1))))
        # Equality constraints E_0 @ z_1 = c_0(z_0, p_0)
        _E = [E_q_dyn, E_u_dyn, E_rpi]
        self.model.E[0] = np.vstack(_E)
        def c0(z, p):
            reachability_slack = z[self.n_q+self.n_u+self.n_u+self.n_z:self.n_q+self.n_u+self.n_u+self.n_z+self.n_q]
            u_prev = z[self.n_q+self.n_u:self.n_q+self.n_u+self.n_u]
            return ca.vertcat(state_dynamics_constraint(z)+reachability_slack, u_prev, rpi_eq_constraint(z, p))
        self.model.eq[0] = c0
        self.model.neq[0] = self.model.E[0].shape[0]
        # Parametric simple bounds
        self.model.lbidx[0] = np.arange(self.n_q, self.n_q+self.n_u)
        self.model.ubidx[0] = np.arange(self.n_q, self.n_q+self.n_u)
        # Cost
        _input_stage_cost = get_input_stage_cost(0)
        self.model.objective[0] = lambda z, p: _input_stage_cost(z) \
                                            + reachability_slack_penalty_cost(z, p)
        # Inequality constraints -inf <= h_0(z_0, p_0) <= 0
        self.model.ineq[0] = lambda z: rpi_norm_constraint(z)
        self.model.hu[0] = 0
        self.model.hl[0] = -np.inf
        self.model.nh[0] = 1

        # Initial stage
        # Decision variables: z_1 = [q_0, u_0, u_prev, state_bound_slack]
        self.model.nvar[1] = self.n_q + self.n_u + self.n_u + 1
        # Parameters: p_1 = [state_ub, state_lb, state_bound_slack_weight]
        self.model.npar[1] = self.n_q + self.n_q + 1
        # State dynamics
        E_q_dyn = np.hstack((np.eye(self.n_q), np.zeros((self.n_q, self.n_u+self.n_u))))
        # Input dynamics
        E_u_dyn = np.hstack((np.zeros((self.n_u, self.n_q+self.n_u)), np.eye(self.n_u)))
        # Equality constraints E_1 @ z_2 = c_1(z_1, p_1)
        _E = [E_q_dyn, E_u_dyn]
        self.model.E[1] = np.vstack(_E)
        self.model.eq[1] = lambda z: ca.vertcat(state_dynamics_constraint(z), input_dynamics_constraint(z))
        self.model.neq[1] = self.model.E[1].shape[0]
        # Parametric simple bounds only on u_0
        self.model.lbidx[1] = np.arange(self.n_q, self.n_q+self.n_u)
        self.model.ubidx[1] = np.arange(self.n_q, self.n_q+self.n_u)
        # Cost
        _state_stage_cost = get_state_stage_cost(0)
        _input_stage_cost = get_input_stage_cost(1)
        _rate_stage_cost = get_rate_stage_cost(0)
        self.model.objective[1] = lambda z, p: _state_stage_cost(z) \
                                            + _input_stage_cost(z) \
                                            + _rate_stage_cost(z) \
                                            + state_bound_slack_penalty_cost(z, p)
        # Inequality constraints -inf <= h_1(z_1, p_1) <= 0
        self.model.ineq[1] = lambda z, p: ca.vertcat(input_rate_constraint(z), initial_state_soft_bounds(z, p))
        self.model.hu[1] = np.zeros(2*self.n_u+2*self.n_q)
        self.model.hl[1] = -np.inf*np.ones(2*self.n_u+2*self.n_q)
        self.model.nh[1] = 2*self.n_u + 2*self.n_q
                
        for k in range(2, self.N-1):
            # Decision variables: z_k = [q_{k-1}, u_{k-1}, u_{k-2}]
            self.model.nvar[k] = self.n_q + self.n_u + self.n_u
            # Parameters: p_k = []
            self.model.npar[k] = 0
            # State dynamics
            E_q_dyn = np.hstack((np.eye(self.n_q), np.zeros((self.n_q, self.n_u+self.n_u))))
            # Input dynamics
            E_u_dyn = np.hstack((np.zeros((self.n_u, self.n_q+self.n_u)), np.eye(self.n_u)))
            # Equality constraints E_k @ z_{k+1} = c_k(z_k, p_k)
            _E = [E_q_dyn, E_u_dyn]
            self.model.E[k] = np.vstack(_E)
            self.model.eq[k] = lambda z: ca.vertcat(state_dynamics_constraint(z), input_dynamics_constraint(z))
            self.model.neq[k] = self.model.E[k].shape[0]
            # Parametric simple bounds
            self.model.lbidx[k] = np.arange(0, self.n_q+self.n_u)
            self.model.ubidx[k] = np.arange(0, self.n_q+self.n_u)
            # Cost
            _state_stage_cost = get_state_stage_cost(k-1)
            _input_stage_cost = get_input_stage_cost(k)
            _rate_stage_cost = get_rate_stage_cost(k-1)
            self.model.objective[k] = lambda z: _state_stage_cost(z) \
                                                + _input_stage_cost(z) \
                                                + _rate_stage_cost(z)
            # Inequality constraints -inf <= h_k(z_k, p_k) <= 0
            self.model.ineq[k] = lambda z: ca.vertcat(input_rate_constraint(z))
            self.model.hu[k] = np.zeros(2*self.n_u)
            self.model.hl[k] = -np.inf*np.ones(2*self.n_u)
            self.model.nh[k] = 2*self.n_u

        # Second to last stage
        # Decision variables: z_{N-1} = [q_{N-2}, u_{N-2}, u_{N-3}]
        self.model.nvar[self.N-1] = self.n_q + self.n_u + self.n_u
        # Parameters: p_{N-1} = []
        self.model.npar[self.N-1] = 0
        # State dynamics
        E_q_dyn = np.hstack((np.eye(self.n_q), np.zeros((self.n_q, self.n_y+self.n_ss_pts+self.n_y))))
        # Equality constraints E_{N-1} @ z_N = c_{N-1}(z_{N-1}, p_{N-1})
        _E = [E_q_dyn]
        self.model.E[self.N-1] = np.vstack(_E)
        self.model.eq[self.N-1] = lambda z: ca.vertcat(state_dynamics_constraint(z))
        self.model.neq[self.N-1] = self.model.E[self.N-1].shape[0]
        # Parametric simple bounds
        self.model.lbidx[self.N-1] = np.arange(0, self.n_q+self.n_u)
        self.model.ubidx[self.N-1] = np.arange(0, self.n_q+self.n_u)
        # Cost
        _state_stage_cost = get_state_stage_cost(self.N-2)
        _input_stage_cost = get_input_stage_cost(self.N-1)
        _rate_stage_cost = get_rate_stage_cost(self.N-2)
        self.model.objective[self.N-1] = lambda z: _state_stage_cost(z) \
                                                    + _input_stage_cost(z) \
                                                    + _rate_stage_cost(z)
        # Inequality constraints -inf <= h_{N-1}(z_{N-1}, p_{N-1}) <= 0
        self.model.ineq[self.N-1] = lambda z: ca.vertcat(input_rate_constraint(z))
        self.model.hu[self.N-1] = np.zeros(2*self.n_u)
        self.model.hl[self.N-1] = -np.inf*np.ones(2*self.n_u)
        self.model.nh[self.N-1] = 2*self.n_u
        
        # Terminal stage
        # Decision variables: z_N = [q_{N-1}, y, lambda, safe_set_slack]
        self.model.nvar[self.N] = self.n_q + self.n_y + self.n_ss_pts + self.n_y
        # Parameters: p_{N-1} = [SS_Y, SS_Q, safe_set_slack_weight]
        self.model.npar[self.N] = self.n_ss_pts*self.n_y + self.n_ss_pts + self.n_y
        # Terminal state to output
        n_fy = self.dynamics.fun_Fx_eq.numel_out(0)
        E_y = np.hstack((np.eye(n_fy), np.zeros((n_fy, self.n_y + 1))))
        # Safe set convex hull
        E_ch = np.hstack((np.zeros((self.n_y, n_fy)), np.eye(self.n_y), np.zeros((self.n_y, 1))))
        # Safe set multiplier
        E_mult = np.concatenate((np.zeros(n_fy + self.n_y), np.array([1]))).reshape((1,-1))
        # Equality constraints E_N @ z_{N+1} = c_N(z_N, p_N)
        _E = [E_y, E_ch, E_mult]
        self.model.E[self.N] = np.vstack(_E)
        self.model.eq[self.N] = lambda z, p: ca.vertcat(terminal_output_contraint(z), safe_set_convex_hull_constraint(z, p), safe_set_multiplier_sum_constraint(z))
        self.model.neq[self.N] = self.model.E[self.N].shape[0]
        # Parametric simple bounds
        self.model.lbidx[self.N] = np.arange(self.n_q+self.n_y, self.n_q+self.n_y+self.n_ss_pts)
        self.model.ubidx[self.N] = np.arange(self.n_q+self.n_y, self.n_q+self.n_y+self.n_ss_pts)
        # Cost
        _state_stage_cost = get_state_stage_cost(self.N-1)
        self.model.objective[self.N] = lambda z, p: _state_stage_cost(z) \
                                                    + safe_set_cost_to_go(z, p) \
                                                    + safe_set_slack_penalty_cost(z, p)
        # Inequality constraints
        self.model.nh[self.N] = 0

        # Output stage
        # Decision variables: z_{N+1} = [output, convex hull, multiplier sum]
        self.model.nvar[self.N+1] = n_fy + self.n_y + 1
        self.model.xfinalidx = range(n_fy + self.n_y + 1)
        # Parameters: p_{N+1} = []
        self.model.npar[self.N+1] = 0
        self.model.nh[self.N+1] = 0
        
        # Build model
        self.options = forcespro.CodeOptions(self.solver_name)
        self.options.overwrite = 1
        self.options.printlevel = 2 if self.verbose else 0
        self.options.optlevel = self.opt_level
        self.options.BuildSimulinkBlock = False
        self.options.cleanup = True
        self.options.platform = 'Generic'
        self.options.gnu = True
        self.options.sse = True
        self.options.noVariableElimination = True
        self.options.maxit = self.max_iters
        self.options.solver_timeout = 1
        self.options.server = 'https://forces-6-0-1.embotech.com'

        self.options.nlp.stack_parambounds = True
        self.options.nlp.linear_solver = 'normal_eqs'
        self.options.nlp.TolStat = 1e-6
        self.options.nlp.TolEq = 1e-6
        self.options.nlp.TolIneq = 1e-6
        
        self.solver = self.model.generate_solver(self.options)

    def _load_solver(self, solver_dir):
        self.solver = forcespro.nlp.Solver.from_directory(solver_dir)

    def _solve_forces(self, qt, ut, qb_prev, E_sched, q_ws, u_ws):
        # Tightened state and input bounds
        q_lb, q_ub, u_lb, u_ub = [], [], [], []
        for k in range(self.N):
            q_lb.append(np.array(self.fq_lb_cond(E_sched[k],0)).squeeze())
            q_ub.append(np.array(self.fq_ub_cond(E_sched[k],0)).squeeze())
            u_lb.append(np.array(self.fu_lb_cond(E_sched[k],0)).squeeze())
            u_ub.append(np.array(self.fu_ub_cond(E_sched[k],0)).squeeze())

        n_fy = self.dynamics.fun_Fx_eq.numel_out(0)
        
        problem = dict()

        # Previous nominal state
        problem['xinit'] = np.concatenate((qb_prev, ut))

        # Output, safe set, and convex hull multiplier constraints should all be equal to zero
        problem['xfinal'] = np.zeros(n_fy + self.n_y + 1)

        # Bounds on previous input
        ub, lb = [self.input_ub], [self.input_lb]
        # Bounds on initial input
        ub += [u_ub[0]]
        lb += [u_lb[0]]
        # Tightened bounds on states and inputs over the horizon
        for k in range(1, self.N-1):
            ub += [q_ub[k], u_ub[k]]
            lb += [q_lb[k], u_lb[k]]
        # Bounds on convex hull multipliers
        ub.append(np.ones(self.n_ss_pts))
        lb.append(np.zeros(self.n_ss_pts))
        problem['ub'] = np.concatenate(ub)
        problem['lb'] = np.concatenate(lb)

        # Parameters
        parameters = []
        # RPI parameters
        Q = np.array(self.f_E_cond(E_sched[0], 0))
        parameters += [qt, Q.ravel(order='F')]
        # Reachability slack
        parameters += [self.reachability_slack_quad]
        # Initial state bounds
        parameters += [q_ub[0], q_lb[0]]
        # Initial state slack
        parameters += [self.state_bound_slack_quad]
        # Safe set parameters
        parameters += [self.SS_Y_sel.ravel(), self.SS_Q_sel, self.convex_hull_slack_quad]
        problem['all_parameters'] = np.concatenate(parameters)

        # Initial guess
        if self.use_ws:
            y_ws = self.Y_ws.squeeze()
            l_ws = self.lmbd_ws.squeeze()
        else:
            y_ws = np.zeros(self.n_y)
            l_ws = np.zeros(self.n_ss_pts)

        initial_guess = [qb_prev, ut, ut, np.zeros(self.n_z), np.zeros(self.n_q)]
        initial_guess += [q_ws[0], u_ws[1], u_ws[0], np.zeros(1)]
        for k in range(1, self.N-1):
            initial_guess += [q_ws[k], u_ws[k+1], u_ws[k]]
        initial_guess += [q_ws[-1], y_ws, l_ws, np.zeros(self.n_y)]
        initial_guess += [np.zeros(n_fy + self.n_y + 1)]
        problem['x0'] = np.concatenate(initial_guess)

        problem['solver_timeout'] = 0.9*self.dt

        output, exitflag, solve_info = self.solver.solve(problem)

        if exitflag == 1:
            success = True
            status = 'Successfully Solved'
            
            if self.N+2 < 10:
                stage_key = lambda x: 'x%d' % x
            else:
                stage_key = lambda x: 'x%02d' % x

            # Unpack solution
            q_sol, u_sol = [], [ut]
            for k in range(self.N-1):
                sol_k = output[stage_key(k+2)]
                q_sol.append(sol_k[:self.n_q])
                u_sol.append(sol_k[self.n_q:self.n_q+self.n_u])
            sol_N = output[stage_key(self.N+1)]
            q_sol.append(sol_N[:self.n_q]) 
            l_sol = sol_N[self.n_q+self.n_y:self.n_q+self.n_y+self.n_ss_pts]
            q_sol = np.array(q_sol)
            u_sol = np.array(u_sol)

            # y_sol = sol_N[self.n_q:self.n_q+self.n_y]
            # ss_s_sol = sol_N[self.n_q+self.n_y+self.n_ss_pts:self.n_q+self.n_y+self.n_ss_pts+self.n_y]

            # sol_0 = output[stage_key(1)]
            # reach_s_sol = sol_0[self.n_q+self.n_u+self.n_u+self.n_z:self.n_q+self.n_u+self.n_u+self.n_z+self.n_q]
            # sol_1 = output[stage_key(2)]
            # bound_s_sol = sol_1[self.n_q+self.n_u+self.n_u]
            # self.print_method('Reachability slack:')
            # self.print_method(str(reach_s_sol))
            # self.print_method('Initial state bound slack:')
            # self.print_method(str(bound_s_sol))
            # self.print_method('Safe set slack:')
            # self.print_method(str(ss_s_sol))
        else:
            success = False
            status = f'Solving Failed, exitflag = {exitflag}'
            q_sol = q_ws
            u_sol = u_ws
            l_sol = l_ws

        return [q_sol, u_sol, l_sol], success, status
    
    def _error_inv_schedule(self, q_ws):
        #temporary speed segementation
        
        n_seg=self.s_rngs.shape[0]-1
        sched=np.zeros((self.N+1,1))

        for i in range(q_ws.shape[0]):

            speed_idx=np.argmin(np.abs(self.vel_rngs-q_ws[i,-1]))
            if self.vel_rngs[speed_idx]>q_ws[i,-1]:
                speed_idx=speed_idx-1
            sched[i]=speed_idx

            arc_idx=np.argmin(np.abs(self.s_rngs-min(q_ws[i,0],self.track_L)))

            if self.s_rngs[arc_idx]>q_ws[i,0]:
                arc_idx=arc_idx-1
            sched[i]=sched[i]*n_seg+arc_idx      

        return sched

    def _build_invs_constraints(self):

        n_vel=len(self.E_invs)
        n_seg=len(self.E_invs[0])

        self.tight_q_ub=[copy.deepcopy(self.state_ub) for _ in range(n_seg*n_vel)]
        self.tight_q_lb=[copy.deepcopy(self.state_lb) for _ in range(n_seg*n_vel)]
        self.tight_u_ub=[copy.deepcopy(self.input_ub) for _ in range(n_seg*n_vel)]
        self.tight_u_lb=[copy.deepcopy(self.input_lb) for _ in range(n_seg*n_vel)]

        sym_d=ca.MX.sym("dummy")

        sym_q=ca.MX.sym("q", self.n_q)

        fq_ub=[]
        fq_lb=[]
        fu_ub=[]
        fu_lb=[]

        fq_def_ub=ca.Function("default_q_constraint", [sym_d], [self.state_ub])
        fq_def_lb=ca.Function("default_q_constraint", [sym_d], [self.state_lb])
        fu_def_ub=ca.Function("default_q_constraint", [sym_d], [self.input_ub])
        fu_def_lb=ca.Function("default_q_constraint", [sym_d], [self.input_lb])

        f_E_inv=[]
        P_def=ca.inv(1e5*ca.DM.eye(3))
        Q_q_def=ca.vertcat(ca.chol(P_def).T,ca.DM([(self.n_q-1)*[1.0]]))
        f_E_def= ca.Function("def_f_e_inv", [sym_d], [Q_q_def])


        for i in range(n_seg*n_vel):
                v_idx=int(i/n_seg)
                a_idx=i-v_idx*n_seg
                Q=self.P_perm@self.E_invs[v_idx][a_idx].T
                Q_u=self.F_invs[v_idx][a_idx]@self.P_perm.T@Q
                Q_v=Q_u[0,:]
                Q_del=Q_u[1,:]
                Q_q=np.vstack((Q,Q_v))
                # P=Q_q.T@Q_q
                # Q=np.linalg.inv(Q)
                if self.use_E:
                    f_E_inv.append(ca.Function("einv", [sym_d], [Q_q]))
                    for k in range(1,self.n_q):
                        norm_k=np.linalg.norm(Q_q[k,:], 2)
                        self.tight_q_ub[i][k]-=norm_k
                        self.tight_q_lb[i][k]+=norm_k
                    
                    # for k in range(1,self.n_u):
                    # P= 0.30788795
                    # norm_v= P**0.5
                    # # norm_v= 2*P**0.5
                    # self.tight_u_ub[i][0]-=norm_v
                    # self.tight_u_lb[i][0]+=norm_v

                    # norm_del=np.sqrt(self.n_q-1)*np.linalg.norm(Q_del, np.inf)
                    # self.tight_u_ub[i][1]-=norm_del
                    # self.tight_u_lb[i][1]+=norm_del
                        
                else:
                    f_E_inv.append(ca.Function("einv", [sym_d], [Q_q_def]))


                fq_ub.append(ca.Function("fqub", [sym_d], [self.tight_q_ub[i]]))
                fq_lb.append(ca.Function("fqlb", [sym_d], [self.tight_q_lb[i]]))               

                fu_ub.append(ca.Function("fuub", [sym_d], [self.tight_u_ub[i]]))
                fu_lb.append(ca.Function("fulb", [sym_d], [self.tight_u_lb[i]]))

        self.fq_ub_cond = ca.Function.conditional("fq_ub_sched", fq_ub, fq_def_ub)
        self.fq_lb_cond = ca.Function.conditional("fq_ub_sched", fq_lb, fq_def_lb)
        self.fu_ub_cond = ca.Function.conditional("fu_ub_sched", fu_ub, fu_def_ub)
        self.fu_lb_cond = ca.Function.conditional("fu_lb_sched", fu_lb, fu_def_lb)
        self.f_E_cond   = ca.Function.conditional("f_E_sched",   f_E_inv, f_E_def)

        self.n_z = 3
    
    def _build_solver_soft(self):
        def get_state_stage_cost(k):
            if self.costs['state'][k] is not None:
                def _J(z):
                    q_k = z[:self.n_q]
                    return self.costs['state'][k](q_k)
            else:
                self.print_method(f'No state cost for stage {k}')
                def _J(z):
                    return 0
            return _J
        
        def get_input_stage_cost(k):
            if self.costs['input'][k] is not None:
                def _J(z):
                    u_k = z[self.n_q:self.n_q+self.n_u]
                    return self.costs['input'][k](u_k)
            else:
                self.print_method(f'No input cost for stage {k}')
                def _J(z):
                    return 0
            return _J
        
        def get_rate_stage_cost(k):
            if self.costs['rate'][k] is not None:
                def _J(z):
                    u_k = z[self.n_q:self.n_q+self.n_u]
                    u_km1 = z[self.n_q+self.n_u:self.n_q+self.n_u+self.n_u]
                    return self.costs['rate'][k]((u_k - u_km1)/self.dt)
            else:
                self.print_method(f'No rate cost for stage {k}')
                def _J(z):
                    return 0
            return _J
        
        def safe_set_cost_to_go(z, p):
            l = z[self.n_q+self.n_y:self.n_q+self.n_y+self.n_ss_pts]
            SS_Q = p[self.n_ss_pts*self.n_y:self.n_ss_pts*self.n_y+self.n_ss_pts]
            return ca.dot(SS_Q, l)
        
        def safe_set_slack_penalty_cost(z, p):
            safe_set_slack = z[self.n_q+self.n_y+self.n_ss_pts:self.n_q+self.n_y+self.n_ss_pts+self.n_y]
            safe_set_slack_weight = p[self.n_ss_pts*self.n_y+self.n_ss_pts:self.n_ss_pts*self.n_y+self.n_ss_pts+self.n_y]
            return ca.bilin(ca.diag(safe_set_slack_weight), safe_set_slack, safe_set_slack)
        
        def reachability_slack_penalty_cost(z, p):
            reachability_slack = z[self.n_q+self.n_u+self.n_u+self.n_z:self.n_q+self.n_u+self.n_u+self.n_z+self.n_q]
            reachability_slack_weight = p[self.n_q+self.n_q*self.n_z:self.n_q+self.n_q*self.n_z+self.n_q]
            return ca.bilin(ca.diag(reachability_slack_weight), reachability_slack, reachability_slack)
        
        def state_bound_slack_penalty_cost(z, p):
            state_bound_slack = z[self.n_q+self.n_u+self.n_u:self.n_q+self.n_u+self.n_u+self.n_q]
            state_bound_slack_weight = p[self.n_q+self.n_q]
            return state_bound_slack_weight*ca.sumsqr(state_bound_slack)
        
        # Equality constraint functions
        def state_dynamics_constraint(z):
            q_k = z[:self.n_q]
            u_k = z[self.n_q:self.n_q+self.n_u]
            return self.dynamics.fd(q_k, u_k)
        
        def input_dynamics_constraint(z):
            u_k = z[self.n_q:self.n_q+self.n_u]
            return u_k
        
        def terminal_output_contraint(z):
            q = z[:self.n_q]
            y = z[self.n_q:self.n_q+self.n_y]
            return self.dynamics.fun_Fx_eq(q, y)
        
        def safe_set_multiplier_sum_constraint(z):
            l = z[self.n_q+self.n_y:self.n_q+self.n_y+self.n_ss_pts]
            return ca.sum1(l) - 1

        def safe_set_convex_hull_constraint(z, p):
            y = z[self.n_q:self.n_q+self.n_y]
            l = z[self.n_q+self.n_y:self.n_q+self.n_y+self.n_ss_pts]
            s = z[self.n_q+self.n_y+self.n_ss_pts:self.n_q+self.n_y+self.n_ss_pts+self.n_y]
            SS_Y = ca.reshape(p[:self.n_ss_pts*self.n_y], (self.n_y, self.n_ss_pts))
            return SS_Y @ l + s - y
        
        def rpi_eq_constraint(z, p):
            x = z[self.n_q+self.n_u+self.n_u:self.n_q+self.n_u+self.n_u+self.n_z]
            q_meas = p[:self.n_q]
            Q = ca.reshape(p[self.n_q:self.n_q+self.n_q*self.n_z], (self.n_q, self.n_z))
            return Q @ x + q_meas
        
        # Inequality constraint functions
        def rpi_norm_constraint(z):
            x = z[self.n_q+self.n_u+self.n_u:self.n_q+self.n_u+self.n_u+self.n_z]
            return ca.sumsqr(x) - 1
        
        def input_rate_constraint(z):
            u_k = z[self.n_q:self.n_q+self.n_u]
            u_km1 = z[self.n_q+self.n_u:self.n_q+self.n_u+self.n_u]
            return ca.vertcat((u_k - u_km1)/self.dt - self.input_rate_ub, self.input_rate_lb - (u_k - u_km1)/self.dt)
        
        def state_soft_bounds(z, p):
            q = z[:self.n_q]
            state_bound_slack = z[self.n_q+self.n_u+self.n_u:self.n_q+self.n_u+self.n_u+self.n_q]
            state_ub = p[:self.n_q]
            state_lb = p[self.n_q:self.n_q+self.n_q]
            _ub = q + state_bound_slack - state_ub
            _lb = state_lb - (q + state_bound_slack)
            return ca.vertcat(_ub, _lb)
        
        # Forces pro model
        self.model = forcespro.nlp.SymbolicModel(self.N+2)

        # Previous stage
        # Decision variables: z_0 = [q_prev, u_{-1}, u_prev, z, reachability_slack]
        self.model.nvar[0] = self.n_q + self.n_u + self.n_u + self.n_z + self.n_q
        self.model.xinitidx = np.concatenate((np.arange(0, self.n_q), np.arange(self.n_q+self.n_u, self.n_q+self.n_u+self.n_u)))
        # Parameters: p_0 = [q_meas, Q, reachability_slack_weight]
        self.model.npar[0] = self.n_q + self.n_q*self.n_z + self.n_q
        # State dynamics
        E_q_dyn = np.hstack((np.eye(self.n_q), np.zeros((self.n_q, self.n_u+self.n_u))))
        # Input dynamics
        E_u_dyn = np.hstack((np.zeros((self.n_u, self.n_q+self.n_u)), np.eye(self.n_u)))
        # RPI constraint
        E_rpi = np.hstack((np.eye(self.n_q), np.zeros((self.n_q, self.n_u+self.n_u))))
        # Equality constraints E_0 @ z_1 = c_0(z_0, p_0)
        _E = np.vstack([E_q_dyn, E_u_dyn, E_rpi])
        self.model.E[0] = np.hstack((_E, np.zeros((_E.shape[0],self.n_q))))
        def c0(z, p):
            reachability_slack = z[self.n_q+self.n_u+self.n_u+self.n_z:self.n_q+self.n_u+self.n_u+self.n_z+self.n_q]
            u_prev = z[self.n_q+self.n_u:self.n_q+self.n_u+self.n_u]
            return ca.vertcat(state_dynamics_constraint(z)+reachability_slack, u_prev, rpi_eq_constraint(z, p))
        self.model.eq[0] = c0
        self.model.neq[0] = self.model.E[0].shape[0]
        # Parametric simple bounds
        self.model.lbidx[0] = np.arange(self.n_q, self.n_q+self.n_u)
        self.model.ubidx[0] = np.arange(self.n_q, self.n_q+self.n_u)
        # Cost
        _input_stage_cost = get_input_stage_cost(0)
        self.model.objective[0] = lambda z, p: _input_stage_cost(z) \
                                            + reachability_slack_penalty_cost(z, p)
        # Inequality constraints -inf <= h_0(z_0, p_0) <= 0
        self.model.ineq[0] = lambda z: rpi_norm_constraint(z)
        self.model.hu[0] = 0
        self.model.hl[0] = -np.inf
        self.model.nh[0] = 1

        # Initial stage
        # Decision variables: z_1 = [q_0, u_0, u_prev, state_bound_slack]
        self.model.nvar[1] = self.n_q + self.n_u + self.n_u + self.n_q
        # Parameters: p_1 = [state_ub, state_lb, state_bound_slack_weight]
        self.model.npar[1] = self.n_q + self.n_q + 1
       
        # State dynamics
        E_q_dyn = np.hstack((np.eye(self.n_q), np.zeros((self.n_q, self.n_u+self.n_u))))
        # Input dynamics
        E_u_dyn = np.hstack((np.zeros((self.n_u, self.n_q+self.n_u)), np.eye(self.n_u)))
        # Equality constraints E_1 @ z_2 = c_1(z_1, p_1)
        _E = np.vstack([E_q_dyn, E_u_dyn])
        self.model.E[1] = np.hstack((_E, np.zeros((_E.shape[0],self.n_q))))
        self.model.eq[1] = lambda z: ca.vertcat(state_dynamics_constraint(z), input_dynamics_constraint(z))
        self.model.neq[1] = self.model.E[1].shape[0]
        # Parametric simple bounds only on u_0
        self.model.lbidx[1] = np.arange(self.n_q, self.n_q+self.n_u)
        self.model.ubidx[1] = np.arange(self.n_q, self.n_q+self.n_u)
        # Cost
        _state_stage_cost = get_state_stage_cost(0)
        _input_stage_cost = get_input_stage_cost(1)
        _rate_stage_cost = get_rate_stage_cost(0)
        self.model.objective[1] = lambda z, p: _state_stage_cost(z) \
                                            + _input_stage_cost(z) \
                                            + _rate_stage_cost(z) \
                                            + state_bound_slack_penalty_cost(z, p)
        # Inequality constraints -inf <= h_1(z_1, p_1) <= 0
        self.model.ineq[1] = lambda z, p: ca.vertcat(input_rate_constraint(z), state_soft_bounds(z, p))
        self.model.hu[1] = np.zeros(2*self.n_u+2*self.n_q)
        self.model.hl[1] = -np.inf*np.ones(2*self.n_u+2*self.n_q)
        self.model.nh[1] = 2*self.n_u + 2*self.n_q
                
        for k in range(2, self.N-1):
            # Decision variables: z_k = [q_{k-1}, u_{k-1}, u_{k-2}, state_bound_slack]
            self.model.nvar[k] = self.n_q + self.n_u + self.n_u + self.n_q
            # Parameters: p_k = [state_ub, state_lb, state_bound_slack_weight]
            self.model.npar[k] = self.n_q + self.n_q + 1
            # State dynamics
            E_q_dyn = np.hstack((np.eye(self.n_q), np.zeros((self.n_q, self.n_u+self.n_u))))
            # Input dynamics
            E_u_dyn = np.hstack((np.zeros((self.n_u, self.n_q+self.n_u)), np.eye(self.n_u)))
            # Equality constraints E_k @ z_{k+1} = c_k(z_k, p_k)
            _E = np.vstack([E_q_dyn, E_u_dyn])
            self.model.E[k] = np.hstack((_E, np.zeros((_E.shape[0],self.n_q))))
            self.model.eq[k] = lambda z: ca.vertcat(state_dynamics_constraint(z), input_dynamics_constraint(z))
            self.model.neq[k] = self.model.E[k].shape[0]
            # Parametric simple bounds only on u_{k-1}
            self.model.lbidx[k] = np.arange(self.n_q, self.n_q+self.n_u)
            self.model.ubidx[k] = np.arange(self.n_q, self.n_q+self.n_u)
            # Cost
            _state_stage_cost = get_state_stage_cost(k-1)
            _input_stage_cost = get_input_stage_cost(k)
            _rate_stage_cost = get_rate_stage_cost(k-1)
            self.model.objective[k] = lambda z, p: _state_stage_cost(z) \
                                                + _input_stage_cost(z) \
                                                + _rate_stage_cost(z) \
                                                + state_bound_slack_penalty_cost(z, p)
            # Inequality constraints -inf <= h_k(z_k, p_k) <= 0
            self.model.ineq[k] = lambda z, p: ca.vertcat(input_rate_constraint(z), state_soft_bounds(z, p))
            self.model.hu[k] = np.zeros(2*self.n_u+2*self.n_q)
            self.model.hl[k] = -np.inf*np.ones(2*self.n_u+2*self.n_q)
            self.model.nh[k] = 2*self.n_u + 2*self.n_q

        # Second to last stage
        # Decision variables: z_{N-1} = [q_{N-2}, u_{N-2}, u_{N-3}, state_bound_slack]
        self.model.nvar[self.N-1] = self.n_q + self.n_u + self.n_u + self.n_q
        # Parameters: p_{N-1} = [state_ub, state_lb, state_bound_slack_weight]
        self.model.npar[self.N-1] = self.n_q + self.n_q + 1
        # State dynamics
        E_q_dyn = np.hstack((np.eye(self.n_q), np.zeros((self.n_q, self.n_y+self.n_ss_pts+self.n_y))))
        # Equality constraints E_{N-1} @ z_N = c_{N-1}(z_{N-1}, p_{N-1})
        _E = [E_q_dyn]
        self.model.E[self.N-1] = np.vstack(_E)
        self.model.eq[self.N-1] = lambda z: ca.vertcat(state_dynamics_constraint(z))
        self.model.neq[self.N-1] = self.model.E[self.N-1].shape[0]
        # Parametric simple bounds only on u_{k-1}
        self.model.lbidx[self.N-1] = np.arange(self.n_q, self.n_q+self.n_u)
        self.model.ubidx[self.N-1] = np.arange(self.n_q, self.n_q+self.n_u)
        # Cost
        _state_stage_cost = get_state_stage_cost(self.N-2)
        _input_stage_cost = get_input_stage_cost(self.N-1)
        _rate_stage_cost = get_rate_stage_cost(self.N-2)
        self.model.objective[self.N-1] = lambda z, p: _state_stage_cost(z) \
                                                    + _input_stage_cost(z) \
                                                    + _rate_stage_cost(z) \
                                                    + state_bound_slack_penalty_cost(z, p)
        # Inequality constraints -inf <= h_{N-1}(z_{N-1}, p_{N-1}) <= 0
        self.model.ineq[self.N-1] = lambda z, p: ca.vertcat(input_rate_constraint(z), state_soft_bounds(z, p))
        self.model.hu[self.N-1] = np.zeros(2*self.n_u+2*self.n_q)
        self.model.hl[self.N-1] = -np.inf*np.ones(2*self.n_u+2*self.n_q)
        self.model.nh[self.N-1] = 2*self.n_u + 2*self.n_q
        
        # Terminal stage
        # Decision variables: z_N = [q_{N-1}, y, lambda, safe_set_slack]
        self.model.nvar[self.N] = self.n_q + self.n_y + self.n_ss_pts + self.n_y
        # Parameters: p_{N-1} = [SS_Y, SS_Q, safe_set_slack_weight]
        self.model.npar[self.N] = self.n_ss_pts*self.n_y + self.n_ss_pts + self.n_y
        # Terminal state to output
        n_fy = self.dynamics.fun_Fx_eq.numel_out(0)
        E_y = np.hstack((np.eye(n_fy), np.zeros((n_fy, self.n_y + 1))))
        # Safe set convex hull
        E_ch = np.hstack((np.zeros((self.n_y, n_fy)), np.eye(self.n_y), np.zeros((self.n_y, 1))))
        # Safe set multiplier
        E_mult = np.concatenate((np.zeros(n_fy + self.n_y), np.array([1]))).reshape((1,-1))
        # Equality constraints E_N @ z_{N+1} = c_N(z_N, p_N)
        _E = [E_y, E_ch, E_mult]
        self.model.E[self.N] = np.vstack(_E)
        self.model.eq[self.N] = lambda z, p: ca.vertcat(terminal_output_contraint(z), safe_set_convex_hull_constraint(z, p), safe_set_multiplier_sum_constraint(z))
        self.model.neq[self.N] = self.model.E[self.N].shape[0]
        # Parametric simple bounds
        self.model.lbidx[self.N] = np.arange(self.n_q+self.n_y, self.n_q+self.n_y+self.n_ss_pts)
        self.model.ubidx[self.N] = np.arange(self.n_q+self.n_y, self.n_q+self.n_y+self.n_ss_pts)
        # Cost
        _state_stage_cost = get_state_stage_cost(self.N-1)
        self.model.objective[self.N] = lambda z, p: _state_stage_cost(z) \
                                                    + safe_set_cost_to_go(z, p) \
                                                    + safe_set_slack_penalty_cost(z, p)
        # Inequality constraints
        self.model.nh[self.N] = 0

        # Output stage
        # Decision variables: z_{N+1} = [output, convex hull, multiplier sum]
        self.model.nvar[self.N+1] = n_fy + self.n_y + 1
        self.model.xfinalidx = range(n_fy + self.n_y + 1)
        # Parameters: p_{N+1} = []
        self.model.npar[self.N+1] = 0
        self.model.nh[self.N+1] = 0
        
        # Build model
        self.options = forcespro.CodeOptions(self.solver_name)
        self.options.overwrite = 1
        self.options.printlevel = 2 if self.verbose else 0
        self.options.optlevel = self.opt_level
        self.options.BuildSimulinkBlock = False
        self.options.cleanup = True
        self.options.platform = 'Generic'
        self.options.gnu = True
        self.options.sse = True
        self.options.noVariableElimination = True
        self.options.maxit = self.max_iters
        self.options.solver_timeout = 1
        self.options.server = 'https://forces-6-0-1.embotech.com'

        self.options.nlp.stack_parambounds = True
        self.options.nlp.linear_solver = 'normal_eqs'
        self.options.nlp.TolStat = 1e-6
        self.options.nlp.TolEq = 1e-6
        self.options.nlp.TolIneq = 1e-6
        
        self.solver = self.model.generate_solver(self.options)

    def _solve_forces_soft(self, qt, ut, qb_prev, E_sched, q_ws, u_ws):
        mult = np.array([1, 1, 1, 1])
        # Tightened state and input bounds
        q_lb, q_ub, u_lb, u_ub = [], [], [], []
        for k in range(self.N):
            q_lb.append(np.multiply(np.array(self.fq_lb_cond(E_sched[k],0)).squeeze(), mult))
            q_ub.append(np.multiply(np.array(self.fq_ub_cond(E_sched[k],0)).squeeze(), mult))
            u_lb.append(np.array(self.fu_lb_cond(E_sched[k],0)).squeeze())
            u_ub.append(np.array(self.fu_ub_cond(E_sched[k],0)).squeeze())

        n_fy = self.dynamics.fun_Fx_eq.numel_out(0)
        
        problem = dict()

        # Previous nominal state
        problem['xinit'] = np.concatenate((qb_prev, ut))

        # Output, safe set, and convex hull multiplier constraints should all be equal to zero
        problem['xfinal'] = np.zeros(n_fy + self.n_y + 1)

        # Bounds on previous input
        ub, lb = [self.input_ub], [self.input_lb]
        # Bounds on initial input
        ub += [u_ub[0]]
        lb += [u_lb[0]]
        # Tightened bounds on states and inputs over the horizon
        for k in range(1, self.N-1):
            ub += [u_ub[k]]
            lb += [u_lb[k]]
        # Bounds on convex hull multipliers
        ub.append(np.ones(self.n_ss_pts))
        lb.append(np.zeros(self.n_ss_pts))
        problem['ub'] = np.concatenate(ub)
        problem['lb'] = np.concatenate(lb)

        # Parameters
        parameters = []
        # RPI parameters
        Q = np.array(self.f_E_cond(E_sched[0], 0))
        parameters += [qt, Q.ravel(order='F')]
        # Reachability slack
        parameters += [self.reachability_slack_quad]
        # Soft state bounds and slack penalty weight
        for k in range(self.N-1):
            parameters += [q_ub[k], q_lb[k], self.state_bound_slack_quad]
        # Safe set parameters
        parameters += [self.SS_Y_sel.ravel(), self.SS_Q_sel, self.convex_hull_slack_quad]
        problem['all_parameters'] = np.concatenate(parameters)

        # Initial guess
        if self.use_ws:
            y_ws = self.Y_ws.squeeze()
            l_ws = self.lmbd_ws.squeeze()
        else:
            y_ws = np.zeros(self.n_y)
            l_ws = np.zeros(self.n_ss_pts)

        initial_guess = [qb_prev, ut, ut, np.zeros(self.n_z), np.zeros(self.n_q)]
        for k in range(self.N-1):
            initial_guess += [q_ws[k], u_ws[k+1], u_ws[k], np.zeros(self.n_q)]
        initial_guess += [q_ws[-1], y_ws, l_ws, np.zeros(self.n_y)]
        initial_guess += [np.zeros(n_fy + self.n_y + 1)]
        problem['x0'] = np.concatenate(initial_guess)

        problem['solver_timeout'] = 0.9*self.dt
        # problem['solver_timeout'] = 2*self.dt

        output, exitflag, solve_info = self.solver.solve(problem)

        if exitflag in [1]:
            success = True
            # status = 'Successfully Solved'
            
            if self.N+2 < 10:
                stage_key = lambda x: 'x%d' % x
            else:
                stage_key = lambda x: 'x%02d' % x

            # Unpack solution
            q_sol, u_sol = [], [ut]
            for k in range(self.N-1):
                sol_k = output[stage_key(k+2)]
                q_sol.append(sol_k[:self.n_q])
                u_sol.append(sol_k[self.n_q:self.n_q+self.n_u])
            sol_N = output[stage_key(self.N+1)]
            q_sol.append(sol_N[:self.n_q]) 
            l_sol = sol_N[self.n_q+self.n_y:self.n_q+self.n_y+self.n_ss_pts]
            q_sol = np.array(q_sol)
            u_sol = np.array(u_sol)
        else:
            success = False
            # status = f'Solving Failed, exitflag = {exitflag}'
            q_sol = q_ws
            u_sol = u_ws
            l_sol = l_ws

        # self.print_method('=======================================')
        # self.print_method(f'input ub: {u_ub[0]}, lb: {u_lb[0]}')
        # self.print_method(f'\n{q_ub[0][3]}\n{qt[3]}\n{q_sol[:,3]}\n{u_sol[:,0]}')

        return [q_sol, u_sol, l_sol], success, exitflag
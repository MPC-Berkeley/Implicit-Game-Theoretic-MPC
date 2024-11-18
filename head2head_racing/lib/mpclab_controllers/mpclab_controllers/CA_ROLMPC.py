import time, copy
import casadi as ca
import numpy as np
import matplotlib.pyplot as plt
import pdb
import scipy.interpolate
import scipy.stats as ss
import os
from collections import deque
import pickle as pkl
import pathlib

import copy

from typing import Tuple, List, Dict


from mpclab_common.models.dynamics_models import CasadiDynamicsModel
from mpclab_common.pytypes import VehicleState, VehicleActuation, VehiclePrediction

from mpclab_controllers.abstract_controller import AbstractController
from mpclab_controllers.utils.controllerTypes import CALMPCParams
# from mpclab_controllers.utils.LTV_LMPC_utils import PredictiveModel, LMPC, MPCParams



class ROLMPC(AbstractController):

    def __init__(self, dynamics: CasadiDynamicsModel,
                       costs: Dict[str, List[ca.Function]],
                       constraints: Dict[str, ca.Function],
                       bounds: Dict[str, VehicleState],
                       control_params=CALMPCParams(),
                       qp_interface='casadi',
                       ws=True,
                       use_Einv=True,
                       print_method=print):
        
        self.dynamics       = dynamics
        self.track          = dynamics.track

        self.track_L        = self.track.track_length
        self.costs          = costs
        self.constraints    = constraints
        self.qp_interface   = qp_interface
        self.print_method   = print_method

        self.n_u            = self.dynamics.n_u
        self.n_q            = self.dynamics.n_q
        self.n_z            = self.n_q + self.n_u

        self.R              = self.dynamics.R

        self.verbose        = control_params.verbose

        self.dt             = control_params.dt
        self.N              = control_params.N

        self.keep_init_safe_set = control_params.keep_init_safe_set

        self.delay          = control_params.delay
        self.delay_buffer   = []
        if self.delay is None:
            self.delay = np.zeros(self.dynamics.n_u)
            self.delay_buffer = None

        self.use_ws=ws
        self.use_E=use_Einv

        self.n_ss_pts       = control_params.n_ss_pts
        self.n_ss_its       = control_params.n_ss_its

       

        

        self.curv=self.dynamics.get_curvature

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

        self.P_perm=np.array([[0,1,0],[0,0,1],[1,0,0]])

        self._build_invs_constraints()

        self.u_rate_ub=ca.DM([bounds['du_ub'].u.u_a, bounds['du_ub'].u.u_steer ])
        self.u_rate_lb=ca.DM([bounds['du_lb'].u.u_a, bounds['du_lb'].u.u_steer ])


        #####
        
        #####

        

        
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
        c2g_output_ss=self.costs["state"][0](lftd_outputs_ss[:,0])
        c2g_output_ss=np.cumsum(c2g_output_ss[::-1])[::-1]

        if iter_idx is not None:
            self.SS_data[iter_idx] = dict(output=lftd_outputs_ss, cost_to_go=c2g_output_ss)
        else:
            self.SS_data.append(dict(output=lftd_outputs_ss, cost_to_go=c2g_output_ss))

    def _add_point_to_safe_set(self, vehicle_state: VehicleState):
        q, u = self.dynamics.state2qu(vehicle_state)
    
        lftd_Y=np.hstack((self.SS_data[-1]['output'][-1,self.n_u:], self.dynamics.q2y(q) ))
        self.SS_data[-1]['output'] = np.vstack((self.SS_data[-1]['output'], lftd_Y.reshape((1,-1))))
        self.SS_data[-1]['cost_to_go']+=self.costs["state"][0](lftd_Y[0])
        self.SS_data[-1]['cost_to_go'] = np.append(self.SS_data[-1]['cost_to_go'], self.costs["state"][0](lftd_Y[0]))


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
        um1 = self.u_prev
        qb_prev=self.qb_prev
        if self.delay_buffer is not None:
            delay_steps = int(np.amin(self.delay))
            u_delay = np.hstack([np.array(self.delay_buffer[i])[:delay_steps].reshape((-1,1)) for i in range(self.dynamics.n_u)])
            q_bar = self._evaluate_dynamics(q0, u_delay)
            q0 = q_bar[-1]
            um1 = u_delay[-1]
            qb_prev=q_bar[-2]

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
    
        SS_Y = np.vstack(SS_Y)
        SS_Q = np.concatenate(SS_Q)

        self.SS_Y_sel = SS_Y
        self.SS_Q_sel = SS_Q

        # seg=np.argmin(np.abs(self.segs-self.curv(q0[0])))

        E_sched = self._error_inv_schedule(q_ws)

        qu_sol, success, status = self._solve_casadi(q0, u0, qb_prev, E_sched, q_ws, self.u_ws)
        if not success:
            self.print_method('Warning: NLP returned ' + str(status))

        if success:
            # Unpack solution
            if self.verbose:
                self.print_method('Current state q: ' + str(q0))
                self.print_method(status)
        
            q_sol = qu_sol[0]
            u_sol = qu_sol[1]

            e= ca.vec(q_sol[0,:])-ca.vec(q0)
            
            vel_idx=int(E_sched[0]/len(self.F_invs[0]))
            arc_idx=int(E_sched[0])-vel_idx*len(self.F_invs[0])
            # pdb.set_trace()
            # print(vel_idx, arc_idx)
            # u_sol[1,1]=u_sol[1,1]-(self.F_invs[vel_idx][arc_idx]@self.P_perm.T@e[:-1])[1]

            # if self.warm_start_with_nonlinear_rollout:
            #     q_sol = self._evaluate_dynamics(q0, u_sol)
            # else:
                # q_sol = qu_sol[:,:self.n_q]
        
            lmbd_sol=qu_sol[-1]
        else:
            u_sol = self.u_ws
            q_sol = self._evaluate_dynamics(q0, u_sol[1:,:])
            lmbd_sol=None
        

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
        self.opti=ca.Opti()
        p_opts = {'expand':False, 'print_time':0, 'verbose' :False}
        s_opts = {'sb': 'yes','print_level': 0, 'max_cpu_time':0.09, 'honor_original_bounds': 'yes'}
        self.opti.solver("ipopt", p_opts, s_opts)

        q_bar = self.opti.variable(self.N+1,self.n_q)

        u_bar = self.opti.variable(self.N+1,self.n_u)
        Y_bar = self.opti.variable(1, self.R*self.n_u)

        slack=self.opti.variable(1)
        self.opti.subject_to(slack>=0)
        
        lmbd = self.opti.variable(1,int(self.n_ss_pts/self.n_ss_its)*self.n_ss_its)
        SS_Y = self.opti.parameter(int(self.n_ss_pts/self.n_ss_its)*self.n_ss_its, self.R*self.n_u)
        SS_Q = self.opti.parameter(int(self.n_ss_pts/self.n_ss_its)*self.n_ss_its,1)
        
        qt   = self.opti.parameter(1, self.n_q)
        ut   = self.opti.parameter(1, self.n_u)
        qb_prev   = self.opti.parameter(1, self.n_q)

        E_sched = self.opti.parameter(self.N+1,1)

        z_inv = self.opti.variable(1,self.n_q-1)
        
        ## Error invariant constraint of the form 1-(e-ci).T@Q(e-ci)>=0 for error=q0-q_0
        err=qt-q_bar[0,:]
        
        e_inv_constraint = self.f_E_cond(E_sched[0],0)@z_inv.T-err.T

        self.opti.subject_to(e_inv_constraint==0)
        self.opti.subject_to(z_inv@z_inv.T<=1+.0*slack)
        # self.opti.subject_to((err)@ca.diag([.1, 3., 10., 1])@(err).T<=.1+slack)
        # self.opti.subject_to(q_bar[0,:]==self.qt_param)
        # pdb.set_trace()
        
        # Pick the nominal state q_bar[0,:] at time t, such that it reachable from the solution q_bar[0,:] at time t-1,
        # by picking an input within constraints
        res=q_bar[0,:].T-self.dynamics.fd(qb_prev,u_bar[0,:])
      
        self.opti.subject_to(res.T@res<=slack)
        
        # Setting acceleration constraint in terms of s
        self.opti.subject_to(self.opti.bounded(self.u_rate_lb[0]-.1*slack,ca.diff(ca.vertcat(ut[0],u_bar[1:,0]))/self.dt,.1*slack+self.u_rate_ub[0]))
        self.opti.subject_to(self.opti.bounded(self.u_rate_lb[1]-.1*slack,ca.diff(ca.vertcat(ut[1],u_bar[1:,1]))/self.dt,.1*slack+self.u_rate_ub[1]))
        # self.opti.subject_to(self.opti.bounded(self.a_min*self.dt**2, ca.diff(q_bar[:,0]-ca.vertcat(qb_prev[0],q_bar[:-1,0])),self.a_max*self.dt**2))

        # Terminal constraint
        self.opti.subject_to(self.opti.bounded(0,lmbd,1))
        self.opti.subject_to(ca.sum1(lmbd.T)==1.0)
        # self.opti.subject_to(lmbd@SS_Y==Y_bar)

        # self.opti.subject_to(self.opti.bounded(ca.DM(*Y_bar.shape)-0.001*slack,lmbd@SS_Y-Y_bar, ca.DM(*Y_bar.shape)+0.001*slack))
        # This constraint enforces F_x(Y_var)=q_bar[N,:]
        self.opti.subject_to((lmbd@SS_Y-Y_bar)@(lmbd@SS_Y-Y_bar).T<=0.001)
        flat_map=self.dynamics.fun_Fx_eq(ca.vec(q_bar[-1,:]),Y_bar)
        # self.opti.subject_to(flat_map.T@flat_map<=0.1*slack)
        self.opti.subject_to(flat_map==0)
        # self.opti.subject_to(self.opti.bounded(ca.DM(self.n_q-1,1)-0.01*slack,flat_map,ca.DM(self.n_q-1,1)+0.01*slack))

        cost=lmbd@SS_Q+10**4*slack#+10**2*err.T@err
        for k in range(self.N):
            self.opti.subject_to(q_bar[k+1,:].T==self.dynamics.fd(q_bar[k,:],u_bar[k+1,:]))
            # self.opti.subject_to(self.opti.bounded(self.state_lb, q_bar[k,:], self.state_ub))
            # self.opti.subject_to(self.opti.bounded(self.input_lb, u_bar[k,:], self.input_ub))
            self.opti.subject_to(self.opti.bounded(self.fq_lb_cond(E_sched[k],0)-0.0*slack, q_bar[k,:].T, 0.0*slack+self.fq_ub_cond(E_sched[k],0)))
            self.opti.subject_to(self.opti.bounded(self.fu_lb_cond(E_sched[k],0)-0.0*slack, u_bar[k+1,:].T, 0.0*slack+self.fu_ub_cond(E_sched[k],0)))
            cost+=self.costs["state"][0](q_bar[k,0])

            cost += ca.bilin(1*ca.DM.eye(2), u_bar[k+1,:], u_bar[k+1,:])
            cost += ca.bilin(1*ca.DM.eye(2), u_bar[k+1,:]-u_bar[k,:], u_bar[k+1,:]-u_bar[k,:])/(self.dt**2)

        self.opti.minimize(cost)
        if self.use_ws:
            self.nlp=self.opti.to_function("ROLMPC", 
                [qt, ut, qb_prev, E_sched, SS_Y, SS_Q, q_bar, u_bar, Y_bar, lmbd],
                [q_bar, u_bar, lmbd])
        else:
            self.nlp=self.opti.to_function("ROLMPC", 
                [qt, ut, qb_prev, E_sched, SS_Y, SS_Q],
                [q_bar, u_bar])
            

        # saving variables and parameters as member functions for debugging using opti.debug    
        self._q=q_bar
        self._u=u_bar
        self._Y=Y_bar
        self._lmbd=lmbd
        self._SSY=SS_Y
        self._SSQ=SS_Q
        self._Esch=E_sched
        self._qt=qt
        self._ut=ut
        self._qbp=qb_prev

    def _solve_casadi(self, qt, ut, qb_prev, E_sched, q_ws, u_ws):

        status="Warm-start"
        lmbd_sol=None
        
        try:
            if self.use_ws:
                q_sol, u_sol, lmbd_sol =self.nlp(qt, ut, qb_prev, E_sched, self.SS_Y_sel, self.SS_Q_sel, q_ws, u_ws, self.Y_ws, self.lmbd_ws)
                # print(q_sol)
                
                # self._update_opti_params(qt, ut, qb_prev, E_sched, q_ws, u_ws)
                # sol=self.opti.solve()
                # dbug=self.opti.debug
                # pdb.set_trace()
            else:
                q_sol, u_sol  =self.nlp(qt, ut, qb_prev, E_sched, self.SS_Y_sel, self.SS_Q_sel)
            status="Solved"
            success=True
        except:
            q_sol=q_ws
            u_sol=u_ws
            success=False

        # pdb.set_trace()
        

        return [q_sol,u_sol, lmbd_sol], success, status
    

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
        P_def=ca.inv(ca.diag([0.5, 3., 10.]))
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
                    #     norm_k=np.sqrt(self.n_q-1)*np.linalg.norm(Q_del, np.inf)
                    #     self.tight_u_ub[i][k]-=norm_k
                    #     self.tight_u_lb[i][k]+=norm_k
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
    
 
    
    def _update_opti_params(self, qt, ut, qb_prev, E_sched, q_ws, u_ws):
        self.opti.set_value(self._qt, qt)
        self.opti.set_value(self._ut, ut)
        self.opti.set_value(self._qbp, qb_prev)
        self.opti.set_value(self._Esch, E_sched)
        self.opti.set_value(self._SSY, self.SS_Y_sel)
        self.opti.set_value(self._SSQ, self.SS_Q_sel)
        self.opti.set_initial(self._lmbd, self.lmbd_ws)
        self.opti.set_initial(self._q, q_ws)
        self.opti.set_initial(self._u, u_ws)
        self.opti.set_initial(self._Y, self.Y_ws)

    





            






        


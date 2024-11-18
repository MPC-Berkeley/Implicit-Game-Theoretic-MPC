#!/usr/bin python3

import warnings

import numpy as np
import scipy as sp
import casadi as ca

import copy
import time

from typing import Dict

from mpclab_common.models.dynamics_models import CasadiDynamicsModel
from mpclab_common.pytypes import VehicleState, VehiclePrediction

from mpclab_controllers.abstract_controller import AbstractController
from mpclab_controllers.utils.controllerTypes import CAMPCCParams

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import pdb

# LTV MPC formulated using CasADi
class CA_MPCC(AbstractController):
    def __init__(self, dynamics: CasadiDynamicsModel, 
                       costs: Dict[str, ca.Function], 
                       constraints: Dict[str, ca.Function],
                       bounds: Dict[str, VehicleState],
                       control_params: CAMPCCParams=CAMPCCParams(),
                       print_method=print):
        self.dynamics       = dynamics
        self.dt             = dynamics.dt
        self.track          = self.dynamics.track
        self.costs          = costs
        self.constraints    = constraints

        self.verbose        = control_params.verbose

        if print_method is None:
            self.print_method = lambda s: None
        else:
            self.print_method = print_method

        self.L              = self.track.track_length
        self.n_u            = self.dynamics.n_u + 1
        self.n_q            = self.dynamics.n_q + 1

        self.N              = control_params.N

        self.pos_idx        = control_params.pos_idx

        # List of indexes for the constraints at each time step
        self.soft_constraint_idxs   = control_params.soft_constraint_idxs
        if self.soft_constraint_idxs is not None:
            self.soft_constraint_quad      = [np.array(q) for q in control_params.soft_constraint_quad]
            self.soft_constraint_lin       = [np.array(l) for l in control_params.soft_constraint_lin]

        self.parametric_contouring_cost   = control_params.parametric_contouring_cost
        if self.parametric_contouring_cost:
            self.q_c    = ca.MX.sym('q_c', 1)
            self.q_cN   = ca.MX.sym('q_cN', 1)
        else:
            self.q_c    = control_params.contouring_cost
            self.q_cN   = control_params.contouring_cost_N

        self.q_l            = control_params.lag_cost
        self.q_lN           = control_params.lag_cost_N
        self.q_p            = control_params.performance_cost
        self.q_v            = control_params.vs_cost
        self.q_dv           = control_params.vs_rate_cost

        self.track_tightening   = control_params.track_tightening
        self.soft_track     = control_params.soft_track
        self.q_tq           = control_params.track_slack_quad
        self.q_tl           = control_params.track_slack_lin

        self.vs_max         = control_params.vs_max
        self.vs_min         = control_params.vs_min
        self.vs_rate_max    = control_params.vs_rate_max
        self.vs_rate_min    = control_params.vs_rate_min

        self.delay          = control_params.delay
        self.delay_buffer   = []
        if self.delay is None:
            self.delay = np.zeros(self.dynamics.n_u)
            self.delay_buffer = None

        self.solver_name    = control_params.solver_name
        self.code_gen       = control_params.code_gen
        self.jit            = control_params.jit
        self.opt_flag       = control_params.opt_flag

        self.debug_plot     = control_params.debug_plot

        self.solver_name    = control_params.solver_name
        
        # Process box constraints
        self.state_ub, self.input_ub = self.dynamics.state2qu(bounds['qu_ub'])
        self.state_lb, self.input_lb = self.dynamics.state2qu(bounds['qu_lb'])
        _, self.input_rate_ub = self.dynamics.state2qu(bounds['du_ub'])
        _, self.input_rate_lb = self.dynamics.state2qu(bounds['du_lb'])

        self.x_ub  = np.append(self.state_ub, 2*self.L)
        self.x_lb  = np.append(self.state_lb, -1)
        self.w_ub  = np.append(self.input_ub, self.vs_max)
        self.w_lb  = np.append(self.input_lb, self.vs_min)
        self.dw_ub = np.append(self.input_rate_ub, self.vs_rate_max)
        self.dw_lb = np.append(self.input_rate_lb, self.vs_rate_min)

        _ub = [np.tile(np.concatenate((self.x_ub, self.w_ub)), self.N+1), np.tile(self.dw_ub, self.N)]
        _lb = [np.tile(np.concatenate((self.x_lb, self.w_lb)), self.N+1), np.tile(self.dw_lb, self.N)]
        if self.soft_track:
            _ub.append(np.inf*np.ones(2))
            _lb.append(np.zeros(2))
        self.ub = np.concatenate(_ub)
        self.lb = np.concatenate(_lb)

        self.q_pred = np.zeros((self.N+1, self.dynamics.n_q))
        self.s_pred = np.zeros(self.N+1)

        self.u_pred = np.zeros((self.N, self.dynamics.n_u))
        self.vs_pred = np.zeros(self.N)

        self.du_pred = np.zeros((self.N, self.dynamics.n_u))
        self.dvs_pred = np.zeros(self.N)

        self.u_prev = np.zeros(self.dynamics.n_u)
        self.vs_prev = 0

        self.dw_pred = np.zeros((self.N, self.n_u))

        self.u_ws = None
        self.vs_ws = None

        self.du_ws = None
        self.dvs_ws = None

        self.l_ws = None

        self.state_input_prediction = VehiclePrediction()

        self.initialized = False

        self._build_solver()
        
    def initialize(self):
        pass

    def step(self, vehicle_state: VehicleState,
                reference: np.ndarray = np.array([]),
                parameters: np.ndarray = np.array([])):
        self.solve(vehicle_state, reference, parameters)

        u = self.u_pred[0]
        self.dynamics.qu2state(vehicle_state, None, u)
        self.dynamics.qu2prediction(self.state_input_prediction, self.q_pred, self.u_pred)
        self.state_input_prediction.t = vehicle_state.t

        # Update delay buffer
        if self.delay_buffer is not None:
            for i in range(self.dynamics.n_u):
                self.delay_buffer[i].append(u[i])

        # Construct initial guess for next iteration
        u_ws = np.vstack((self.u_pred, self.u_pred[-1]))
        vs_ws = np.append(self.vs_pred, self.vs_pred[-1])
        du_ws = np.vstack((self.du_pred[1:], self.du_pred[-1]))
        dvs_ws = np.append(self.dvs_pred[1:], self.dvs_pred[-1])
        self.set_warm_start(u_ws, vs_ws, du_ws, dvs_ws)

        return

    def set_warm_start(self, u_ws: np.ndarray, vs_ws: np.ndarray, du_ws: np.ndarray, dvs_ws: np.ndarray, 
                       l_ws: np.ndarray = None):
        self.u_ws = u_ws
        self.vs_ws = vs_ws

        self.u_prev = u_ws[0]
        self.vs_prev = vs_ws[0]

        self.du_ws = du_ws
        self.dvs_ws = dvs_ws
        
        if l_ws is not None:
            self.l_ws = l_ws

        self.initalized = True

    def _evaluate_dynamics(self, q0, s0, U, VS):
        t = time.time()
        Q, S = [q0], [s0]
        for k in range(U.shape[0]):
            Q.append(self.dynamics.fd(Q[k], U[k]).toarray().squeeze())
            S.append(float(self.fs_d(S[k], VS[k])))
        if self.verbose:
            self.print_method('Dynamics evalution time: ' + str(time.time()-t))
        return np.array(Q), np.array(S)
    
    def solve(self, state: VehicleState,
              reference: np.ndarray = np.array([]),
              parameters: np.ndarray = np.array([])):
        
        state.e.psi = np.mod(state.e.psi, 2*np.pi)

        # Default values for reference interpolation
        if len(reference) == 0:
            reference = np.zeros(self.N+1)
        else:
            if len(np.where(reference > 1)[0]) > 0 or len(np.where(reference < -1)[0]) > 0:
                self.print_method('Reference out of bounds, clamping to [-1, 1]')
            reference[reference > 1] = 1
            reference[reference < -1] = -1

        q0, _ = self.dynamics.state2qu(state)
        s0, _, _ = self.track.global_to_local((state.x.x, state.x.y, 0))

        if self.delay_buffer is not None:
            delay_steps = int(np.amin(self.delay))
            u_delay = np.hstack([np.array(self.delay_buffer[i])[:delay_steps].reshape((-1,1)) for i in range(self.dynamics.n_u)])
            q_bar, s_bar = self._evaluate_dynamics(q0, s0, u_delay, self.vs_ws[1:1+delay_steps])
            q0, s0 = q_bar[-1], s_bar[-1]
        q_ws, s_ws = self._evaluate_dynamics(q0, s0, self.u_ws[1:], self.vs_ws[1:])

        # Initial guess vector
        _D = [np.hstack((q_ws, s_ws.reshape((-1,1)), self.u_ws, self.vs_ws.reshape((-1,1)))).ravel(), np.hstack((self.du_ws, self.dvs_ws.reshape((-1,1)))).ravel()]
        if self.soft_track:
            _D.append(np.zeros(2))
        D = np.concatenate(_D)

        # Parameter vector
        P = np.concatenate((q0, [s0], self.u_prev, [self.vs_prev], s_ws, reference, parameters))
        
        lbg = np.concatenate((np.zeros(self.n_f), -np.inf*np.ones(self.n_c)))
        ubg = np.concatenate((np.zeros(self.n_f), np.zeros(self.n_c)))

        sol = self.solver(x0=D, lbx=self.lb, ubx=self.ub, lbg=lbg, ubg=ubg, p=P)

        if self.solver.stats()['success']:
            # Unpack solution
            xw_sol = np.array(sol['x'][:(self.n_q+self.n_u)*(self.N+1)]).squeeze().reshape((self.N+1, self.n_q+self.n_u))
            dw_sol = np.array(sol['x'][(self.n_q+self.n_u)*(self.N+1):(self.n_q+self.n_u)*(self.N+1)+self.n_u*self.N]).squeeze().reshape((self.N, self.n_u))

            u_sol, vs_sol = xw_sol[1:,self.n_q:self.n_q+self.dynamics.n_u], xw_sol[1:,-1]
            du_sol, dvs_sol = dw_sol[:,:self.dynamics.n_u], dw_sol[:,self.dynamics.n_u:]
        else:
            u_sol, vs_sol = self.u_ws[1:], self.vs_ws[1:]
            du_sol, dvs_sol = self.du_ws, self.dvs_ws

        q_sol, s_sol = self._evaluate_dynamics(q0, s0, u_sol, vs_sol)
        if self.debug_plot:
            self._update_debug_plot(q_sol, s_sol, u_sol, vs_sol)

        self.q_pred = q_sol
        self.s_pred = s_sol
        self.u_pred = u_sol
        self.vs_pred = vs_sol
        self.du_pred = du_sol
        self.dvs_pred = dvs_sol

    def get_prediction(self):
        return self.state_input_prediction

    def _build_solver(self):
        # Compute spline approximation of track
        S = np.linspace(0, self.track.track_length, 100)
        X, Y, Xi, Yi, Xo, Yo = [], [], [], [], [], []
        for s in S:
            # Centerline
            x, y, _ = self.track.local_to_global((s, 0, 0))
            X.append(x)
            Y.append(y)
            # Inside boundary
            xi, yi, _ = self.track.local_to_global((s, self.track.half_width-self.track_tightening, 0))
            Xi.append(xi)
            Yi.append(yi)
            # Outside boundary
            xo, yo, _ = self.track.local_to_global((s, -(self.track.half_width-self.track_tightening), 0))
            Xo.append(xo)
            Yo.append(yo)
        self.x_s = ca.interpolant('x_s', 'bspline', [S], X)
        self.y_s = ca.interpolant('y_s', 'bspline', [S], Y)
        self.xi_s = ca.interpolant('xi_s', 'bspline', [S], Xi)
        self.yi_s = ca.interpolant('yi_s', 'bspline', [S], Yi)
        self.xo_s = ca.interpolant('xo_s', 'bspline', [S], Xo)
        self.yo_s = ca.interpolant('yo_s', 'bspline', [S], Yo)

        # Compute derivatives of track
        s_sym = ca.MX.sym('s', 1)
        self.dsdx_s = ca.Function('dsdx_s', [s_sym], [ca.jacobian(self.x_s(s_sym), s_sym)])
        self.dsdy_s = ca.Function('dsdy_s', [s_sym], [ca.jacobian(self.y_s(s_sym), s_sym)])
        
        # Dynamcis augmented with arc length dynamics
        q_sym = ca.MX.sym('q', self.dynamics.n_q)
        u_sym = ca.MX.sym('u', self.dynamics.n_u)
        vs_sym = ca.MX.sym('vs', 1)

        x_sym = ca.vertcat(q_sym, s_sym)
        w_sym = ca.vertcat(u_sym, vs_sym)

        n_z = self.n_q + self.n_u

        # Approximate arc length dynamics
        self.fs_d = ca.Function('fs_d', [s_sym, vs_sym], [s_sym + self.dt*vs_sym])

        # Contouring and lag errors
        s_mod = ca.fmod(s_sym, self.L)
        # Reference interpolation variable must be in range [-1, 1] (outside, inside)
        z_sym = ca.MX.sym('z', 1)
        t = ca.atan2(self.dsdy_s(s_mod), self.dsdx_s(s_mod))
        x_int = self.xo_s(s_mod) + (z_sym+1)/2*(self.xi_s(s_mod)-self.xo_s(s_mod))
        y_int = self.yo_s(s_mod) + (z_sym+1)/2*(self.yi_s(s_mod)-self.yo_s(s_mod))
        ec =  ca.sin(t)*(x_sym[self.pos_idx[0]]-x_int) - ca.cos(t)*(x_sym[self.pos_idx[1]]-y_int)
        el = -ca.cos(t)*(x_sym[self.pos_idx[0]]-x_int) - ca.sin(t)*(x_sym[self.pos_idx[1]]-y_int)
        f_e = ca.Function('ec', [x_sym, z_sym], [ca.vertcat(ec, el)])

        # q_0, ..., q_N
        q_ph = [ca.MX.sym(f'q_ph_{k}', self.dynamics.n_q) for k in range(self.N+1)] # State
        s_ph = [ca.MX.sym(f's_{k}', 1) for k in range(self.N+1)]
        x_ph = [ca.vertcat(q_ph[k], s_ph[k]) for k in range(self.N+1)]

        # u_-1, ..., u_N-1
        u_ph = [ca.MX.sym(f'u_ph_{k}', self.dynamics.n_u) for k in range(self.N+1)] # Inputs
        vs_ph = [ca.MX.sym(f'vs_{k}', 1) for k in range(self.N+1)]
        w_ph = [ca.vertcat(u_ph[k], vs_ph[k]) for k in range(self.N+1)]

        # du_0, ..., du_N-1
        du_ph = [ca.MX.sym(f'du_ph_{k}', self.dynamics.n_u) for k in range(self.N)] # Input rates
        dvs_ph = [ca.MX.sym(f'dvs_{k}', 1) for k in range(self.N)]
        dw_ph = [ca.vertcat(du_ph[k], dvs_ph[k]) for k in range(self.N)]

        if self.soft_track:
            s_track = ca.MX.sym('s_track', 2)

        # Parameters
        sp_ph = [ca.MX.sym(f'sp_{k}', 1) for k in range(self.N+1)] # Approximate progress from previous time step
        xw0_ph = ca.MX.sym('xw0', n_z) # Initial state
        z_ph = ca.MX.sym('z_ph', self.N+1) # Reference interpolation 

        state_cost_params = []
        input_cost_params = []
        rate_cost_params = []
        constraint_params = []

        # Cost over the horizon
        J = ca.DM.zeros(1)
        for k in range(self.N+1):
            if k < self.N:
                P_cl = ca.diag(ca.vertcat(self.q_c, self.q_l))
            else:
                P_cl = ca.diag(ca.vertcat(self.q_cN, self.q_lN))
            e = f_e(x_ph[k], z_ph[k])
            Je_k = 0.5*ca.bilin(P_cl, e, e)

            # State costs
            if self.costs['state'][k]:
                if self.costs['state'][k].n_in() == 2:
                    pq_k = ca.MX.sym(f'pq_{k}', self.costs['state'][k].numel_in(1))
                    Jx_k = self.costs['state'][k](q_ph[k], pq_k)
                    state_cost_params.append(pq_k)
                else:
                    Jx_k = self.costs['state'][k](q_ph[k])
            else:
                Jx_k = ca.DM.zeros(1)

            # Input costs
            if self.costs['input'][k]:
                if self.costs['input'][k].n_in() == 2:
                    pu_k = ca.MX.sym(f'pu_{k}', self.costs['input'][k].numel_in(1))
                    Jw_k = self.costs['input'][k](u_ph[k])
                    input_cost_params.append(pu_k)
                else:    
                    Jw_k = self.costs['input'][k](u_ph[k])
            else:
                Jw_k = ca.DM.zeros(1)
            # Progress rate cost
            Jw_k += 0.5*self.q_v*vs_ph[k]**2 - self.q_p*vs_ph[k]

            J += Je_k + Jx_k + Jw_k

            if k < self.N:
                # Input rate costs
                if self.costs['rate'][k]:
                    if self.costs['rate'][k].n_in() == 2:
                        pdu_k = ca.MX.sym(f'pdu_{k}', self.costs['rate'][k].numel_in(1))
                        Jdw_k = self.costs['rate'][k](du_ph[k], pdu_k)
                        rate_cost_params.append(pdu_k)
                    else:
                        Jdw_k = self.costs['rate'][k](du_ph[k])
                else:
                    Jdw_k = ca.DM.zeros(1)
                # Progress accel cost
                Jdw_k += 0.5*self.q_dv*dvs_ph[k]**2

                J += Jdw_k

            # Slack costs
            Js_k = ca.DM.zeros(1)
            if self.soft_track:
                Js_k += 0.5*ca.bilin(self.q_tq*ca.DM.eye(2), s_track, s_track) + self.q_tl*ca.sum1(s_track)

            J += Js_k
            
        # Kinodynamic constraints
        F = [ca.vertcat(x_ph[0], w_ph[0]) - xw0_ph]
        for k in range(self.N):
            F.append(x_ph[k+1] - ca.vertcat(self.dynamics.fd(q_ph[k], u_ph[k+1]), self.fs_d(s_ph[k], vs_ph[k+1])))
            F.append(w_ph[k+1] - (w_ph[k] + self.dt*dw_ph[k]))
        F = ca.vertcat(*F)

        # Inequality constraints
        C = []
        for k in range(self.N+1):
            # State input constraints
            if self.constraints['state_input'][k]:
                if self.constraints['state_input'][k].n_in() == 3:
                    pqu_k = ca.MX.sym(f'pqu_{k}', self.constraints['state_input'][k].numel_in(2))
                    Cqu = self.constraints['state_input'][k](q_ph[k], u_ph[k], pqu_k)
                    constraint_params.append(pqu_k)
                else:
                    Cqu = self.constraints['state_input'][k](q_ph[k], u_ph[k])
                C.append(Cqu)

            # Input rate constraints
            if k < self.N:
                if self.constraints['rate'][k]:
                    Cdu = self.constraints['rate'][k](du_ph[k])
                    C.append(Cdu)

            # Track boundary constraints
            if k >= 1:
                xi, yi = self.xi_s(ca.fmod(sp_ph[k], self.L)), self.yi_s(ca.fmod(sp_ph[k], self.L))
                xo, yo = self.xo_s(ca.fmod(sp_ph[k], self.L)), self.yo_s(ca.fmod(sp_ph[k], self.L))
                n, d = -(xo - xi), yo - yi
                if self.soft_track:
                    C.append((n*q_ph[k][self.pos_idx[0]] - d*q_ph[k][self.pos_idx[1]]) - ca.fmax(n*xi-d*yi, n*xo-d*yo) - s_track[0])
                    C.append(ca.fmin(n*xi-d*yi, n*xo-d*yo) - (n*q_ph[k][self.pos_idx[0]] - d*q_ph[k][self.pos_idx[1]]) - s_track[1])
                else:
                    C.append((n*q_ph[k][self.pos_idx[0]] - d*q_ph[k][self.pos_idx[1]]) - ca.fmax(n*xi-d*yi, n*xo-d*yo))
                    C.append(ca.fmin(n*xi-d*yi, n*xo-d*yo) - (n*q_ph[k][self.pos_idx[0]] - d*q_ph[k][self.pos_idx[1]]))
        C = ca.vertcat(*C)

        self.n_c = C.size1()
        self.n_f = F.size1()
        
        # Form decision vector using augmented states (x_k, w_k) and inputs dw_k
        # D = [(x_0, w_-1), ..., (x_N, w_N-1), dw_0, ..., dw_N-1]
        D = []
        for x, w in zip(x_ph, w_ph):
            D.extend([x, w])
        D += dw_ph
        if self.soft_track:
            D.append(s_track)
        D = ca.vertcat(*D)
        
        # Parameters
        P = [ca.vertcat(xw0_ph, *sp_ph, z_ph)] + state_cost_params + input_cost_params + rate_cost_params + constraint_params
        if self.parametric_contouring_cost:
            P += [ca.vertcat(self.q_c, self.q_cN)]
        P = ca.vertcat(*P)

        prob = dict(x=D, f=J, g=ca.vertcat(F, C), p=P)
        ipopt_opts = dict(max_iter=200, 
                        linear_solver='ma57')
        solver_opts = dict(error_on_fail=False, ipopt=ipopt_opts)
        if self.code_gen:
            solver = ca.nlpsol('nlp', 'ipopt', prob, solver_opts)
            solver.generate_dependencies('nlp.c')

            import subprocess
            self.print_method('Compiling NLP solver')
            subprocess.Popen(['gcc', '-fPIC', '-shared', '-'+self.opt_flag, 'nlp.c', '-o', 'nlp.so']).wait()

            # Create a new NLP solver instance from the compiled code
            self.solver = ca.nlpsol('nlp', 'ipopt', 'nlp.so')
        else:
            self.solver = ca.nlpsol('nlp', 'ipopt', prob, solver_opts)

if __name__ == "__main__":
    pass
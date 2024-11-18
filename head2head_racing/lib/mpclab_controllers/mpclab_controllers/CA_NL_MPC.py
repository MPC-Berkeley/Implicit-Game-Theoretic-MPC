#!/usr/bin python3

import warnings

import numpy as np
import casadi as ca

import copy
import time
import pathlib

from typing import Tuple, List, Dict
from collections import deque

from mpclab_common.models.dynamics_models import CasadiDynamicsModel
from mpclab_common.pytypes import VehicleState, VehicleActuation, VehiclePrediction

from mpclab_controllers.abstract_controller import AbstractController
from mpclab_controllers.utils.controllerTypes import CANLMPCParams

import pdb

# Nonlinear MPC formulated using CasADi
class CA_NL_MPC(AbstractController):
    def __init__(self, dynamics: CasadiDynamicsModel, 
                       costs: Dict[str, ca.Function], 
                       constraints: Dict[str, ca.Function],
                       bounds: Dict[str, VehicleState],
                       control_params: CANLMPCParams=CANLMPCParams(),
                       print_method=print):

        self.dynamics = dynamics
        self.dt = dynamics.dt
        self.costs = costs
        self.constraints = constraints
        if print_method is None:
            self.print_method = lambda s: None
        else:
            self.print_method = print_method

        self.n_u = self.dynamics.n_u
        self.n_q = self.dynamics.n_q
        self.n_z = self.n_q + self.n_u

        self.N = control_params.N

        self.max_iters      = control_params.max_iters
        self.linear_solver  = control_params.linear_solver

        self.reg            = control_params.reg

        self.soft_state_bound_idxs      = control_params.soft_state_bound_idxs
        if self.soft_state_bound_idxs is not None:
            self.soft_state_bound_quad      = np.diag(control_params.soft_state_bound_quad)
            self.soft_state_bound_lin       = np.array(control_params.soft_state_bound_lin)

        # List of indexes for the constraints at each time step
        self.soft_constraint_idxs   = control_params.soft_constraint_idxs
        if self.soft_constraint_idxs is not None:
            self.soft_constraint_quad   = [np.diag(q) for q in control_params.soft_constraint_quad]
            self.soft_constraint_lin    = [np.array(l) for l in control_params.soft_constraint_lin]

        self.wrapped_state_idxs     = control_params.wrapped_state_idxs
        self.wrapped_state_periods  = control_params.wrapped_state_periods

        self.delay                  = control_params.delay
        self.delay_buffer           = []
        if self.delay is None:
            self.delay = np.zeros(self.dynamics.n_u)
            self.delay_buffer = None

        self.solver_name            = control_params.solver_name
        self.verbose                = control_params.verbose
        self.code_gen               = control_params.code_gen
        self.jit                    = control_params.jit
        self.opt_flag               = control_params.opt_flag
        
        # Process box constraints
        self.state_ub, self.input_ub = self.dynamics.state2qu(bounds['qu_ub'])
        self.state_lb, self.input_lb = self.dynamics.state2qu(bounds['qu_lb'])
        _, self.input_rate_ub = self.dynamics.state2qu(bounds['du_ub'])
        _, self.input_rate_lb = self.dynamics.state2qu(bounds['du_lb'])

        self.qu_ub = np.concatenate((self.state_ub, self.input_ub))
        self.qu_lb = np.concatenate((self.state_lb, self.input_lb))
        self.X_lb = np.concatenate((np.tile(self.qu_lb, self.N), np.tile(self.input_rate_lb, self.N)))
        self.X_ub = np.concatenate((np.tile(self.qu_ub, self.N), np.tile(self.input_rate_ub, self.N)))

        self.n_c = [0 for _ in range(self.N+1)]
        self.n_d = [0 for _ in range(self.N)]

        self.n_sc = [0 for _ in range(self.N+1)]
        if self.soft_constraint_idxs is not None:
            self.n_sc = [len(si) for si in self.soft_constraint_idxs]

        self.q_pred = np.zeros((self.N+1, self.n_q))
        self.u_pred = np.zeros((self.N, self.n_u))
        self.du_pred = np.zeros((self.N, self.n_u))
        self.u_prev = np.zeros(self.n_u)

        self.u_ws = np.zeros((self.N+1, self.n_u))
        self.du_ws = np.zeros((self.N, self.n_u))

        self.state_input_prediction = None
        self.initialized = True

        self._build_solver()

    def initialize(self):
        pass

    def step(self, vehicle_state: VehicleState,
                parameters: np.ndarray = np.array([])):
        self.solve(vehicle_state, parameters)

        u = self.u_pred[0]
        self.dynamics.qu2state(vehicle_state, None, u)
        if self.state_input_prediction is None:
            self.state_input_prediction = VehiclePrediction()
        self.dynamics.qu2prediction(self.state_input_prediction, self.q_pred, self.u_pred)
        self.state_input_prediction.t = vehicle_state.t

        # Update delay buffer
        if self.delay_buffer is not None:
            for i in range(self.dynamics.n_u):
                self.delay_buffer[i].append(u[i])

        # Construct initial guess for next iteration
        u_ws = np.vstack((self.u_pred, self.u_pred[-1]))
        du_ws = np.vstack((self.du_pred[1:], self.du_pred[-1]))
        self.set_warm_start(u_ws, du_ws)

        return

    def set_warm_start(self, u_ws: np.ndarray, du_ws: np.ndarray, 
                       state: VehicleState = None,
                       parameters: np.ndarray = np.array([])):
        self.u_ws = u_ws
        self.u_prev = u_ws[0]
        self.du_ws = du_ws

        if self.delay_buffer is not None:
            for i in range(self.dynamics.n_u):
                self.delay_buffer.append(deque(self.u_ws[1:1+self.delay[i],i], maxlen=self.delay[i]))

    def solve(self, state: VehicleState,
                parameters: np.ndarray = np.array([])):
        q0, _ = self.dynamics.state2qu(state)

        if self.delay_buffer is not None:
            delay_steps = int(np.amin(self.delay))
            u_delay = np.hstack([np.array(self.delay_buffer[i])[:delay_steps].reshape((-1,1)) for i in range(self.dynamics.n_u)])
            q_bar = self._evaluate_dynamics(q0, u_delay)
            q0 = q_bar[-1]
        q_ws = self._evaluate_dynamics(q0, self.u_ws[1:])
        if self.wrapped_state_idxs is not None:
            for i, p in zip(self.wrapped_state_idxs, self.wrapped_state_periods):
                q_ws[:,i] = np.unwrap(q_ws[:,i], period=p)

        # Construct initial guess for the decision variables and the runtime problem data
        X = np.concatenate((np.hstack((q_ws[1:], self.u_ws[1:])).ravel(), 
                            self.du_ws.ravel()))
        if self.soft_state_bound_idxs is not None:
            X = np.append(X, np.zeros(2*len(self.soft_state_bound_idxs)))
        if self.soft_constraint_idxs is not None:
            X = np.append(X, np.zeros(np.sum(self.n_sc)))

        P = np.concatenate((q0, self.u_prev, parameters))
        
        if self.reg > 0:
            P = np.concatenate((P, q_ws.ravel()))
        
        solver_args = {}
        solver_args['x0'] = X
        solver_args['lbx'] = self.X_lb
        solver_args['ubx'] = self.X_ub
        solver_args['lbg'] = np.concatenate((-1e10*np.ones(np.sum(self.n_c)), np.zeros(np.sum(self.n_d))))
        solver_args['ubg'] = np.concatenate((np.zeros(np.sum(self.n_c)), np.zeros(np.sum(self.n_d))))
        solver_args['p'] = P
        
        sol = self.solver(**solver_args)
        success = self.solver.stats()['success']

        if success:
            # Unpack solution
            X = sol['x'].toarray().squeeze()
            qu_sol = X[:self.n_z*(self.N)].reshape((self.N, self.n_z))
            du_sol = X[self.n_z*(self.N):self.n_z*(self.N)+self.n_u*self.N].reshape((self.N, self.n_u))
            u_sol = qu_sol[:,self.n_q:self.n_q+self.n_u]
            q_sol = np.vstack((q0.reshape((1,-1)), qu_sol[:,:self.n_q]))
        else:
            self.print_method(self.solver.stats()['return_status'])
            q_sol = q_ws
            u_sol = self.u_ws[1:]
            du_sol = self.du_ws
        
        self.q_pred = q_sol
        self.u_pred = u_sol
        self.du_pred = du_sol

        return success
    
    def get_prediction(self):
        return self.state_input_prediction

    def _evaluate_dynamics(self, q0, U):
        t = time.time()
        Q = [q0]
        for k in range(U.shape[0]):
            Q.append(self.dynamics.fd(Q[k], U[k]).toarray().squeeze())
        if self.verbose:
            self.print_method('Dynamics evalution time: ' + str(time.time()-t))
        return np.array(Q)

    def _build_solver(self):
        # q_0, ..., q_N
        q_ph = [ca.MX.sym(f'q_ph_{k}', self.n_q) for k in range(self.N+1)] # State
        # u_-1, u_0, ..., u_N-1
        u_ph = [ca.MX.sym(f'u_ph_{k}', self.n_u) for k in range(self.N+1)] # Inputs
        # du_0, ..., du_N-1
        du_ph = [ca.MX.sym(f'du_ph_{k}', self.n_u) for k in range(self.N)] # Input rates

        qu0_ph = ca.vertcat(q_ph[0], u_ph[0]) # Initial state

        if self.soft_state_bound_idxs is not None:
            state_ub_slack_ph = ca.MX.sym('state_ub_slack_ph', len(self.soft_state_bound_idxs))
            state_lb_slack_ph = ca.MX.sym('state_ub_slack_ph', len(self.soft_state_bound_idxs))

        constraint_slack_ph = []

        pJq_ph = []
        pJu_ph = []
        pJdu_ph = []
        pCqu_ph = []
        pqws_ph = []

        J = 0
        C = []
        D = []
        for k in range(self.N+1):
            # State costs
            if self.costs['state'][k]:
                if self.costs['state'][k].n_in() == 2:
                    pq_k = ca.MX.sym(f'pq_{k}', self.costs['state'][k].numel_in(1))
                    J += self.costs['state'][k](q_ph[k], pq_k)
                    pJq_ph.append(pq_k)
                else:
                    J += self.costs['state'][k](q_ph[k])
            
            # Input costs
            if self.costs['input'][k]:
                if self.costs['input'][k].n_in() == 2:
                    pu_k = ca.MX.sym(f'pu_{k}', self.costs['input'][k].numel_in(1))
                    J += self.costs['input'][k](u_ph[k], pu_k)
                    pJu_ph.append(pu_k)
                else:
                    J += self.costs['input'][k](u_ph[k])

            # Inequality constraints
            if self.constraints['state_input'][k]:
                if self.constraints['state_input'][k].n_in() == 3:
                    pqu_k = ca.MX.sym(f'pqu_{k}', self.constraints['state_input'][k].numel_in(2))
                    Cqu_k = self.constraints['state_input'][k](q_ph[k], u_ph[k], pqu_k)
                    pCqu_ph.append(pqu_k)
                else:
                    Cqu_k = self.constraints['state_input'][k](q_ph[k], u_ph[k])

                if self.n_sc[k] > 0:
                    constraint_slack_k = ca.MX.sym(f'constraint_slack_{k}', self.n_sc[k])
                    constraint_slack_ph.append(constraint_slack_k)

                    # Cost on slack variables
                    J += (1/2)*ca.bilin(self.soft_constraint_quad[k], constraint_slack_k, constraint_slack_k) \
                            + ca.dot(self.soft_constraint_lin[k], constraint_slack_k)
                    
                    # Setup vector of slack variables and add to constraint vector
                    e = [0 for _ in range(Cqu_k.size1())]
                    for i, j in enumerate(self.soft_constraint_idxs[k]):
                        e[j] = constraint_slack_k[i]
                    e = ca.vertcat(*e)
                    Cqu_k -= e

                C.append(Cqu_k)
                self.n_c[k] += Cqu_k.size1()

            # Regularization
            if self.reg > 0:
                pqws_ph.append(ca.MX.sym(f'qws_ph_{k}', self.n_q))
                J += (1/2)*self.reg*ca.sumsqr(q_ph[k]-pqws_ph[k])

            if k >= 1 and self.soft_state_bound_idxs is not None:
                Csq_k = ca.vertcat(q_ph[k][self.soft_state_bound_idxs] - self.state_ub[self.soft_state_bound_idxs] - state_ub_slack_ph,
                                   self.state_lb[self.soft_state_bound_idxs] - q_ph[k][self.soft_state_bound_idxs] + state_lb_slack_ph)
                C.append(Csq_k)
                self.n_c[k] += Csq_k.size1()
                self.state_ub[self.soft_state_bound_idxs] = 1e10
                self.state_lb[self.soft_state_bound_idxs] = -1e10

            if k < self.N:
                # Input rate costs
                if self.costs['rate'][k]:
                    if self.costs['rate'][k].n_in() == 2:
                        pdu_k = ca.MX.sym(f'pdu_{k}', self.costs['rate'][k].numel_in(1))
                        J += self.costs['rate'][k](du_ph[k], pdu_k)
                        pJdu_ph.append(pdu_k)
                    else:
                        J += self.costs['rate'][k](du_ph[k])

                if self.constraints['rate'][k]:
                    Cdu_k = self.constraints['rate'][k](du_ph[k]), du_ph[k]
                    C.append(Cdu_k)
                    self.n_c[k] += Cdu_k.size1()

                # Kinodynamic constraints
                D_k = ca.vertcat(q_ph[k+1] - self.dynamics.fd(q_ph[k], u_ph[k]),
                                 u_ph[k+1] - (u_ph[k] + self.dt*du_ph[k]))
                D.append(D_k)
                self.n_d[k] += D_k.size1()

        if self.soft_state_bound_idxs is not None:
            J += 0.5*ca.bilin(self.soft_state_bound_quad, state_ub_slack_ph, state_ub_slack_ph) \
                    + ca.dot(self.soft_state_bound_lin, state_ub_slack_ph)
            J += 0.5*ca.bilin(self.soft_state_bound_quad, state_lb_slack_ph, state_lb_slack_ph) \
                    + ca.dot(self.soft_state_bound_lin, state_lb_slack_ph)

        # Form decision vector using augmented states (q_k, u_k) and inputs du_k
        # X = [(q_1, u_0), ..., (q_N, u_N-1), du_0, ..., du_N-1]
        X = []
        for q, u in zip(q_ph[1:], u_ph[1:]):
            X.extend([q, u])
        X += du_ph
        if self.soft_state_bound_idxs is not None:
            X += [state_ub_slack_ph, state_lb_slack_ph]
            self.X_lb = np.append(self.X_lb, np.zeros(2*len(self.soft_state_bound_idxs)))
            self.X_ub = np.append(self.X_ub, 1e10*np.ones(2*len(self.soft_state_bound_idxs)))
        if self.soft_constraint_idxs is not None:
            X += constraint_slack_ph
            self.X_lb = np.append(self.X_lb, np.zeros(np.sum(self.n_sc)))
            self.X_ub = np.append(self.X_ub, 1e10*np.ones(np.sum(self.n_sc)))

        # Parameters
        P = [qu0_ph] + pJq_ph + pJu_ph + pJdu_ph + pCqu_ph + pqws_ph

        X = ca.vertcat(*X)
        P = ca.vertcat(*P)

        self.f_J = ca.Function('f_J', [X, P], [J])
        self.f_C = ca.Function('f_C', [X, P], [ca.vertcat(*C)])
        self.f_D = ca.Function('f_D', [X, P], [ca.vertcat(*D)])

        nlp = dict(x=X, p=P, f=J, g=ca.vertcat(*C, *D))

        if self.code_gen:
            code_gen_opts = dict(jit=True, 
                                jit_name=self.solver_name, 
                                compiler='shell',
                                jit_options=dict(compiler='gcc', flags=['-%s' % self.opt_flag], verbose=self.verbose))
        else:
            code_gen_opts = dict(jit=False)

        ipopt_opts = dict(max_iter=self.max_iters, 
                          linear_solver=self.linear_solver, 
                          warm_start_init_point='yes',
                          mu_strategy='adaptive',
                          mu_init=1e-5,
                          mu_min=1e-15,
                          barrier_tol_factor=1,
                          print_level=5 if self.verbose else 0)
        options = dict(error_on_fail=False, 
                            verbose_init=self.verbose, 
                            ipopt=ipopt_opts,
                            **code_gen_opts)
        self.solver = ca.nlpsol('solver', 'ipopt', nlp, options)

if __name__ == "__main__":
    pass
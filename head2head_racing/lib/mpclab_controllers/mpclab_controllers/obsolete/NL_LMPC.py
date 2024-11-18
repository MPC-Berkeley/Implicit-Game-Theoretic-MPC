#!/usr/bin python3

import pdb
import array
import warnings

import numpy as np
from numpy import linalg as la

import casadi as ca

import sys, os, pathlib

sys.path.append(os.path.join(os.path.expanduser('~'), 'forces_pro_client'))
import forcespro
import forcespro.nlp

from mpclab_common.models.dynamics_models import CasadiDynamicsModel
from mpclab_common.rosbag_utils import rosbagData
from mpclab_common.pytypes import VehicleState, VehicleActuation, VehiclePrediction

from mpclab_controllers.abstract_controller import AbstractController
from mpclab_controllers.utils.controllerTypes import NLLMPCParams

class NL_LMPC(AbstractController):
    def __init__(self, dynamics, track, control_params=NLLMPCParams()):

        assert isinstance(dynamics, CasadiDynamicsModel)
        self.dynamics = dynamics

        self.track = track

        self.dt = control_params.dt
        self.n = control_params.n
        self.d = control_params.d

        # Make sure to check these assumptions about how the state vector is ordered
        self.s_idx = self.n-2
        self.x_tran_idx = self.n-1

        self.N = control_params.N

        self.Q = control_params.Q
        self.R = control_params.R
        self.Q_f = control_params.Q_f
        self.R_d = control_params.R_d
        self.Q_s = control_params.Q_s
        self.l_s = control_params.l_s
        self.Q_ch = control_params.Q_ch

        assert len(self.Q) == self.n
        assert len(self.R) == self.d
        assert len(self.Q_f) == self.n
        assert len(self.R_d) == self.d
        assert len(self.Q_ch) == self.n

        self.state_ub = control_params.state_ub
        self.state_lb = control_params.state_lb
        self.input_ub = control_params.input_ub
        self.input_lb = control_params.input_lb
        self.input_rate_ub = control_params.input_rate_ub
        self.input_rate_lb = control_params.input_rate_lb

        assert len(self.state_ub) == self.n
        assert len(self.state_lb) == self.n
        assert len(self.input_ub) == self.d
        assert len(self.input_lb) == self.d
        assert len(self.input_rate_ub) == self.d
        assert len(self.input_rate_lb) == self.d

        self.optlevel = control_params.optlevel
        self.slack = control_params.slack
        self.solver_dir = control_params.solver_dir

        self.n_ss_pts = control_params.n_ss_pts
        self.n_ss_its = control_params.n_ss_its
        self.ss_selection_weights = np.array(control_params.ss_selection_weights)
        assert len(self.ss_selection_weights) == self.n

        self.ss_dataset = {'state' : [], 'input' : [], 'cost' : []}
        self.n_ss = self.n_ss_pts * self.n_ss_its
        self.n_it = 0
        self.n_data = []
        self.lap_times = []
        self.lap_time = 0

        self.u_prev = np.zeros(self.d)
        self.x_pred = np.zeros((self.N+1, self.n))
        self.u_pred = np.zeros((self.N, self.d))
        self.x_ws = None
        self.u_ws = None
        self.convex_hull_multipliers_ws = (1/self.n_ss)*np.ones(self.n_ss)
        self.halfwidths = None
        self.query_point = None

        self.model = None
        self.options = None
        self.solver = None
        self.solver_name = 'NL_LMPC_solver_forces_pro'

        self.state_input_prediction = VehiclePrediction()
        self.safe_set_points = VehiclePrediction()
        self.safe_set_x = np.zeros((self.n_ss, self.n))

        self.initialized = False

        if self.solver_dir:
            self.solver_dir = pathlib.Path(self.solver_dir).expanduser()  # allow the use of ~
            self._load_solver(self.solver_dir)
        else:
            self._build_solver()

    def initialize(self, ss_states=None, ss_inputs=None, ss_costs=None):
        if ss_states and ss_inputs and ss_costs:
            for i in range(len(ss_states)):
                assert ss_inputs[i].shape[0] == ss_states[i].shape[0]
                assert ss_costs[i].shape[0] == ss_states[i].shape[0]
                self.n_data.append(ss_states[i].shape[0])

                self.ss_dataset['state'].append(ss_states[i])
                self.ss_dataset['input'].append(ss_inputs[i])
                self.ss_dataset['cost'].append(ss_costs[i])

                self.lap_times.append(ss_costs[i][0])

                self.n_it += 1

        self.initialized = True

    def step(self, vehicle_state: VehicleState, env_state=None):
        x_ref = np.tile(np.zeros(self.n), (self.N+1,1))
        info = self.solve(vehicle_state, x_ref)

        self.dynamics.qu2state(vehicle_state, None, self.u_pred[0])
        self.dynamics.qu2prediction(self.state_input_prediction, self.x_pred, self.u_pred)
        self.dynamics.qu2prediction(self.safe_set_points, self.safe_set_x, None)
        self.state_input_prediction.t = vehicle_state.t
        self.safe_set_points.t = vehicle_state.t

        return info

    def add_iteration_data(self, ss_states, ss_inputs, ss_costs):
        assert ss_inputs.shape[0] == ss_states.shape[0]
        assert ss_costs.shape[0] == ss_states.shape[0]

        self.ss_dataset['state'].append(ss_states)
        self.ss_dataset['input'].append(ss_inputs)
        self.ss_dataset['cost'].append(ss_costs)

        self.lap_times.append(ss_costs[0])

        self.n_data.append(ss_states.shape[0])

        self.n_it += 1
        self.lap_time = 0

    def add_point_to_prev_iter(self, state, input):
        # Add points from current iteration to end of last iteration
        state[self.s_idx] += self.track.track_length
        self.ss_dataset['state'][-1] = np.vstack((self.ss_dataset['state'][-1], state.reshape((1,-1))))
        self.ss_dataset['input'][-1] = np.vstack((self.ss_dataset['input'][-1], input.reshape((1,-1))))
        self.ss_dataset['cost'][-1] = np.append(self.ss_dataset['cost'][-1], self.ss_dataset['cost'][-1][-1]-1)

        self.n_data[-1] += 1

    '''
    Select a safe set which is centered about the query point x
    '''
    def _select_safe_set(self, x):
        ss_x = []; ss_u = []; ss_v = []

        sorted_lap_idxs = np.argsort(np.array(self.lap_times))
        for i in range(self.n_ss_its):
            lap_idx = sorted_lap_idxs[min(i, len(sorted_lap_idxs)-1)]
            x_data = self.ss_dataset['state'][lap_idx]
            u_data = self.ss_dataset['input'][lap_idx]
            v_data = self.ss_dataset['cost'][lap_idx]
            weighted_diff = np.multiply(x_data-x, self.ss_selection_weights)
            weighted_norm = la.norm(weighted_diff, ord=1, axis=1)
            closest_idx = np.argmin(weighted_norm)

            if closest_idx >= self.n_ss_pts/2:
                ss_idx = np.arange(closest_idx-int(self.n_ss_pts/2), closest_idx+int(self.n_ss_pts/2))
            else:
                ss_idx = np.arange(closest_idx, closest_idx+int(self.n_ss_pts))

            out_idx = np.argwhere(np.array(ss_idx) == self.n_data[lap_idx])
            if out_idx.size != 0:
                out_idx = out_idx[0,0]
                ss_idx = np.concatenate((np.arange(ss_idx[0]-(len(ss_idx)-out_idx), ss_idx[0]), ss_idx[:out_idx]))

            x_pts = x_data[ss_idx]
            u_pts = u_data[ss_idx]
            v_pts = v_data[ss_idx]

            if lap_idx < self.n_it - 1:
                v_pts = v_pts + v_data[0]
            elif np.argwhere(self.x_pred[:,self.s_idx] > self.track.track_length).size > 0:
                s_pred = self.x_pred[:,self.s_idx]
                curr_lap_remaining = self.N - np.sum(s_pred > self.track.track_length)
                curr_lap_projected = self.lap_time + curr_lap_remaining
                v_pts = v_pts + curr_lap_projected

            ss_x.append(x_pts); ss_u.append(u_pts); ss_v.append(v_pts)

        return np.vstack(ss_x), np.vstack(ss_u), np.concatenate(ss_v)

    def set_warm_start(self, x_ws, u_ws):
        if x_ws.shape[0] != self.N+1 or x_ws.shape[1] != self.n:
            raise(RuntimeError('Warm start state sequence of shape (%i,%i) is incompatible with required shape (%i,%i)' % (x_ws.shape[0],x_ws.shape[1],self.N+1,self.n)))
        if u_ws.shape[0] != self.N or u_ws.shape[1] != self.d:
            raise(RuntimeError('Warm start state sequence of shape (%i,%i) is incompatible with required shape (%i,%i)' % (u_ws.shape[0],u_ws.shape[1],self.N,self.d)))

        self.x_ws = x_ws
        self.u_ws = u_ws

        self.halfwidths = np.zeros(self.N+1)
        for i in range(self.N+1):
            self.halfwidths[i] = self.track.get_halfwidth(self.x_ws[i,self.s_idx])

    def solve(self, state, x_ref, input_prev=None):
        if not self.initialized:
            raise(RuntimeError('NL MPC controller is not initialized, run NL_MPC.initialize() before calling NL_MPC.solve()'))

        x, _ = self.dynamics.state2qu(state)
        if input_prev is not None:
            self.u_prev = self.dynamics.input2u(input_prev)

        if self.x_ws is None:
            warnings.warn('Initial guess of open loop state sequence not provided, using zeros')
            self.x_ws = np.zeros(self.N+1, self.n)
            self.halfwidths = np.zeros(self.N+1)
            for i in range(self.N+1):
                self.halfwidths[i] = self.track.get_halfwidth(self.x_ws[i,self.s_idx])
        if self.u_ws is None:
            warnings.warn('Initial guess of open loop input sequence not provided, using zeros')
            self.u_ws = np.zeros(self.N, self.d)
        if self.query_point is None:
            self.query_point = self.x_ws[-1]

        ss_x, ss_u, ss_v = self._select_safe_set(self.query_point)
        if state.s < 1.0:
            for i in range(ss_x.shape[0]):
                if ss_x[i,self.s_idx] > self.track.track_length:
                    ss_x[i,self.s_idx] -= self.track.track_length

        # Construct initial guess for the decision variables and the runtime problem data
        parameters = []; initial_guess = []
        for i in range(self.N):
            initial_guess.append(self.x_ws[i]) # Initial guess for state
            initial_guess.append(self.u_ws[i]) # Initial guess for input
            if i == 0:
                initial_guess.append(self.u_prev)
            else:
                initial_guess.append(self.u_ws[i-1]) # Initial guess for previous input

            parameters.append(x_ref[i])
            parameters.append(self.Q)
            parameters.append(self.R)
            parameters.append(self.R_d)

            if i == self.N-1:
                # initial_guess.append(self.convex_hull_multipliers_ws) # Initial guess for convex hull multipliers
                initial_guess.append((1/self.n_ss)*np.ones(self.n_ss))
                initial_guess.append(np.zeros(self.n)) # Initial guess for convex hull slack
                parameters.append(self.Q_ch)
                parameters.append(ss_x.flatten())
                parameters.append(ss_v)

            if self.slack:
                initial_guess.append(np.zeros((1,)))
                parameters.append(np.array(self.Q_s).reshape((-1)))
                parameters.append(np.array(self.l_s).reshape((-1)))
                parameters.append(np.array(self.halfwidths[i]-0.17).reshape((-1)))

        initial_guess.append(self.x_ws[-1])
        initial_guess.append(np.zeros((1,))) # Initial guess for slack variable

        parameters.append(x_ref[-1])
        parameters.append(self.Q_f)

        if self.slack:
            initial_guess.append(np.zeros((1,)))
            parameters.append(np.array(self.Q_s).reshape((-1)))
            parameters.append(np.array(self.l_s).reshape((-1)))
            parameters.append(np.array(self.halfwidths[-1]-0.17).reshape((-1)))

        initial_guess = np.concatenate(initial_guess)
        parameters = np.concatenate(parameters)

        problem = dict()
        problem['xinit'] = np.concatenate((x, self.u_prev))
        problem['xfinal'] = np.array([0])
        problem['all_parameters'] = parameters
        problem['x0'] = initial_guess

        output, exitflag, solve_info = self.solver.solve(problem)

        if exitflag == 1:
            info = {"success": True, "return_status": "Successfully Solved", "solve_time": solve_info.solvetime, "info": solve_info, "output": output}
            # Unpack solution
            for k in range(self.N):
                sol = output["x%02d" % (k+1)]
                self.x_pred[k,:] = sol[:self.n]
                self.u_pred[k,:] = sol[self.n:self.n+self.d]
                if k == self.N-1:
                    self.convex_hull_multipliers_ws = sol[self.n+self.d+self.d:self.n+self.d+self.d+self.n_ss]
            sol = output["x%02d" % (self.N+1)]
            self.x_pred[self.N,:] = sol[:self.n]
            self.query_point = self.x_pred[self.N]
            # self.query_point = np.dot(self.convex_hull_multipliers_ws, ss_xp1)

            # Construct initial guess for next iteration
            x_ws = self.x_pred[1:]
            u_ws = self.u_pred[1:]
            x_ws = np.vstack((x_ws, np.array(self.dynamics.f_d_rk4(x_ws[-1], np.dot(self.convex_hull_multipliers_ws, ss_u))).squeeze()))
            u_ws = np.vstack((u_ws, np.dot(self.convex_hull_multipliers_ws, ss_u)))
            self.set_warm_start(x_ws, u_ws)

            self.safe_set_x = ss_x
        else:
            info = {"success": False, "return_status": 'Solving Failed, exitflag = %d' % exitflag, "solve_time": None, "info": solve_info, "output": None}
            self.convex_hull_multipliers_ws = (1/self.n_ss)*np.ones(self.n_ss)

        self.u_prev = self.u_pred[0]
        self.lap_time += 1

        self.add_point_to_prev_iter(x, self.u_pred[0])

        return info

    def get_prediction(self):
        return self.state_input_prediction

    def get_safe_set(self):
        return self.safe_set_points

    def _load_solver(self, solver_dir):
        self.solver = forcespro.nlp.Solver.from_directory(solver_dir)

    '''
    Total stages: N+1
    Stages 0 to N-2:
    - decision variables: z_a = [state, input, input prev, (lane slack)]
    - parameters: p_a = [state ref, state quadratic, input quadratic, input rate quadratic, (lane slack quadratic, lane slack linear, track half width)]
    Stage N-1:
    - decision variables: z_b = [state, input, input prev, convex hull multipliers, convex hull slack, (lane slack)]
    - parameters: p_b = [state ref, state quadratic, input quadratic, input rate quadratic, convex hull slack quadratic, safe set states, safe set cost-to-gos, (lane slack quadratic, lane slack linear, track half width)]
    Stage N:
    - decision variables: z_c = [state, dummy variable, (lane slack)]
    - parameters: p_c = [state ref, state quadratic, (lane slack quadratic, lane slack linear, track half width)]
    '''
    def _build_solver(self):
        def stage_quadratic_cost(z, p):
            q = z[:self.n]
            u = z[self.n:self.n+self.d]
            u_prev = z[self.n+self.d:self.n+self.d+self.d]

            q_ref = p[:self.n]
            Q = ca.diag(p[self.n:self.n+self.n])
            R = ca.diag(p[self.n+self.n:self.n+self.n+self.d])
            R_d = ca.diag(p[self.n+self.n+self.d:self.n+self.n+self.d+self.d])

            return ca.bilin(Q, q-q_ref, q-q_ref) \
                + ca.bilin(R, u, u) \
                + ca.bilin(R_d, u-u_prev, u-u_prev)

        def terminal_quadratic_cost(z, p):
            q = z[:self.n]

            q_ref = p[:self.n]
            Q = ca.diag(p[self.n:self.n+self.n])

            return ca.bilin(Q, q-q_ref, q-q_ref)

        def convex_hull_slack_cost(z, p):
            convex_hull_slack = z[self.n+self.d+self.d+self.n_ss:self.n+self.d+self.d+self.n_ss+self.n]

            Q_ch = ca.diag(p[self.n+self.n+self.d+self.d:self.n+self.n+self.d+self.d+self.n])

            return ca.bilin(Q_ch, convex_hull_slack, convex_hull_slack)

        def lmpc_cost_to_go(z, p):
            convex_hull_multipliers = z[self.n+self.d+self.d:self.n+self.d+self.d+self.n_ss]

            ss_v = p[self.n+self.n+self.d+self.d+self.n+self.n_ss*self.n:self.n+self.n+self.d+self.d+self.n+self.n_ss*self.n+self.n_ss]

            return ca.dot(convex_hull_multipliers, ss_v)

        def dynamics_equality_constraint(z, p):
            q = z[:self.n]
            u = z[self.n:self.n+self.d]

            return self.dynamics.f_d_rk4(q, u)

        def previous_input_equality_constraint(z, p):
            u = z[self.n:self.n+self.d]

            return u

        def terminal_convex_hull_equality_constraint(z, p):
            convex_hull_multipliers = z[self.n+self.d+self.d:self.n+self.d+self.d+self.n_ss]
            convex_hull_slack = z[self.n+self.d+self.d+self.n_ss:self.n+self.d+self.d+self.n_ss+self.n]

            x_ss = p[self.n+self.n+self.d+self.d+self.n:self.n+self.n+self.d+self.d+self.n+self.n_ss*self.n]
            x_ss = ca.reshape(x_ss, (self.n,self.n_ss))

            return ca.vertcat(ca.mtimes(x_ss,convex_hull_multipliers)-convex_hull_slack, ca.sum1(convex_hull_multipliers)-1)

        def input_rate_constraint(z, p):
            u = z[self.n:self.n+self.d]
            u_prev = z[self.n+self.d:self.n+self.d+self.d]

            return u - u_prev

        def lane_slack_cost(z, p):
            lane_slack = z[-1]

            Q_s = p[-3]; l_s = p[-2]

            return Q_s*ca.power(lane_slack, 2) + l_s*lane_slack

        def soft_lane_constraint(z, p):
            e_y = z[self.x_tran_idx]; lane_slack = z[-1]

            half_width = p[-1]

            return ca.vertcat(e_y-lane_slack-half_width, e_y+lane_slack+half_width)

        # Build solver
        self.model = forcespro.nlp.SymbolicModel(self.N+1)

        for i in range(self.N):
            # Number of decision variables
            stage_n_var = self.n + self.d + self.d # q, u, u_prev
            # Number of parameters
            stage_n_par = self.n + self.n + self.d + self.d # q_ref, diag(Q), diag(R), diag(R_d)
            if i == self.N-1:
                stage_n_var += self.n_ss + self.n # convex hull multipliers, convex hull slack
                stage_n_par += self.n + self.n_ss*self.n + self.n_ss # convex hull slack quadratic, safe set states, safe set cost-to-gos
            if self.slack:
                stage_n_var += 1 # lane slack
                stage_n_par += 1 + 1 + 1 # quad slack, lin slack, half width

            self.model.nvar[i] = stage_n_var
            self.model.npar[i] = stage_n_par

            # Objective function
            if i == self.N-1:
                if self.slack:
                    stage_objective_function = lambda z, p: stage_quadratic_cost(z, p) + lmpc_cost_to_go(z, p) + convex_hull_slack_cost(z, p) + lane_slack_cost(z, p)
                else:
                    stage_objective_function = lambda z, p: stage_quadratic_cost(z, p) + lmpc_cost_to_go(z, p) + convex_hull_slack_cost(z, p)
            else:
                if self.slack:
                    stage_objective_function = lambda z, p: stage_quadratic_cost(z, p) + lane_slack_cost(z, p)
                else:
                    stage_objective_function = lambda z, p: stage_quadratic_cost(z, p)

            self.model.objective[i] = stage_objective_function

            # Nonlinear inequality constraints
            stage_n_h = self.d # d_u
            stage_hu = self.input_rate_ub*self.dt
            stage_hl = self.input_rate_lb*self.dt
            if self.slack:
                stage_n_h += 2 # soft lane boundary constraints
                stage_hu = np.concatenate((stage_hu, np.array([0, np.inf])))
                stage_hl = np.concatenate((stage_hl, np.array([-np.inf, 0])))
                stage_inequality_constraints = lambda z, p: ca.vertcat(input_rate_constraint(z, p), soft_lane_constraint(z, p))
            else:
                stage_inequality_constraints = lambda z, p: input_rate_constraint(z, p)

            self.model.nh[i] = stage_n_h
            self.model.ineq[i] = stage_inequality_constraints
            self.model.hu[i] = stage_hu
            self.model.hl[i] = stage_hl

            # Simple upper and lower bounds on decision vector
            stage_ub = np.concatenate((self.state_ub, self.input_ub, self.input_ub))
            stage_lb = np.concatenate((self.state_lb, self.input_lb, self.input_lb))
            if i == self.N-1:
                stage_ub = np.concatenate((stage_ub, np.ones(self.n_ss), np.inf*np.ones(self.n)))
                stage_lb = np.concatenate((stage_lb, np.zeros(self.n_ss), -np.inf*np.ones(self.n)))
            if self.slack:
                # Remove hard constraints on lane boundaries
                stage_ub[self.x_tran_idx] = np.inf; stage_lb[self.x_tran_idx] = -np.inf
                # Positivity constraint on slack variable
                stage_ub = np.append(stage_ub, np.inf)
                stage_lb = np.append(stage_lb, 0)

            self.model.ub[i] = stage_ub
            self.model.lb[i] = stage_lb

            # Equality constraints in the form
            # E_k*z_kp1 = f(z_k, p_k)
            # Where E_k is a selection matrix of dimension n_eq x n_var_kp1
            if i < self.N-2:
                n_eq = self.n + self.d

                E_q = np.hstack((np.eye(self.n), np.zeros((self.n, self.d+self.d))))
                E_u = np.hstack((np.zeros((self.d, self.n+self.d)), np.eye(self.d)))
                E = np.vstack((E_q, E_u))
                if self.slack:
                    E = np.hstack((E, np.zeros((E.shape[0], 1))))

                stage_equality_constraint = lambda z, p: ca.vertcat(dynamics_equality_constraint(z, p), previous_input_equality_constraint(z, p))
            elif i == self.N-2:
                n_eq = self.n + self.d

                E_q = np.hstack((np.eye(self.n), np.zeros((self.n, self.d+self.d+self.n_ss+self.n))))
                E_u = np.hstack((np.zeros((self.d, self.n+self.d)), np.eye(self.d), np.zeros((self.d, self.n_ss+self.n))))
                E = np.vstack((E_q, E_u))
                if self.slack:
                    E = np.hstack((E, np.zeros((E.shape[0], 1))))

                stage_equality_constraint = lambda z, p: ca.vertcat(dynamics_equality_constraint(z, p), previous_input_equality_constraint(z, p))
            elif i == self.N-1:
                n_eq = self.n + self.n + 1

                E_q = np.hstack((np.eye(self.n), np.zeros((self.n,1))))
                E_convex_hull = np.hstack((np.eye(self.n), np.zeros((self.n,1))))
                E_multiplier = np.hstack((np.zeros((1,self.n)), np.ones((1,1))))
                E = np.vstack((E_q, E_convex_hull, E_multiplier))
                if self.slack:
                    E = np.hstack((E, np.zeros((E.shape[0], 1))))

                stage_equality_constraint = lambda z, p: ca.vertcat(dynamics_equality_constraint(z, p), terminal_convex_hull_equality_constraint(z, p))

            self.model.neq[i] = n_eq
            self.model.eq[i] = stage_equality_constraint
            self.model.E[i] = E

        # Number of decision variables
        terminal_n_var = self.n + 1 # q, sum of convex hull multipliers
        if self.slack:
            terminal_n_var += 1

        # Number of parameters
        terminal_n_par = self.n + self.n # q_ref, diag(Q)
        if self.slack:
            terminal_n_par += 1 + 1 + 1 # quad slack, lin slack, half width

        # Objective function
        if self.slack:
            terminal_objective_function = lambda z, p: terminal_quadratic_cost(z, p) + lane_slack_cost(z, p)
        else:
            terminal_objective_function = lambda z, p: terminal_quadratic_cost(z, p)

        terminal_n_h = 0
        terminal_hu = []
        terminal_hl = []
        terminal_inequality_constraints = None
        if self.slack:
            terminal_n_h += 2 # soft lane boundary constraints
            terminal_hu = np.concatenate((terminal_hu, np.array([0, np.inf])))
            terminal_hl = np.concatenate((terminal_hl, np.array([-np.inf, 0])))
            terminal_inequality_constraints = lambda z, p: soft_lane_constraint(z, p)

        terminal_ub = np.append(self.state_ub, np.inf)
        terminal_lb = np.append(self.state_lb, -np.inf)
        if self.slack:
            # Remove hard constraints on lane boundaries
            terminal_ub[self.x_tran_idx] = np.inf; terminal_lb[self.x_tran_idx] = -np.inf
            # Positivity constraint on slack variable
            terminal_ub = np.append(terminal_ub, np.inf)
            terminal_lb = np.append(terminal_lb, 0)

        self.model.nvar[-1] = terminal_n_var
        self.model.npar[-1] = terminal_n_par

        self.model.objective[-1] = terminal_objective_function
        self.model.nh[-1] = terminal_n_h
        self.model.ineq[-1] = terminal_inequality_constraints
        self.model.hu[-1] = terminal_hu
        self.model.hl[-1] = terminal_hl

        self.model.ub[-1] = terminal_ub
        self.model.lb[-1] = terminal_lb

        # Initial conditions
        self.model.xinitidx = list(range(self.n)) + list(range(self.n+self.d, self.n+self.d+self.d))
        # Terminal conditions (sum of convex hull multipliers needs to be equal to 1)
        self.model.xfinalidx = [self.n]

        # Define solver options
        self.options = forcespro.CodeOptions(self.solver_name)

        self.options.overwrite = True
        self.options.printlevel = 0
        self.options.optlevel = self.optlevel
        self.options.BuildSimulinkBlock = False
        self.options.cleanup = True
        self.options.platform = 'Generic'
        self.options.gnu = True
        self.options.sse = True
        self.options.noVariableElimination = True

        self.options.nlp.linear_solver = 'normal_eqs'
        self.options.nlp.TolStat = 1e-6
        self.options.nlp.TolEq = 1e-6
        self.options.nlp.TolIneq = 1e-6

        outputs = []
        for i in range(len(self.model.nvar)):
            outputs.append(('x%02d' % (i+1), i, range(self.model.nvar[i]), ''))

        # Generate solver (this may take a while for high optlevel)
        self.model.generate_solver(self.options, outputs)
        self.install_dir = self.install()  # install the model to ~/.mpclab_controllers
        self.solver = forcespro.nlp.Solver.from_directory(self.install_dir)

# Test script to ensure controller object is functioning properly
if __name__ == "__main__":
    import pdb

    from mpclab_controllers.utils.controllerTypes import NLLMPCParams

    from mpclab_common.dynamics_models import CasadiDynamicCLBicycle
    from mpclab_common.track import get_track
    from mpclab_common.pytypes import VehicleConfig, VehicleCoords, VehicleActuation
    from mpclab_common.rosbag_utils import rosbagData

    n = 6
    d = 2
    N = 10
    dt = 0.1

    vehicle_config = VehicleConfig()
    dynamics = CasadiDynamicCLBicycle(vehicle_config)
    racetrack = get_track('LTrack_barc')

    solver_dir = ''
    # solver_dir = '/home/edward-zhu/mpclab_controllers/mpclab_controllers/lib/mpclab_controllers/NL_LMPC_solver_forces_pro'

    params = NLLMPCParams(n=n, d=d, N=N,
        Q=np.ones(n), R=np.ones(d),
        Q_f=np.ones(n), R_d=np.ones(d),
        Q_s=100, l_s=1, Q_ch=1000*np.ones(n),
        state_ub=np.ones(n), state_lb=-np.ones(n),
        input_ub=np.ones(d), input_lb=-np.ones(d),
        input_rate_ub=np.ones(d), input_rate_lb=-np.ones(d),
        optlevel=1, slack=True, solver_dir=solver_dir,
        n_ss_pts=10, n_ss_its=3, ss_selection_weights=np.ones(n))

    state_names = ['v_long', 'v_tran', 'psidot', 'e_psi', 's', 'x_tran']
    input_names = ['u_a', 'u_steer']

    # If a path to a rosbag file is provided, load lap data to initialize safe set
    # init_ss_states = []; init_ss_inputs = []; init_ss_costs = []
    # safe_set_init_data_file = '/home/edward-zhu/barc_data/barc_sim_pid_bag_12-02-2020_19-10-01/barc_sim_pid_bag_12-02-2020_19-10-01_0.db3'
    # # safe_set_init_data_file = ''
    # safe_set_topic = '/experiment/barc_1/closed_loop_traj'
    # rb_data = rosbagData(safe_set_init_data_file)
    # attributes = state_names + input_names + ['lap_num']
    # ss_data = rb_data.read_from_topic_to_numpy_array(safe_set_topic, attributes)
    #
    # first_nonzero_idx = np.where(np.any(ss_data[:,:n], axis=1))[0][0]
    # ss_data = ss_data[first_nonzero_idx:]
    #
    # lap = 0
    # lap_ss_states = []; lap_ss_inputs = []
    # # Skip the last lap because the data from that is typically incomplete
    # for i in range(ss_data.shape[0]):
    #     if ss_data[i,-1] > lap:
    #         print('===== Lap with %i data points added to LMPC safe set =====' % (len(lap_ss_states)))
    #         init_ss_states.append(np.array(lap_ss_states))
    #         init_ss_inputs.append(np.array(lap_ss_inputs))
    #         init_ss_costs.append(np.arange(len(lap_ss_states)-1, -1, -1))
    #         lap_ss_states = []; lap_ss_inputs = []
    #         lap += 1
    #
    #     lap_ss_states.append(ss_data[i,:n])
    #     lap_ss_inputs.append(ss_data[i,n:n+d])
    #
    # idx = np.random.randint(1, init_ss_states[0].shape[0])
    # x_0 = init_ss_states[0][idx]
    # x_ref = np.zeros((N+1,n))
    # u_prev = init_ss_inputs[0][idx-1]
    # x_ws = init_ss_states[0][idx:idx+N+1]
    # u_ws = init_ss_inputs[0][idx:idx+N]

    # state = VehicleCoords()
    # dynamics.q2state(state, x_0)
    # input_prev = VehicleActuation()
    # dynamics.u2input(input_prev, u_prev)

    x_ws = np.zeros((N+1, n))
    u_ws = np.zeros((N, d))

    nl_lmpc = NL_LMPC(dynamics, racetrack, params)
    nl_lmpc.initialize(init_ss_states, init_ss_inputs, init_ss_costs)
    nl_lmpc.set_warm_start(x_ws, u_ws)
    nl_lmpc.solver.help()
    print('NL LMPC Controller instantiated successfully')

    # x = np.array([1,1,1,1,0.1,1])
    # x_ss, u_ss, v_ss = nl_lmpc._select_safe_set(x)
    # pdb.set_trace()

    u, preds, ss_selected, info = nl_lmpc.solve(state, x_ref, input_prev)

    print(info)

    pdb.set_trace()

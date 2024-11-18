#!/usr/bin python3

import pdb
import array
import warnings

import numpy as np

import casadi as ca

import sys, os, pathlib

sys.path.append(os.path.join(os.path.expanduser('~'), 'forces_pro_client'))
import forcespro
import forcespro.nlp

from mpclab_common.models.dynamics_models import CasadiDynamicsModel
from mpclab_common.pytypes import VehicleState, VehicleActuation, VehiclePrediction

from mpclab_controllers.abstract_controller import AbstractController
from mpclab_controllers.utils.controllerTypes import MPCCParams


class MPCC(AbstractController):
    def __init__(self, dynamics, track, control_params=MPCCParams(), name=None):

        assert isinstance(dynamics, CasadiDynamicsModel)
        self.dynamics = dynamics

        self.track = track

        self.dt = control_params.dt
        self.n = control_params.n
        self.d = control_params.d

        # Slack
        self.slack = control_params.slack
        self.Q_s = control_params.Q_s
        self.l_s = control_params.l_s

        self.N = control_params.N

        self.m = self.dynamics.model_config.mass
        self.lf = self.dynamics.model_config.wheel_dist_front
        self.lr = self.dynamics.model_config.wheel_dist_rear

        self.lencar = self.lf + self.lr
        self.widthcar = self.lencar / 1.7  # Ratio of car length to width

        # MPCC params
        self.N = control_params.N
        self.Qc = control_params.Qc
        self.Ql = control_params.Ql
        self.Q_theta = control_params.Q_theta
        self.R_d = control_params.R_d
        self.R_delta = control_params.R_delta

        # Input Box Constraints
        self.state_ub = control_params.state_ub
        self.state_lb = control_params.state_lb
        self.input_ub = control_params.input_ub
        self.input_lb = control_params.input_lb
        self.input_rate_ub = control_params.input_rate_ub
        self.input_rate_lb = control_params.input_rate_lb

        self.trackvars = ['sval', 'tval', 'xtrack', 'ytrack', 'psitrack', '', '', 'g_upper', 'g_lower']
        self.uvars = ['u_a', 'u_delta', 'u_theta']
        if self.slack:
            self.zvars = ['vx', 'vy', 'psidot', 'posx', 'posy', 'psi', 'u_a', 'u_delta', 'u_theta', 'u_a_prev', \
                          'u_delta_prev', 'u_theta_prev', 'obs_slack']
        else:
            self.zvars = ['vx', 'vy', 'psidot', 'posx', 'posy', 'psi', 'u_a', 'u_delta', 'u_theta', 'u_a_prev', \
                          'u_delta_prev', 'u_theta_prev']
        if self.slack:
            self.pvars = ['xt', 'yt', 'psit', 'theta_hat', 'Qc', 'Ql', 'Q_theta', 'R_d', 'R_delta', \
                          'r', 'x_ob', 'y_ob', 'psi_ob', 'l_ob', 'w_ob', 'deactivate_ob', 'Q_s', 'l_s']
        else:
            self.pvars = ['xt', 'yt', 'psit', 'theta_hat', 'Qc', 'Ql', 'Q_theta', 'R_d', 'R_delta', \
                      'r', 'x_ob', 'y_ob', 'psi_ob', 'l_ob', 'w_ob', 'deactivate_ob']

        self.half_width = track.half_width
        self.track_lu_table = track.table
        self.track_length = track.track_length
        self.table_density = track.table_density

        self.optlevel = control_params.optlevel

        self.solver_dir = control_params.solver_dir

        self.u_prev = np.zeros(self.d)
        self.x_pred = np.zeros((self.N, self.n))
        self.u_pred = np.zeros((self.N, self.d))
        self.x_ws = None
        self.u_ws = None

        self.model = None
        self.options = None
        self.solver = None
        if name is None:
            self.solver_name = 'MPCC_solver_forces_pro'
        else:
            self.solver_name = name
        self.state_input_prediction = VehiclePrediction()

        self.initialized = False

        self.first = True

    def initialize(self):
        if self.solver_dir:
            self.solver_dir = pathlib.Path(self.solver_dir).expanduser()  # allow the use of ~
            self._load_solver(self.solver_dir)
        else:
            self._build_solver()

        self.initialized = True

    def get_prediction(self):
        return self.state_input_prediction

    def set_warm_start(self, x_ws, u_ws):
        if x_ws.shape[0] != self.N or x_ws.shape[1] != self.n: # TODO: self.N+1
            raise (RuntimeError(
                'Warm start state sequence of shape (%i,%i) is incompatible with required shape (%i,%i)' % (
                x_ws.shape[0], x_ws.shape[1], self.N, self.n)))
        if u_ws.shape[0] != self.N or u_ws.shape[1] != self.d:
            raise (RuntimeError(
                'Warm start state sequence of shape (%i,%i) is incompatible with required shape (%i,%i)' % (
                u_ws.shape[0], u_ws.shape[1], self.N, self.d)))

        self.x_ws = x_ws
        self.u_ws = u_ws

        # During the initial warmstart from PID, not interested in virtual theta values
        if self.first:
            self.u_ws[:, self.uvars.index('u_theta')] = np.zeros(self.N)
            self.first = False

        self.theta_prev = self.u_ws[:, self.uvars.index('u_theta')]
        if self.theta_prev[0] > self.track_length:
            self.theta_prev -= self.track_length


    def step(self, estimated_state: VehicleState, env_state : VehiclePrediction, env_state_: VehicleState):
        obstacle = dict()  # TODO: Change format for new VehiclePrediction class
        x = []; y = []; psi = []
        if env_state and env_state.s and env_state.x_tran and env_state.e_psi:
            for s, x_tran, e_psi in zip(env_state.s, env_state.x_tran, env_state.e_psi):
                global_coord = self.track.local_to_global((s, x_tran, e_psi))
                x.append(global_coord[0])
                y.append(global_coord[1])
                psi.append(global_coord[2])
        elif env_state and env_state.x and env_state.y and env_state.psi:
            x.append(env_state_.x.x)
            y.append(env_state_.x.y)
            psi.append(env_state_.e.psi)
            for x_, y_, psi_ in zip(env_state.x, env_state.y, env_state.psi):
                x.append(x_)
                y.append(y_)
                psi.append(psi_)
        obstacle["x_ob"] = x if x else np.tile(0, (self.N,))
        obstacle["y_ob"] = y if y else np.tile(0, (self.N,))
        obstacle["psi_ob"] = psi if psi else np.tile(0, (self.N,))
        obstacle["obs_deactivate"] = 0  # TODO: reactivate
        obstacle["l_ob"] = self.lencar
        obstacle["w_ob"] = self.widthcar
        control, state_input_prediction, info = self.solve(estimated_state, obstacle)

        estimated_state.u.u_a = control.u_a
        estimated_state.u.u_steer = control.u_steer
        self.state_input_prediction = state_input_prediction
        return info

    def solve(self, state, obstacle: dict = None):
        if not self.initialized:
            raise (RuntimeError(
                'MPCC controller is not initialized, run MPCC.initialize() before calling MPCC.solve()'))

        x_ob = obstacle['x_ob']
        y_ob = obstacle['y_ob']
        psi_ob = obstacle['psi_ob']
        l_ob = obstacle['l_ob']
        w_ob = obstacle['w_ob']
        obs_deactivate = obstacle['obs_deactivate']

        x, _ = self.dynamics.state2qu(state)

        if self.x_ws is None:
            warnings.warn('Initial guess of open loop state sequence not provided, using zeros')
            self.x_ws = np.zeros((self.N, self.n))
        if self.u_ws is None:
            warnings.warn('Initial guess of open loop input sequence not provided, using zeros')
            self.u_ws = np.zeros((self.N, self.d))

        # get track linearization
        index_lin_points = self.table_density * self.theta_prev
        index_lin_points = index_lin_points.astype(np.int32)
        track_lin_points = self.track_lu_table[index_lin_points, :]
        #######################################################################
        # set params and warmstart
        parameters = []; initial_guess = []
        for stageidx in range(self.N):
            initial_guess.append(self.x_ws[stageidx])  # x
            initial_guess.append(self.u_ws[stageidx])  # u
            initial_guess.append(self.u_ws[stageidx - 1])  # u_prev

            stage_p = []
            stage_p.append(track_lin_points[stageidx, self.trackvars.index('xtrack')])
            stage_p.append(track_lin_points[stageidx, self.trackvars.index('ytrack')])
            stage_p.append(track_lin_points[stageidx, self.trackvars.index('psitrack')])
            stage_p.append(track_lin_points[stageidx, self.trackvars.index('sval')])  # theta_hat
            stage_p.extend([self.Qc, self.Ql, self.Q_theta, self.R_d, self.R_delta, self.half_width])
            stage_p.extend([x_ob[stageidx], y_ob[stageidx], psi_ob[stageidx], l_ob, w_ob,
                            obs_deactivate])  # deactivate obstacle by default

            parameters.append(stage_p)
            if self.slack:
                initial_guess.append(np.zeros((1,)))
                parameters.append(np.array(self.Q_s).reshape((-1)))
                parameters.append(np.array(self.l_s).reshape((-1)))

        parameters = np.concatenate(parameters)
        initial_guess = np.concatenate(initial_guess)

        # problem dictionary, arrays have to be flattened
        problem = dict()
        # problem["xinit"] = np.concatenate((x, self.u_prev))
        problem["xinit"] = np.concatenate((x, self.u_prev))
        problem["all_parameters"] = parameters
        problem["x0"] = initial_guess

        output, exitflag, solve_info = self.solver.solve(problem)

        if exitflag == 1:
            info = {"success": True, "return_status": "Successfully Solved", "solve_time": solve_info.solvetime,
                    "info": solve_info}
            for k in range(self.N):
                sol = output["x%02d" % (k + 1)]
                self.x_pred[k, :] = sol[:self.n]
                self.u_pred[k, :] = sol[self.n:self.n + self.d]

            # Construct initial guess for next iteration
            x_ws = self.x_pred[1:]
            u_ws = self.u_pred[1:]
            x_ws = np.vstack((x_ws, np.array(
                self.dynamics.f_d_rk4(x_ws[-1], u_ws[-1][:-1])).squeeze()))
            u_ws = np.vstack((u_ws, u_ws[-1])) # stack previous input
            self.set_warm_start(x_ws, u_ws)

            u = self.u_pred[0]

            state_input_prediction = VehiclePrediction(t=state.t,
                                                       x=array.array('d', self.x_pred[:, self.zvars.index('posx')]),
                                                       y=array.array('d', self.x_pred[:, self.zvars.index('posy')]),
                                                       psi=array.array('d', self.x_pred[:, self.zvars.index('psi')]),
                                                       v_x=array.array('d', self.x_pred[:, self.zvars.index('vx')]),
                                                       v_y=array.array('d', self.x_pred[:, self.zvars.index('vy')]),
                                                       psidot=array.array('d', self.x_pred[:, self.zvars.index('psidot')]),
                                                       u_a=array.array('d', self.u_pred[:, 0]),
                                                       u_steer=array.array('d', self.u_pred[:, 1]))
        else:
            info = {"success": False, "return_status": 'Solving Failed, exitflag = %d' % exitflag, "solve_time": None,
                    "info": solve_info}
            u = np.zeros(self.d)
            state_input_prediction = VehiclePrediction()

            # REINITIALIZE
            self.x_ws[1: , self.zvars.index('posx'):self.zvars.index('psi') + 1] = \
                track_lin_points[1:, self.trackvars.index('xtrack'):self.trackvars.index('psitrack') + 1]
            self.x_ws[1:, self.zvars.index('vx')] = 0.2 * np.ones((self.N-1,))
            self.set_warm_start(self.x_ws, self.u_ws)
        print(info)
        self.u_prev = u

        control = VehicleActuation()
        self.dynamics.u2input(control, u)

        return control, state_input_prediction, info

    def _load_solver(self, solver_dir):
        self.solver = forcespro.nlp.Solver.from_directory(solver_dir)


    def _build_solver(self):
        def nonlinear_ineq(z, p):
            # Extract parameters
            xt = p[self.pvars.index('xt')]
            yt = p[self.pvars.index('yt')]
            psit = p[self.pvars.index('psit')]
            sin_psit = ca.sin(psit)
            cos_psit = ca.cos(psit)
            theta_hat = p[self.pvars.index('theta_hat')]
            r = p[self.pvars.index('r')]

            # Extract relevant states
            posx = z[self.zvars.index('posx')]
            posy = z[self.zvars.index('posy')]
            theta = z[self.zvars.index('u_theta')]
            obstacle_slack = z[self.zvars.index('obs_slack')]
            # Compute approximate linearized contouring and lag error
            xt_hat = xt + cos_psit * (theta - theta_hat)
            yt_hat = yt + sin_psit * (theta - theta_hat)

            # Inside: track <=> tval <= 0
            tval = (xt_hat - posx) ** 2 + (yt_hat - posy) ** 2 - (r - self.widthcar) ** 2

            # Ellipsoidal obstacle
            x_ob = p[self.pvars.index('x_ob')]
            y_ob = p[self.pvars.index('y_ob')]
            psi_ob = p[self.pvars.index('psi_ob')]
            l_ob = p[self.pvars.index('l_ob')]
            w_ob = p[self.pvars.index('w_ob')]
            deactivate_ob = p[self.pvars.index('deactivate_ob')]

            # Implicit ellipse eqn
            dx = posx - x_ob
            dy = posy - y_ob
            s = ca.sin(psi_ob)
            c = ca.cos(psi_ob)
            # Tighten constraint with car length/width
            a = np.sqrt(2) * (l_ob / 2 + self.lencar / 2)
            b = np.sqrt(2) * (w_ob / 2 + self.widthcar / 2)

            # Implicit ellipse value ielval = 1 defines obstacle ellipse
            ielval = (1 / a ** 2) * (c * dx + s * dy) * (c * dx + s * dy) + (1 / b ** 2) * (s * dx - c * dy) * (
                    s * dx - c * dy)
            # Constraint value squished constraint obsval>0.5 -> outside
            if self.slack:
                obsval = 1 / (1 + ca.exp(-(ielval - 1 + deactivate_ob))) + obstacle_slack
            else:
                obsval = 1 / (1 + ca.exp(-(ielval - 1 + deactivate_ob)))
            # Concatenate
            hval = ca.vertcat(tval, obsval)
            return hval

        def objective_cost(z, p):
            # Extract parameters
            xt = p[self.pvars.index('xt')]
            yt = p[self.pvars.index('yt')]
            psit = p[self.pvars.index('psit')]
            sin_psit = ca.sin(psit)
            cos_psit = ca.cos(psit)
            theta_hat = p[self.pvars.index('theta_hat')]
            Qc = p[self.pvars.index('Qc')]
            Ql = p[self.pvars.index('Ql')]
            Q_theta = p[self.pvars.index('Q_theta')]
            R_d = p[self.pvars.index('R_d')]
            R_delta = p[self.pvars.index('R_delta')]

            # Extract states
            posx = z[self.zvars.index('posx')]
            posy = z[self.zvars.index('posy')]
            theta = z[self.zvars.index('u_theta')]
            theta_prev = z[self.zvars.index('u_theta_prev')]
            obstacle_slack = z[self.zvars.index('obs_slack')]
            Q_s = p[self.pvars.index('Q_s')]
            l_s = p[self.pvars.index('l_s')]
            thetadot = theta - theta_hat

            # Extract inputs
            u_a = z[self.zvars.index('u_a')]
            u_delta = z[self.zvars.index('u_delta')]

            # Approximate linearized contouring and lag error
            xt_hat = xt + cos_psit * (theta - theta_hat)
            yt_hat = yt + sin_psit * (theta - theta_hat)

            e_cont = sin_psit * (xt_hat - posx) - cos_psit * (yt_hat - posy)
            e_lag = cos_psit * (xt_hat - posx) + sin_psit * (yt_hat - posy)
            if self.slack:
                cost = ca.bilin(Qc, e_cont, e_cont) + ca.bilin(Ql, e_lag, e_lag) + \
                       ca.bilin(R_d, u_a, u_a) + ca.bilin(R_delta, u_delta, u_delta) - Q_theta * thetadot + obstacle_slack*obstacle_slack*Q_s + \
                obstacle_slack*l_s
            else:
                cost = ca.bilin(Qc, e_cont, e_cont) + ca.bilin(Ql, e_lag, e_lag) + \
                       ca.bilin(R_d, u_a, u_a) + ca.bilin(R_delta, u_delta, u_delta) -Q_theta * thetadot
            return cost

        def stage_input_rate_constraint(z, p):
            # Extract inputs
            u_a = z[self.zvars.index('u_a')]
            u_delta = z[self.zvars.index('u_delta')]

            u_a_prev = z[self.zvars.index('u_a_prev')]
            u_delta_prev = z[self.zvars.index('u_delta_prev')]

            u_a_dot = u_a - u_a_prev
            u_delta_dot = u_delta - u_delta_prev

            return ca.vertcat(u_a_dot, u_delta_dot)

        def stage_equality_constraint(z, p):
            q = z[:self.n]
            u = z[self.n:self.n+self.d]

            integrated = self.dynamics.f_d_rk4(q, u[:-1])

            return ca.vertcat(integrated, u)

        # Forces model
        self.model = forcespro.nlp.SymbolicModel(self.N)

        # Number of parameters
        """STAGE"""
        # for i in range(self.N):
        if self.slack:
            self.model.nvar = self.n + self.d + self.d + 1 # Stage variables z = [x, u, u_prev]
        else:
            self.model.nvar   = self.n + self.d + self.d # Stage variables z = [x, u, u_prev]

        self.model.nh     = 2 + self.d - 1    # # of inequality constraints

        self.model.neq    = self.n + self.d  # of equality constraints
        self.model.npar   = len(self.pvars)  # number of real-time parameters (p)

        self.model.objective = lambda z, p: objective_cost(z, p)

        # dynamics only in state Variables

        stage_E_q = np.hstack((np.eye(self.n), np.zeros((self.n, self.d + self.d))))
        stage_E_u = np.hstack((np.zeros((self.d, self.n + self.d)), np.eye(self.d)))
        stage_E = np.vstack((stage_E_q, stage_E_u))
        if self.slack:
            stage_E = np.hstack((stage_E, np.zeros((stage_E.shape[0], 1))))
        self.model.E = stage_E
        self.model.eq = stage_equality_constraint


        self.model.ineq = lambda z, p: ca.vertcat(nonlinear_ineq(z, p),  # ego, enemy states
                                                  stage_input_rate_constraint(z, p))
        if self.slack:
            self.model.hu = np.concatenate((np.array([0.0, 5.0]), self.input_rate_ub * self.dt))  # upper bound for nonlinear constraints, pg. 23 of FP manual
            self.model.hl = np.concatenate((np.array([-100.0, 0.8]), self.input_rate_lb * self.dt))
        else:
            self.model.hu = np.concatenate((np.array([0.0, 5.0]),
                                            self.input_rate_ub * self.dt))  # upper bound for nonlinear constraints, pg. 23 of FP manual
            self.model.hl = np.concatenate((np.array([-100.0, 0.50]), self.input_rate_lb * self.dt))

        # Input box constraints
        if self.slack:
            self.model.ub = np.concatenate((self.state_ub, self.input_ub, self.input_ub, np.array([0.5])))  # [x, u, u_prev]
            self.model.lb = np.concatenate((self.state_lb, self.input_lb, self.input_lb, np.array([0.0])))  # [x, u, u_prev]
        else:
            self.model.ub = np.concatenate((self.state_ub, self.input_ub, self.input_ub))  # [x, u, u_prev]
            self.model.lb = np.concatenate((self.state_lb, self.input_lb, self.input_lb))  # [x, u, u_prev]
        # put initial condition on all state variables x
        self.model.xinitidx = list(np.arange(self.n)) + list(np.arange(self.n + self.d, self.n + self.d + self.d))  # x, u_prev

        # Set solver options
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
        self.options.nlp.TolStat = 1e-4
        self.options.nlp.TolEq = 1e-4
        self.options.nlp.TolIneq = 1e-4

        # Creates code for symbolic model formulation given above, then contacts server to generate new solver
        self.model.generate_solver(self.options)
        self.install_dir = self.install()  # install the model to ~/.mpclab_controllers
        self.solver = forcespro.nlp.Solver.from_directory(self.install_dir)

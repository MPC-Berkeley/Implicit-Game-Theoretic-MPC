#!/usr/bin python3

import warnings
from typing import List

import casadi
import numpy as np

import casadi as ca

import sys, os, pathlib

sys.path.append(os.path.join(os.path.expanduser('~'), 'forces_pro_client'))
import forcespro
import forcespro.nlp

from mpclab_common.models.dynamics_models import CasadiDynamicsModel
from mpclab_common.pytypes import VehicleState, VehicleActuation, VehiclePrediction
from mpclab_common.models.obstacle_types import RectangleObstacle

from mpclab_controllers.abstract_controller import AbstractController
from mpclab_controllers.utils.controllerTypes import MPCCFullModelParams


class MPCC_H2H_DECISION(AbstractController):
    def __init__(self, dynamics, track, control_params=MPCCFullModelParams(), name=None):

        assert isinstance(dynamics, CasadiDynamicsModel)
        self.dynamics = dynamics

        self.track = track

        self.dt = control_params.dt
        self.n = control_params.n
        self.d = control_params.d

        self.slack = control_params.slack
        # Obstacle slack
        self.Q_cs = control_params.Q_cs
        self.l_cs = control_params.l_cs

        # Track slack
        self.Q_ts = control_params.Q_ts
        self.l_ts = control_params.l_ts

        self.N = control_params.N

        self.m = self.dynamics.model_config.mass
        self.lf = self.dynamics.model_config.wheel_dist_front
        self.lr = self.dynamics.model_config.wheel_dist_rear

        self.lencar = self.lf + self.lr
        self.widthcar = self.lencar / 2.5  # Ratio of car length to width

        # MPCC params
        self.N = control_params.N
        self.Qc = control_params.Qc
        self.Ql = control_params.Ql
        self.Q_theta = control_params.Q_theta
        self.Q_xref = control_params.Q_xref
        self.R_d = control_params.R_d
        self.R_delta = control_params.R_delta

        # Input Box Constraints
        self.state_ub = control_params.state_ub
        self.state_lb = control_params.state_lb
        self.input_ub = control_params.input_ub
        self.input_lb = control_params.input_lb
        self.input_rate_ub = control_params.input_rate_ub
        self.input_rate_lb = control_params.input_rate_lb

        self.uvars = ['u_a', 'u_delta']
        if self.slack:
            self.zvars = ['vx', 'vy', 'psidot', 'posx', 'posy', 'psi', 'e_psi', 's', 'x_tran', 'u_a', 'u_delta',
                          'u_a_prev', 'u_delta_prev', 'obs_slack', 'track_slack']
        else:
            self.zvars = ['vx', 'vy', 'psidot', 'posx', 'posy', 'psi', 'e_psi', 's', 'x_tran', 'u_a', 'u_delta',
                          'u_a_prev', 'u_delta_prev']
        if self.slack:
            self.pvars = ['xref',
                          'xref_scale',
                          's0',
                          'x_tran_obs',
                          'dir',
                          'kp1_0', 'kp1_1', 'kp1_2', 'kp1_3', 'kp1_4', 'kp1_5',
                          'kp2_0', 'kp2_1', 'kp2_2', 'kp2_3', 'kp2_4', 'kp2_5',
                          'kp3_0', 'kp3_1', 'kp3_2', 'kp3_3', 'kp3_4', 'kp3_5',
                          'kp4_0', 'kp4_1', 'kp4_2', 'kp4_3', 'kp4_4', 'kp4_5',
                          'kp5_0', 'kp5_1', 'kp5_2', 'kp5_3', 'kp5_4', 'kp5_5',
                          'track_length',
                          'Qc', 'Ql', 'Q_theta', 'Q_xref', 'R_d', 'R_delta',
                          'r', 'x_ob', 'y_ob', 's', 'psi_ob', 'l_ob', 'w_ob', 'deactivate_ob', 'Q_cs', 'l_cs', 'Q_ts',
                          'l_ts']
        else:
            self.pvars = ['xref', 'xt', 'yt', 'psit', 'theta_hat', 'Qc', 'Ql', 'Q_theta', 'Q_xref', 'R_d', 'R_delta', \
                          'r', 'x_ob', 'y_ob', 's', 'psi_ob', 'l_ob', 'w_ob', 'deactivate_ob']

        self.half_width = track.half_width
        self.track_length = track.track_length

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
            self.solver_name = 'MPCC_H2H_solver_forces_pro'
        else:
            self.solver_name = name
        self.state_input_prediction = []
        self.state_input_prediction.append(VehiclePrediction())
        self.state_input_prediction.append(VehiclePrediction())
        self.best_choice = 0

        self.initialized = False

        self.first = True
        self.theta_prev = []
        self.s_prev = []

    def initialize(self):
        if self.solver_dir:
            self.solver_dir = pathlib.Path(self.solver_dir).expanduser()  # allow the use of ~
            self._load_solver(self.solver_dir)
        else:
            self._build_solver()

        self.initialized = True

    def get_prediction(self):
        return self.state_input_prediction[self.best_choice].copy()

    def get_predictions(self):
        return self.state_input_prediction

    def set_warm_start(self, x_ws, u_ws):
        if x_ws.shape[0] != self.N or x_ws.shape[1] != self.n:  # TODO: self.N+1
            raise (RuntimeError(
                'Warm start state sequence of shape (%i,%i) is incompatible with required shape (%i,%i)' % (
                    x_ws.shape[0], x_ws.shape[1], self.N, self.n)))
        if u_ws.shape[0] != self.N or u_ws.shape[1] != self.d:
            raise (RuntimeError(
                'Warm start state sequence of shape (%i,%i) is incompatible with required shape (%i,%i)' % (
                    u_ws.shape[0], u_ws.shape[1], self.N, self.d)))

        self.x_ws = x_ws
        self.u_ws = u_ws

    def step(self, ego_state: VehicleState, tv_state: VehicleState, tv_pred: List[VehiclePrediction] = None,
             is_ego=False, policy=None):
        # evaluate policy for behavior
        if policy is None:
            policy = self.aggressive_blocking_policy
        elif policy == "aggressive_blocking":
            policy = self.aggressive_blocking_policy
        elif policy == "only_right":
            policy = self.only_right_blocking_policy
        x_ref, blocking = policy(ego_state, tv_state)
        x_ref = np.tile(x_ref, (self.N,))
        xref_scale = ego_state.p.s-tv_state.p.s
        # Initialize Obstacle List, TODO maybe keep this to avoid repetitive list filling?
        obstacle = []
        obstacle.append([])
        obstacle.append([])
        for _ in range(self.N):
            obstacle[0].append(RectangleObstacle())
            obstacle[1].append(RectangleObstacle())

        # find out if prediction is parametric, global or both
        contains_parametric = tv_pred[0] is not None and tv_pred[0].s is not None
        contains_global = tv_pred[0] is not None and tv_pred[0].x is not None

        obstacle[0][0] = RectangleObstacle(xc=tv_state.x.x, yc=tv_state.x.y, psi=tv_state.e.psi, s=tv_state.p.s,
                                        x_tran=tv_state.p.x_tran,
                                        h=self.lencar, w=self.widthcar)
        obstacle[1][0] = obstacle[0][0]
        # if not blocking, we need to use all predictions! Otherwise, only interested in the current one to avoid
        # crashing
        length = 1 if blocking else len(tv_pred)
        for k in range(length):
            if not blocking and tv_pred[k] is not None:
                offs = 0
                if tv_pred[k].t < tv_state.t:
                    offs = 1
                if contains_parametric and contains_global:
                    for i, (s, x, y, psi, x_tran) in enumerate(zip(tv_pred[k].s[1+offs:], tv_pred[k].x[1+offs:], tv_pred[k].y[1+offs:], tv_pred[k].psi[1+offs:],
                                                                   tv_pred[k].x_tran[1+offs:])):
                        if i >= self.N:
                            break
                        obstacle[k][i + 1] = RectangleObstacle(xc=x, yc=y, psi=psi, s=s,
                                                            x_tran=x_tran,
                                                            h=self.lencar, w=self.widthcar)
                elif contains_parametric:
                    for i, (s, x_tran, e_psi) in enumerate(zip(tv_pred[k].s[1+offs:], tv_pred[k].x_tran[1+offs:], tv_pred[k].e_psi[1+offs:])):
                        if i >= self.N:
                            break
                        global_coord = self.track.local_to_global((s, x_tran, e_psi))
                        obstacle[k][i + 1] = RectangleObstacle(xc=global_coord[0], yc=global_coord[1], psi=global_coord[2],
                                                            s=s,
                                                            x_tran=x_tran,
                                                            h=self.lencar, w=self.widthcar)
                elif contains_global:
                    for i, (x, y, psi) in enumerate(zip(tv_pred[k].x[1+offs:], tv_pred[k].y[1+offs:], tv_pred[k].psi[1+offs:])):
                        if i >= self.N:  # TODO: temporary solution for NL_MPC N+1 length prediction
                            break
                        obstacle[k][i + 1] = RectangleObstacle(xc=x, yc=y, psi=psi, h=self.lencar, w=self.widthcar)

        if not blocking:
            control_l, info_l, u_l = self.solve(ego_state, ego_state.p.s, x_ref, xref_scale, obstacle[0],  -1, blocking, 0)
            control_r, info_r, u_r = self.solve(ego_state, ego_state.p.s, x_ref, xref_scale, obstacle[1],  1, blocking, 1)
            if not info_l["success"]:
                ego_state.u.u_a = control_r.u_a
                ego_state.u.u_steer = control_r.u_steer
                info = info_r
                self.best_choice = 1
                self.u_prev = u_r
            elif not info_r["success"]:
                ego_state.u.u_a = control_l.u_a
                ego_state.u.u_steer = control_l.u_steer
                info = info_l
                self.best_choice = 0
                self.u_prev = u_l
            elif self.state_input_prediction[0].s[-1] > self.state_input_prediction[1].s[-1]:
                ego_state.u.u_a = control_l.u_a
                ego_state.u.u_steer = control_l.u_steer
                info = info_l
                self.best_choice = 0
                self.u_prev = u_l
            else:
                ego_state.u.u_a = control_r.u_a
                ego_state.u.u_steer = control_r.u_steer
                info = info_r
                self.best_choice = 1
                self.u_prev = u_r
        else:
            control_l, info_l, u_l = self.solve(ego_state, ego_state.p.s, x_ref, xref_scale, obstacle[0],  -1, blocking, 0)
            ego_state.u.u_a = control_l.u_a
            ego_state.u.u_steer = control_l.u_steer
            info = info_l
            self.best_choice = 0
            self.u_prev = u_l
            self.state_input_prediction[1] = VehiclePrediction()

        return info, blocking

    def solve(self, state: VehicleState, s0, x_ref: np.array, xref_scale, obstacle: List[RectangleObstacle], direction, blocking, id):
        if not self.initialized:
            raise (RuntimeError(
                'MPCC controller is not initialized, run MPCC.initialize() before calling MPCC.solve()'))

        x, _ = self.dynamics.state2qu(state)

        if self.x_ws is None:
            warnings.warn('Initial guess of open loop state sequence not provided, using zeros')
            self.x_ws = np.zeros((self.N, self.n))
        if self.u_ws is None:
            warnings.warn('Initial guess of open loop input sequence not provided, using zeros')
            self.u_ws = np.zeros((self.N, self.d))

        # Get track linearization (center-line approximation): (x, y, psi, theta_prev (used in MPCC constraints))
        # --REMOVED--

        # Set up real-time parameters and warm-start
        parameters = []
        initial_guess = []
        key_pts = []
        current_s = state.p.s
        while current_s < 0: current_s += self.track.track_length
        while current_s >= self.track.track_length: current_s -= self.track.track_length
        if len(self.track.key_pts) < 5:
            for i in range(len(self.track.key_pts)):
                key_pts.append(self.track.key_pts[i])
            while len(key_pts) < 5:
                key_pts.append(key_pts[-1])
        else:
            key_pt_idx_s = np.where(current_s >= self.track.key_pts[:, 3])[0][-1]
            difference = max(0, (key_pt_idx_s + 4) - (len(self.track.key_pts) - 1))
            difference_ = difference
            while difference > 0:
                key_pts.append(self.track.key_pts[difference_ - difference])
                difference -= 1
            for i in range(5 - len(key_pts)):
                key_pts.append(self.track.key_pts[key_pt_idx_s + i])

        for stageidx in range(self.N):
            # Default to respecting obstacles
            obs_deactivate = False
            # deactivate prediction obstacle avoidance if in blocking-mode
            if (blocking) and (stageidx > 0 or obstacle[0] is None):
                obs_deactivate = True
            if obstacle[self.N-1].h == 0 and stageidx == self.N-1:
                obs_deactivate = True

            initial_guess.append(self.x_ws[stageidx])  # x
            initial_guess.append(self.u_ws[stageidx])  # u
            initial_guess.append(self.u_ws[stageidx - 1])  # u_prev

            stage_p = []
            stage_p.append(x_ref[stageidx])  # x_ref (for blocking maneuvers)
            stage_p.append(xref_scale)  # x_ref (for blocking maneuvers)
            stage_p.append(s0 if stageidx == self.N-1 else -100)
            stage_p.append(obstacle[stageidx].x_tran if not blocking else -100)
            stage_p.append(direction)
            stage_p.extend(key_pts[0])
            stage_p.extend(key_pts[1])
            stage_p.extend(key_pts[2])
            stage_p.extend(key_pts[3])
            stage_p.extend(key_pts[4])
            stage_p.extend([self.track.track_length])
            stage_p.extend([self.Qc, self.Ql, self.Q_theta, self.Q_xref, self.R_d, self.R_delta, self.half_width])
            stage_p.extend([
                obstacle[stageidx].xc,
                obstacle[stageidx].yc,
                obstacle[stageidx].s,
                obstacle[stageidx].psi,
                obstacle[stageidx].h,
                obstacle[stageidx].w,
                obs_deactivate  # deactivate obstacle by default
            ])

            parameters.append(stage_p)

            # 2 slack constraints, 4 cost variables
            if self.slack:
                initial_guess.append(np.zeros((2,)))
                parameters.extend(np.array([self.Q_cs, self.l_cs, self.Q_ts, self.l_ts]).reshape((4, -1)))

        parameters = np.concatenate(parameters)
        initial_guess = np.concatenate(initial_guess)

        # problem dictionary, arrays have to be flattened
        problem = dict()
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
                self.dynamics.f_d_rk4_forces(x_ws[-1], self.track.track_length, ca.transpose(
                    ca.horzcat(key_pts[0], key_pts[1], key_pts[2], key_pts[3], key_pts[4])),
                                             u_ws[-1])).squeeze()))  # TODO fix this
            u_ws = np.vstack((u_ws, u_ws[-1]))  # stack previous input
            self.set_warm_start(x_ws, u_ws)

            u = self.u_pred[0]

            self.dynamics.qu2prediction(self.state_input_prediction[id], self.x_pred, self.u_pred)


        else:
            info = {"success": False, "return_status": 'Solving Failed, exitflag = %d' % exitflag, "solve_time": None,
                    "info": solve_info}
            u = np.zeros(self.d)
            # self.state_input_prediction = VehiclePrediction()
            #
            # self.x_ws[1:, self.zvars.index('posx'):self.zvars.index('psi') + 1] = \
            #     track_lin_points[1:, :track_lin_points.shape[1] - 1]
            # self.x_ws[1:, self.zvars.index('vx')] = 0.2 * np.ones((self.N - 1,))
            # self.set_warm_start(self.x_ws, self.u_ws)

        control = VehicleActuation()
        self.dynamics.u2input(control, u)

        return control, info, u

    def _load_solver(self, solver_dir):
        self.solver = forcespro.nlp.Solver.from_directory(solver_dir)

    def _build_solver(self):
        def nonlinear_ineq(z, p):
            # Extract real-time parameters
            r = p[self.pvars.index('r')]

            # Extract states
            posx = z[self.zvars.index('posx')]
            posy = z[self.zvars.index('posy')]
            psi = z[self.zvars.index('psi')]
            x_tran = z[self.zvars.index('x_tran')]
            obs_x_tran = p[self.pvars.index('x_tran_obs')]
            s_obs = p[self.pvars.index('s')]
            direction = p[self.pvars.index('dir')]

            if self.slack:
                obstacle_slack = z[self.zvars.index('obs_slack')]
                track_slack = z[self.zvars.index('track_slack')]

            # Track (slacked) boundaries
            # To exclude vehicle boundaries: (r - self.widthcar)
            upper_track_bound = x_tran - track_slack - r
            lower_track_bound = x_tran + track_slack + r

            # Ellipsoidal obstacle
            x_ob = p[self.pvars.index('x_ob')]
            y_ob = p[self.pvars.index('y_ob')]
            psi_ob = p[self.pvars.index('psi_ob')]
            l_ob = p[self.pvars.index('l_ob')]
            w_ob = p[self.pvars.index('w_ob')]
            deactivate_ob = p[self.pvars.index('deactivate_ob')]


            s = ca.sin(psi_ob)
            c = ca.cos(psi_ob)

            # 4 Disks to approximate box car shape
            r_d = self.widthcar / 2
            dl = self.lencar * 0.9 / 3
            s_e = ca.sin(psi)
            c_e = ca.cos(psi)

            dx1 = x_ob - (posx - 3 * dl * c_e / 2)
            dx2 = x_ob - (posx - dl * c_e / 2)
            dx3 = x_ob - (posx + dl * c_e / 2)
            dx4 = x_ob - (posx + 3 * dl * c_e / 2)
            dy1 = y_ob - (posy - 3 * dl * s_e / 2)
            dy2 = y_ob - (posy - dl * s_e / 2)
            dy3 = y_ob - (posy + dl * s_e / 2)
            dy4 = y_ob - (posy + 3 * dl * s_e / 2)

            if self.slack:
                a = (l_ob / 1.5 + r_d) * 1.5 / (1 + obstacle_slack)
                b = (w_ob / 1.5 + r_d) * 1.5 / (1 + obstacle_slack)

            i1 = (c * dx1 - s * dy1) ** 2 * 1 / a ** 2 + (s * dx1 + c * dy1) ** 2 * 1 / b ** 2
            i2 = (c * dx2 - s * dy2) ** 2 * 1 / a ** 2 + (s * dx2 + c * dy2) ** 2 * 1 / b ** 2
            i3 = (c * dx3 - s * dy3) ** 2 * 1 / a ** 2 + (s * dx3 + c * dy3) ** 2 * 1 / b ** 2
            i4 = (c * dx4 - s * dy4) ** 2 * 1 / a ** 2 + (s * dx4 + c * dy4) ** 2 * 1 / b ** 2

            obsval1 = casadi.if_else(deactivate_ob == 1, 2, i1)
            obsval2 = casadi.if_else(deactivate_ob == 1, 2, i2)
            obsval3 = casadi.if_else(deactivate_ob == 1, 2, i3)
            obsval4 = casadi.if_else(deactivate_ob == 1, 2, i4)

            dec_val = ca.if_else(ca.fabs(s_obs-s) < self.lencar, direction*(x_tran-obs_x_tran), 1)
            decision = ca.if_else(obs_x_tran == -100, 1, dec_val)
            # Concatenate
            hval = ca.vertcat(upper_track_bound, lower_track_bound, obsval1, obsval2, obsval3, obsval4, decision)
            return hval

        def objective_cost(z, p):
            # Extract
            xref = p[self.pvars.index('xref')]
            xref_scale = p[self.pvars.index('xref_scale')]
            '''xt = p[self.pvars.index('xt1'):self.pvars.index('xt1') + 10]
            yt = p[self.pvars.index('yt1'):self.pvars.index('yt1') + 10]
            psit = p[self.pvars.index('psit1'):self.pvars.index('psit1') + 10]
            theta_hat = p[self.pvars.index('theta_hat1'):self.pvars.index('theta_hat1') + 10]'''
            kp1 = p[self.pvars.index('kp1_0'):self.pvars.index('kp1_0') + 6]
            kp2 = p[self.pvars.index('kp2_0'):self.pvars.index('kp2_0') + 6]
            kp3 = p[self.pvars.index('kp3_0'):self.pvars.index('kp3_0') + 6]
            kp4 = p[self.pvars.index('kp4_0'):self.pvars.index('kp4_0') + 6]
            kp5 = p[self.pvars.index('kp5_0'):self.pvars.index('kp5_0') + 6]
            t_l = p[self.pvars.index('track_length')]

            Qc = p[self.pvars.index('Qc')]
            Ql = p[self.pvars.index('Ql')]
            Q_theta = p[self.pvars.index('Q_theta')]
            R_d = p[self.pvars.index('R_d')]
            R_delta = p[self.pvars.index('R_delta')]
            Q_xref = p[self.pvars.index('Q_xref')]

            # Extract states
            v_long = z[self.zvars.index('vx')]
            posx = z[self.zvars.index('posx')]
            s = z[self.zvars.index('s')]
            s0 = p[self.pvars.index('s0')]
            e_psi = z[self.zvars.index('e_psi')]
            psi = z[self.zvars.index('psi')]
            posy = z[self.zvars.index('posy')]
            x_tran = z[self.zvars.index('x_tran')]

            if self.slack:
                obstacle_slack = z[self.zvars.index('obs_slack')]
                Q_cs = p[self.pvars.index('Q_cs')]
                l_cs = p[self.pvars.index('l_cs')]

                track_slack = z[self.zvars.index('track_slack')]
                Q_ts = p[self.pvars.index('Q_ts')]
                l_ts = p[self.pvars.index('l_ts')]

            # Obstacle vals
            Q_xref /= (1 + xref_scale)

            '''theta_hat_ = ca.pw_const(s, theta_hat[1:], theta_hat)
            xt_ = ca.pw_const(s, xt[1:], xt)
            yt_ = ca.pw_const(s, yt[1:], yt)
            psit_ = ca.pw_const(s, psit[1:], psit)'''
            (xt_hat, yt_hat, psit_) = self.track.get_centerline_xy_from_s_casadi(s, t_l, ca.transpose(
                ca.horzcat(kp1, kp2, kp3, kp4, kp5)))
            sin_psit = ca.sin(psit_)
            cos_psit = ca.cos(psit_)

            v_proj = ca.cos(psi - psit_) * v_long

            # Extract inputs
            u_a = z[self.zvars.index('u_a')]
            u_delta = z[self.zvars.index('u_delta')]

            # Approximate linearized contouring and lag error
            '''xt_hat = xt_ + cos_psit * (s - theta_hat_)
            yt_hat = yt_ + sin_psit * (s - theta_hat_)'''

            e_cont = -sin_psit * (xt_hat - posx) + cos_psit * (yt_hat - posy)
            e_lag = cos_psit * (xt_hat - posx) + sin_psit * (yt_hat - posy)

            cost = ca.bilin(Qc, e_cont, e_cont) + ca.bilin(Ql, e_lag, e_lag) + \
                   ca.bilin(R_d, u_a, u_a) + ca.bilin(R_delta, u_delta, u_delta)

            cost += ca.if_else(xref == -20, 0, ca.bilin(Q_xref, x_tran - xref, x_tran - xref))
            cost += ca.if_else(s0 == -100, 0, -Q_theta*(s-s0))

            if self.slack:
                cost += ca.power(obstacle_slack, 2) * Q_cs + obstacle_slack * l_cs
                cost += ca.power(track_slack, 2) * Q_ts + track_slack * l_ts

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
            u = z[self.n:self.n + self.d]
            kp1 = p[self.pvars.index('kp1_0'):self.pvars.index('kp1_0') + 6]
            kp2 = p[self.pvars.index('kp2_0'):self.pvars.index('kp2_0') + 6]
            kp3 = p[self.pvars.index('kp3_0'):self.pvars.index('kp3_0') + 6]
            kp4 = p[self.pvars.index('kp4_0'):self.pvars.index('kp4_0') + 6]
            kp5 = p[self.pvars.index('kp5_0'):self.pvars.index('kp5_0') + 6]
            t_l = p[self.pvars.index('track_length')]
            integrated = self.dynamics.f_d_rk4_forces(q, t_l, ca.transpose(ca.horzcat(kp1, kp2, kp3, kp4, kp5)), u)
            return ca.vertcat(integrated, u)

        # Forces model
        self.model = forcespro.nlp.SymbolicModel(self.N)

        # Number of parameters
        self.model.nvar = self.n + self.d + self.d  # Stage variables z = [x, u, u_prev]

        if self.slack:
            self.model.nvar += 2  # Stage variables z = [x, u, u_prev, 2 slacks]

        self.model.nh = self.d + 6 + 1 # of inequality constraints

        self.model.neq = self.n + self.d  # of equality constraints
        self.model.npar = len(self.pvars)  # number of real-time parameters (p)

        self.model.objective = lambda z, p: objective_cost(z, p)

        # dynamics only in state Variables
        stage_E_q = np.hstack((np.eye(self.n), np.zeros((self.n, self.d + self.d))))
        stage_E_u = np.hstack((np.zeros((self.d, self.n + self.d)), np.eye(self.d)))
        stage_E = np.vstack((stage_E_q, stage_E_u))

        # Equality constraints
        if self.slack:
            stage_E = np.hstack((stage_E, np.zeros((stage_E.shape[0], 1))))
            stage_E = np.hstack((stage_E, np.zeros((stage_E.shape[0], 1))))  # TODO

        self.model.E = stage_E
        self.model.eq = stage_equality_constraint

        # Inequality constraint bounds
        self.model.ineq = lambda z, p: ca.vertcat(nonlinear_ineq(z, p),  # ego, enemy states
                                                  stage_input_rate_constraint(z, p))

        # Nonlinear constraint bounds
        self.model.hu = np.concatenate((np.array([0.0, ca.inf, ca.inf, ca.inf, ca.inf, ca.inf, ca.inf]),
                                        self.input_rate_ub * self.dt))  # upper bound for nonlinear constraints, pg. 23 of FP manual
        self.model.hl = np.concatenate((np.array([-ca.inf, 0.0, 1.0, 1.0, 1.0, 1.0, 0]), self.input_rate_lb * self.dt))

        # Input box constraints
        if self.slack:
            self.model.ub = np.concatenate(
                (self.state_ub, self.input_ub, self.input_ub, np.array([0.5]),
                 np.array([ca.inf])))  # [x, u, u_prev, obs_slack, track_slack]
            self.model.lb = np.concatenate(
                (self.state_lb, self.input_lb, self.input_lb, np.array([0.0]),
                 np.array([0])))  # [x, u, u_prev, obs_slack, track_slack]
        else:
            self.model.ub = np.concatenate((self.state_ub, self.input_ub, self.input_ub))  # [x, u, u_prev]
            self.model.lb = np.concatenate((self.state_lb, self.input_lb, self.input_lb))  # [x, u, u_prev]

        # Put initial condition on all state variables x
        self.model.xinitidx = list(np.arange(self.n)) + list(
            np.arange(self.n + self.d, self.n + self.d + self.d))  # x, u_prev

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

    def aggressive_blocking_policy(self, ego_state: VehicleState, tv_state: VehicleState,
                                   tv_prediction: VehiclePrediction = None):
        """
        Aggressive Blocking Policy. Will try to match x_tran of tv_state at all costs.
        """
        if tv_state and tv_state.p.s < ego_state.p.s:
            blocking = True
            xt = tv_state.p.x_tran
            x_ref = np.sign(xt) * min(self.track.half_width, abs(float(xt)))
        else:
            # non-blocking mode
            x_ref = -20
            blocking = False
        return x_ref, blocking

    def only_right_blocking_policy(self, ego_state: VehicleState, tv_state: VehicleState,
                                   tv_prediction: VehiclePrediction = None):
        """
        Aggressive Blocking Policy. Will try to match x_tran of tv_state at all costs if TV is on the right
        """
        if tv_state and tv_state.p.s < ego_state.p.s:
            blocking = True
            if tv_state.p.x_tran < ego_state.p.x_tran:
                xt = tv_state.p.x_tran
                x_ref = np.sign(xt) * min(self.track.half_width, abs(float(xt)))
            else:
                x_ref = -20
        else:
            # non-blocking mode
            x_ref = -20
            blocking = False
        return x_ref, blocking

#!/usr/bin python3

import array
from typing import List

import numpy as np

from mpclab_common.models.dynamics_models import CasadiDynamicsModel
from mpclab_common.pytypes import VehicleState, VehicleActuation, VehiclePrediction

from mpclab_controllers.abstract_controller import AbstractController
from mpclab_controllers.utils.controllerTypes import LTVLMPCParams
from mpclab_controllers.utils.LTV_LMPC_utils import PredictiveModel, LMPC, MPCParams

class LTV_LMPC(AbstractController):
    def __init__(self, dynamics, track, control_params=LTVLMPCParams(), print_method=print):
        assert isinstance(dynamics, CasadiDynamicsModel)
        self.dynamics = dynamics
        self.track = track

        self.dt = control_params.dt
        self.n = control_params.n
        self.d = control_params.d

        self.N = control_params.N

        self.n_ss_pts = control_params.n_ss_pts
        self.n_ss_its = control_params.n_ss_its
        self.Q_slack = np.diag(control_params.Q_slack)

        self.model = PredictiveModel(self.n, self.d, self.track, self.n_ss_its)

        # Fx = np.vstack((np.eye(self.n), -np.eye(self.n)))
        # bx = np.vstack((control_params.state_ub.reshape((-1,1)), -control_params.state_lb.reshape((-1,1))))

        # Fx = np.array([[0., 0., 0., 0., 0., 1.],
        #                [0., 0., 0., 0., 0., -1.]])

        # bx = np.array([[self.track.half_width],   # max ey
        #                [self.track.half_width]]) # max ey
        # # bx = np.array([[self.track.half_width],   # max ey
        # #                [self.track.half_width]]) # max ey

        # Fu = np.vstack((np.eye(self.d), -np.eye(self.d)))
        # bu = np.vstack((np.flip(control_params.input_ub).reshape((-1,1)),
        #                 -np.flip(control_params.input_lb).reshape((-1,1))))

        Fx = np.array([[0., 0., 0., 0., 0., 1.],
                    [0., 0., 0., 0., 0., -1.]])

        bx = np.array([[self.track.half_width-0.1],   # max ey
                    [self.track.half_width-0.1]]), # max ey

        Fu = np.kron(np.eye(2), np.array([1, -1])).T
        bu = np.array([[0.436],   # -Min Steering
                    [0.436],   # Max Steering
                    [2.0],  # -Min Acceleration
                    [2.0]]) # Max Acceleration

        self.mpc_params = MPCParams(n=self.n, d=self.d, N=self.N,
                                    Q=np.diag(control_params.Q),
                                    R=np.diag(control_params.R),
                                    Qf=np.diag(control_params.Q_f),
                                    dR=control_params.R_d,
                                    Qslack=control_params.Q_lane,
                                    timeVarying=True,
                                    slacks=True,
                                    Fx=Fx, bx=bx, Fu=Fu, bu=bu)

        self.controller = None

        self.state_input_prediction = None
        self.safe_set = None
        self.initialized = False

        self.print_method = print_method

    def initialize(self, q_laps, u_laps, x_laps):
        for q, u in zip(q_laps, u_laps):
            self.model.addTrajectory(q, u)
        self.controller = LMPC(self.n_ss_pts, self.n_ss_its, self.Q_slack, self.mpc_params, self.model, self.dt, print_method=self.print_method)
        for q, u, x in zip(q_laps, u_laps, x_laps):
            self.controller.addTrajectory(q, u, x)
        self.initialized = True

    def set_warm_start(self, x_seq, u_seq):
        self.controller.xPred, self.controller.uPred = x_seq, u_seq
        self.controller.xLin, self.controller.uLin = x_seq, u_seq

    def set_terminal_set_sample_point(self, x, u):
        self.controller.zt = x
        self.controller.zt_u = u

    def solve(self):
        pass

    def step(self, vehicle_state: VehicleState, env_state=None):
        if not self.initialized:
            raise(RuntimeError('LMPC controller is not initialized'))

        q0, _ = self.state2qu(vehicle_state)
        self.controller.solve(q0)

        # In this LMPC formulation we have u = [u_steer, u_a]. This is opposite
        # to the rest of the mpclab/barc code base
        u = self.controller.uPred[0]

        # This method adds the state and input to the safe sets at the previous iteration.
        # The "s" component will be greater than the track length and the cost-to-go will be negative
        self.controller.addPoint(q0, u)

        self.dynamics.qu2state(vehicle_state, None, np.flip(u))

        if self.state_input_prediction is None:
            self.state_input_prediction = VehiclePrediction()
        if self.safe_set is None:
            self.safe_set = VehiclePrediction()
        self.dynamics.qu2prediction(self.state_input_prediction, self.controller.xPred, np.fliplr(self.controller.uPred))
        self.dynamics.qu2prediction(self.safe_set, self.controller.SS_PointSelectedTot.T, np.fliplr(self.controller.Succ_uSS_PointSelectedTot.T))
        self.state_input_prediction.t = vehicle_state.t
        self.safe_set.t = vehicle_state.t

        return

    def add_trajectory(self, q, u, x):
        self.controller.addTrajectory(q, u, x)
        self.model.addTrajectory(q, u)

    def get_prediction(self):
        return self.state_input_prediction

    def get_safe_set(self):
        return self.safe_set

    def state2qu(self, state: VehicleState):
        q = np.array([state.v.v_long, state.v.v_tran, state.w.w_psi, state.p.e_psi, state.p.s, state.p.x_tran])
        u = np.array([state.u.u_steer, state.u.u_a])
        return q, u

    def state2xu(self, state: VehicleState):
        self.track.local_to_global_typed(state)
        x = np.array([state.v.v_long, state.v.v_tran, state.w.w_psi, state.e.psi, state.x.x, state.x.y])
        u = np.array([state.u.u_steer, state.u.u_a])
        return x, u

if __name__ == "__main__":
    import pdb

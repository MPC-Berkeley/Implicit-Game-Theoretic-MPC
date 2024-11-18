#!/usr/bin/env python3

import numpy as np
import time

from mpclab_simulation.abstractSensor import abstractSensor
from mpclab_simulation.sim_types import ViveSimConfig

from mpclab_common.pytypes import VehicleState, Position, OrientationEuler, BodyAngularVelocity, BodyLinearVelocity, BodyLinearAcceleration
from mpclab_common.models.model_types import PoseVelMeasurement

import pdb

class ViveSimulator(abstractSensor):
    '''
    Class for creating and running a simulated Vive tracking system

    '''

    def __init__(self, params: ViveSimConfig = ViveSimConfig()):
        #limit on how many sigma of noise can be added (e.g. 1 will cap noise at +/- 1 standard deviation)
        #set to None for no bound
        self.n_bound    = params.n_bound

        #standard deviation of white noise added to true state
        self.x_std      = params.x_std
        self.y_std      = params.y_std
        self.z_std      = params.z_std
        self.roll_std   = params.roll_std
        self.pitch_std  = params.pitch_std
        self.yaw_std    = params.yaw_std
        self.v_long_std = params.v_long_std
        self.v_tran_std = params.v_tran_std
        self.v_vert_std = params.v_vert_std
        self.roll_dot_std = params.yaw_dot_std
        self.pitch_dot_std = params.yaw_dot_std
        self.yaw_dot_std = params.yaw_dot_std

        # Offset of vive tracker with respect to com of car
        self.tracker_offset_long = params.offset_long
        self.tracker_offset_tran = params.offset_tran
        self.tracker_offset_vert = params.offset_vert
        self.tracker_offset_yaw = params.offset_yaw

        # Location and orientation of the origin of the simulator global frame with respect to the vive global frame
        # We would get these values by measuring the position of the global origin using the Vive system
        self.vive_origin_x = params.origin_x
        self.vive_origin_y = params.origin_y
        self.vive_origin_z = params.origin_z
        self.vive_origin_yaw = params.origin_yaw
        # self.vive_to_glob_yaw_diff = np.pi - self.vive_origin_yaw
        # self.vive_to_glob_yaw_diff = np.pi/2
        self.vive_to_glob_yaw_diff = 0

        # Vive global frame positions, tracker body frame velocities
        self.vive_meas = PoseVelMeasurement()

        self.initialized = False

    def initialize(self, init_vehicle_state: VehicleState):
        self.initialized = True

    def step(self, vehicle_state: VehicleState):
        if not self.initialized:
            raise RuntimeError('Vive tracking simulator not initialized')

        sim_car_x = self.add_white_noise(vehicle_state.x.x, self.x_std, sigma_max=self.n_bound) if self.x_std is not None else vehicle_state.x.x
        sim_car_y = self.add_white_noise(vehicle_state.x.y, self.y_std, sigma_max=self.n_bound) if self.y_std is not None else vehicle_state.x.y
        sim_car_yaw = self.add_white_noise(vehicle_state.e.psi, self.yaw_std, sigma_max=self.n_bound) if self.yaw_std is not None else vehicle_state.e.psi
        sim_car_v_long = self.add_white_noise(vehicle_state.v.v_long, self.v_long_std, sigma_max=self.n_bound) if self.v_long_std is not None else vehicle_state.v.v_long
        sim_car_v_tran = self.add_white_noise(vehicle_state.v.v_tran, self.v_tran_std, sigma_max=self.n_bound) if self.v_tran_std is not None else vehicle_state.v.v_tran
        sim_car_yaw_dot = self.add_white_noise(vehicle_state.w.w_psi, self.yaw_dot_std, sigma_max=self.n_bound) if self.yaw_dot_std is not None else vehicle_state.w.w_psi

        # These states aren't simulated in planar dynamics but we can still add noise
        sim_car_z = self.add_white_noise(0.0, self.z_std, sigma_max=self.n_bound) if self.z_std is not None else 0.0
        sim_car_v_vert = self.add_white_noise(0.0, self.v_vert_std, sigma_max=self.n_bound) if self.v_vert_std is not None else 0.0
        # sim_car_roll = self.add_white_noise(np.pi/2, self.roll_std, sigma_max=self.n_bound) if self.roll_std is not None else np.pi/2
        sim_car_roll = self.add_white_noise(0.0, self.roll_std, sigma_max=self.n_bound) if self.roll_std is not None else 0.0
        sim_car_pitch = self.add_white_noise(0.0, self.pitch_std, sigma_max=self.n_bound) if self.pitch_std is not None else 0.0
        sim_car_roll_dot = self.add_white_noise(0.0, self.roll_dot_std, sigma_max=self.n_bound) if self.roll_dot_std is not None else 0.0
        sim_car_pitch_dot = self.add_white_noise(0.0, self.pitch_dot_std, sigma_max=self.n_bound) if self.pitch_dot_std is not None else 0.0

        # Pose of tracker in simulator global frame
        sim_tracker_x = sim_car_x + (self.tracker_offset_long*np.cos(sim_car_yaw) - self.tracker_offset_tran*np.sin(sim_car_yaw))
        sim_tracker_y = sim_car_y + (self.tracker_offset_long*np.sin(sim_car_yaw) + self.tracker_offset_tran*np.cos(sim_car_yaw))
        sim_tracker_z = sim_car_z + self.tracker_offset_vert
        sim_tracker_yaw = sim_car_yaw + self.tracker_offset_yaw

        x_bar = sim_tracker_x
        y_bar = sim_tracker_y
        theta_bar = np.arctan2(y_bar, x_bar)
        r = np.sqrt(x_bar**2 + y_bar**2)
        phi = self.vive_to_glob_yaw_diff + theta_bar

        # Pose of tracker in vive global frame
        vive_tracker_x = self.vive_origin_x + r*np.cos(phi)
        vive_tracker_y = self.vive_origin_y + r*np.sin(phi)
        vive_tracker_z = self.vive_origin_z + sim_tracker_z
        vive_tracker_yaw = np.mod(self.vive_origin_yaw + sim_tracker_yaw, 2*np.pi)
        if vive_tracker_yaw > np.pi:
            vive_tracker_yaw = vive_tracker_yaw - 2*np.pi # [-pi, pi]

        self.vive_meas.x = vive_tracker_x
        self.vive_meas.y = vive_tracker_y
        self.vive_meas.z = vive_tracker_z
        self.vive_meas.v_long = sim_car_v_long
        self.vive_meas.v_tran = sim_car_v_tran
        self.vive_meas.v_vert = sim_car_v_vert
        self.vive_meas.roll = sim_car_roll
        self.vive_meas.pitch = sim_car_pitch
        self.vive_meas.yaw = vive_tracker_yaw
        self.vive_meas.roll_dot = sim_car_roll_dot
        self.vive_meas.pitch_dot = sim_car_pitch_dot
        self.vive_meas.yaw_dot = sim_car_yaw_dot

        return {'pose': self.vive_meas}

if __name__ == '__main__':
    conf = ViveSimConfig(origin_x=1.0, origin_y=2.0, origin_yaw=0.0, offset_long=0.1, offset_tran=0.0, offset_yaw=0.0)
    sim = ViveSimulator(conf)
    t0 = time.time()

    init_state = VehicleState(x = Position(x = 0.5, y = 0.5), e = OrientationEuler(psi=np.pi/4), v = BodyLinearVelocity(v_long=0.0, v_tran=0.0), w = BodyAngularVelocity(w_psi=0.0))
    sim.initialize(init_state, 0.0)
    vive_out = sim.step(init_state)

    print(vive_out)

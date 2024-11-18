#!/usr/bin/env python3

import numpy as np
from scipy.spatial.transform import Rotation

import time

from mpclab_simulation.abstractSensor import abstractSensor
from mpclab_simulation.sim_types import IMUSimConfig

from mpclab_common.pytypes import VehicleState, Position
from mpclab_common.models.model_types import IMUMeasurement

import pdb

class IMUSimulator(abstractSensor):
    '''
    Class for creating and running a simulated IMU

    '''

    def __init__(self, params: IMUSimConfig = IMUSimConfig()):
        #limit on how many sigma of noise can be added (e.g. 1 will cap noise at +/- 1 standard deviation)
        #set to None for no bound
        self.n_bound    = params.n_bound

        #standard deviation of white noise added to true state
        self.roll_std   = params.roll_std
        self.pitch_std  = params.pitch_std
        self.yaw_std    = params.yaw_std
        self.roll_dot_std = params.yaw_dot_std
        self.pitch_dot_std = params.yaw_dot_std
        self.yaw_dot_std = params.yaw_dot_std
        self.a_long_std = params.a_long_std
        self.a_tran_std = params.a_tran_std
        self.a_vert_std = params.a_vert_std

        # Offset of imu with respect to com of car
        self.imu_offset_long = params.offset_long
        self.imu_offset_tran = params.offset_tran
        self.imu_offset_vert = params.offset_vert
        self.imu_offset_yaw = params.offset_yaw
        self.imu_offset_pitch = params.offset_pitch
        self.imu_offset_roll = params.offset_roll

        # IMU body frame measurements
        self.imu_meas = IMUMeasurement()

        self.initialized = True

    def initialize(self):
        pass

    def step(self, vehicle_state: VehicleState):

        sim_car_yaw = self.add_white_noise(vehicle_state.e.psi, self.yaw_std, sigma_max=self.n_bound) if self.yaw_std is not None else vehicle_state.e.psi
        sim_car_yaw_dot = self.add_white_noise(vehicle_state.w.w_psi, self.yaw_dot_std, sigma_max=self.n_bound) if self.yaw_dot_std is not None else vehicle_state.w.w_psi
        sim_car_a_long = self.add_white_noise(vehicle_state.a.a_long, self.a_long_std, sigma_max=self.n_bound) if self.a_long_std is not None else vehicle_state.a.a_long
        sim_car_a_tran = self.add_white_noise(vehicle_state.a.a_tran, self.a_tran_std, sigma_max=self.n_bound) if self.a_tran_std is not None else vehicle_state.a.a_tran

        # These states aren't simulated in planar dynamics but we can still add noise
        sim_car_a_vert = self.add_white_noise(0.0, self.a_vert_std, sigma_max=self.n_bound) if self.a_vert_std is not None else 0.0
        sim_car_roll = self.add_white_noise(0.0, self.roll_std, sigma_max=self.n_bound) if self.roll_std is not None else 0.0
        sim_car_pitch = self.add_white_noise(0.0, self.pitch_std, sigma_max=self.n_bound) if self.pitch_std is not None else 0.0
        sim_car_roll_dot = self.add_white_noise(0.0, self.roll_dot_std, sigma_max=self.n_bound) if self.roll_dot_std is not None else 0.0
        sim_car_pitch_dot = self.add_white_noise(0.0, self.pitch_dot_std, sigma_max=self.n_bound) if self.pitch_dot_std is not None else 0.0

        # IMU outputs in simulator global frame
        sim_imu_yaw = sim_car_yaw + self.imu_offset_yaw
        sim_imu_pitch= sim_car_pitch + self.imu_offset_pitch
        sim_imu_roll = sim_car_roll + self.imu_offset_roll

        ## TODO: modify angular velocities and acceleration based on angular offsets from car body frame
        sim_imu_yaw_dot    = sim_car_yaw_dot
        sim_imu_pitch_dot  = sim_car_pitch_dot
        sim_imu_roll_dot   = sim_car_roll_dot

        sim_imu_a_long     = sim_car_a_long
        sim_imu_a_tran     = sim_car_a_tran
        sim_imu_a_vert     = sim_car_a_vert

        if sim_imu_yaw > np.pi:
            sim_imu_yaw = sim_imu_yaw - 2*np.pi # [-pi, pi]

        rot = Rotation.from_euler('ZYX', [sim_imu_yaw, sim_imu_pitch, sim_imu_roll])
        quat = rot.as_quat()

        self.imu_meas.linear_acceleration.x = sim_imu_a_long
        self.imu_meas.linear_acceleration.y = sim_imu_a_tran
        self.imu_meas.linear_acceleration.z = sim_imu_a_vert
        self.imu_meas.angular_velocity.x = sim_imu_roll_dot
        self.imu_meas.angular_velocity.y = sim_imu_pitch_dot
        self.imu_meas.angular_velocity.z = sim_imu_yaw_dot
        self.imu_meas.orientation.x = quat[0]
        self.imu_meas.orientation.y = quat[1]
        self.imu_meas.orientation.z = quat[2]
        self.imu_meas.orientation.w = quat[3]

        return {'imu': self.imu_meas}

if __name__ == '__main__':
    conf = IMUSimConfig(offset_long=0.1, offset_tran=0.0, offset_yaw=0.0)
    sim = IMUSimulator(conf)
    t0 = time.time()

    init_state = VehicleState(x=Position(x = 0.5, y = 0.5), psi=np.pi/4, v_long=0.0, v_tran=0.0, psidot=0.0)
    imu_out = sim.step(init_state)

    print(imu_out)

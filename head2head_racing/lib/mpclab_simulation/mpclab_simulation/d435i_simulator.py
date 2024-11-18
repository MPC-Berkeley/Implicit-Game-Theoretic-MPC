#!/usr/bin/env python3

import numpy as np
from scipy.spatial.transform import Rotation
import time

from mpclab_simulation.abstractSensor import abstractSensor
from mpclab_simulation.sim_types import D435iSimConfig
from mpclab_simulation.t265_simulator import T265Simulator

from mpclab_common.pytypes import VehicleState
from mpclab_common.models.model_types import IMUMeasurement

import pdb


class D435iSimulator(abstractSensor):
    '''
    Class for creating and running a simulated accelerometer and gyroscope from the d435i RealSense camera
    '''

    def __init__(self, params: D435iSimConfig = D435iSimConfig()):
        #limit on how many sigma of noise can be added (e.g. 1 will cap noise at +/- 1 standard deviation)
        #set to None for no bound
        self.n_bound            = params.n_bound

        #standard deviation of white noise added to true state
        self.roll_dot_std       = params.yaw_dot_std
        self.pitch_dot_std      = params.yaw_dot_std
        self.yaw_dot_std        = params.yaw_dot_std
        self.a_long_std         = params.a_long_std
        self.a_tran_std         = params.a_tran_std
        self.a_vert_std         = params.a_vert_std

        # Offset of d435i with respect to com of car
        self.d435_offset_long   = params.offset_long
        self.d435_offset_tran   = params.offset_tran
        self.d435_offset_vert   = params.offset_vert
        self.d435_offset_yaw    = params.offset_yaw
        self.d435_offset_pitch  = params.offset_pitch
        self.d435_offset_roll   = params.offset_roll

        # d435i body frame IMU measurements
        self.d435_imu_meas = IMUMeasurement()

        self.initialized = False

    def initialize(self):
        pass

    def step(self, vehicle_state: VehicleState):
        # Pose of car in simulator global frame
        sim_car_yaw_dot     = self.add_white_noise(vehicle_state.w.w_psi, self.yaw_dot_std, sigma_max=self.n_bound) if self.yaw_dot_std is not None else vehicle_state.w.w_psi
        sim_car_a_long      = self.add_white_noise(vehicle_state.a.a_long, self.a_long_std, sigma_max=self.n_bound) if self.a_long_std is not None else vehicle_state.a.a_long
        sim_car_a_tran      = self.add_white_noise(vehicle_state.a.a_tran, self.a_tran_std, sigma_max=self.n_bound) if self.a_tran_std is not None else vehicle_state.a.a_tran

        # These states aren't simulated in planar dynamics but we can still add noise
        sim_car_a_vert      = self.add_white_noise(0.0, self.a_vert_std, sigma_max=self.n_bound) if self.a_vert_std is not None else 0.0
        sim_car_roll_dot    = self.add_white_noise(0.0, self.roll_dot_std, sigma_max=self.n_bound) if self.roll_dot_std is not None else 0.0
        sim_car_pitch_dot   = self.add_white_noise(0.0, self.pitch_dot_std, sigma_max=self.n_bound) if self.pitch_dot_std is not None else 0.0

        ## TODO: modify angular velocities and acceleration based on angular offsets from car body frame
        sim_d435_yaw_dot    = sim_car_yaw_dot
        sim_d435_pitch_dot  = sim_car_pitch_dot
        sim_d435_roll_dot   = sim_car_roll_dot

        sim_d435_a_long     = sim_car_a_long
        sim_d435_a_tran     = sim_car_a_tran
        sim_d435_a_vert     = sim_car_a_vert
        self.d435_imu_meas.linear_acceleration.x = sim_d435_a_long
        self.d435_imu_meas.linear_acceleration.y = sim_d435_a_tran
        self.d435_imu_meas.linear_acceleration.z = sim_d435_a_vert
        self.d435_imu_meas.angular_velocity.x = sim_d435_roll_dot
        self.d435_imu_meas.angular_velocity.y = sim_d435_pitch_dot
        self.d435_imu_meas.angular_velocity.z = sim_d435_yaw_dot

        return {'imu': self.d435_imu_meas}

if __name__ == '__main__':
    sim = T265Simulator()
    t0 = time.time()
    for i in range(10):
        print(sim.step((0.1,0.1)))
        print('\n')
    print('10 steps in %f seconds'%(time.time() - t0))

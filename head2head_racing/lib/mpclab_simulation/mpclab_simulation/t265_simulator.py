#!/usr/bin/env python3

import numpy as np
from scipy.spatial.transform import Rotation

import time

from mpclab_simulation.abstractSensor import abstractSensor
from mpclab_simulation.sim_types import T265SimConfig

from mpclab_common.pytypes import VehicleState
from mpclab_common.models.model_types import PoseVelMeasurement, IMUMeasurement

import pdb


class T265Simulator(abstractSensor):
    '''
    Class for creating and running a simulated RealSense T265 camera with odometry and IMU outputs
    '''

    def __init__(self, params: T265SimConfig = T265SimConfig()):
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
        self.a_long_std = params.a_long_std
        self.a_tran_std = params.a_tran_std
        self.a_vert_std = params.a_vert_std

        # Offset of t265 with respect to com of car
        self.t265_offset_long = params.offset_long
        self.t265_offset_tran = params.offset_tran
        self.t265_offset_vert = params.offset_vert
        self.t265_offset_yaw = params.offset_yaw

        # Pose of t265 frame with respect to global frame. To be set on initialization
        self.t265_origin_x = None
        self.t265_origin_y = None
        self.t265_origin_z = None
        self.t265_origin_yaw = None

        self.heading_drift_rate = params.heading_drift_rate
        self.dt = params.dt

        # t265 global frame positions, t265 body frame velocities
        self.t265_pose_meas = PoseVelMeasurement()
        # t265 body frame IMU measurements
        self.t265_imu_meas = IMUMeasurement()

        self.initialized = False

    def initialize(self, init_vehicle_state: VehicleState):
        # Initial pose of the car in simulator global frame
        car_init_x = init_vehicle_state.x.x
        car_init_y = init_vehicle_state.x.y
        car_init_z = 0.0
        car_init_yaw = init_vehicle_state.e.psi

        # Location and orientation of the origin of the t265 global frame w.r.t. the simulator global frame
        # The t265 sets its origin once powered on
        self.t265_origin_x = car_init_x + (self.t265_offset_long*np.cos(car_init_yaw) - self.t265_offset_tran*np.sin(car_init_yaw))
        self.t265_origin_y = car_init_y + (self.t265_offset_long*np.sin(car_init_yaw) + self.t265_offset_tran*np.cos(car_init_yaw))
        self.t265_origin_z = car_init_z + self.t265_offset_vert
        self.t265_origin_yaw = car_init_yaw + self.t265_offset_yaw

        self.initialized = True

    def step(self, vehicle_state: VehicleState):
        if not self.initialized:
            raise RuntimeError('T265 simulator not initialized')

        if self.heading_drift_rate is not None:
            self.t265_origin_yaw += self.heading_drift_rate*self.dt

        # Pose of car in simulator global frame
        sim_car_x = self.add_white_noise(vehicle_state.x.x, self.x_std, sigma_max=self.n_bound) if self.x_std is not None else vehicle_state.x.x
        sim_car_y = self.add_white_noise(vehicle_state.x.y, self.y_std, sigma_max=self.n_bound) if self.y_std is not None else vehicle_state.x.y
        sim_car_yaw = self.add_white_noise(vehicle_state.e.psi, self.yaw_std, sigma_max=self.n_bound) if self.yaw_std is not None else vehicle_state.e.psi
        sim_car_v_long = self.add_white_noise(vehicle_state.v.v_long, self.v_long_std, sigma_max=self.n_bound) if self.v_long_std is not None else vehicle_state.v.v_long
        sim_car_v_tran = self.add_white_noise(vehicle_state.v.v_tran, self.v_tran_std, sigma_max=self.n_bound) if self.v_tran_std is not None else vehicle_state.v.v_tran
        sim_car_yaw_dot = self.add_white_noise(vehicle_state.w.w_psi, self.yaw_dot_std, sigma_max=self.n_bound) if self.yaw_dot_std is not None else vehicle_state.w.w_psi
        sim_car_a_long = self.add_white_noise(vehicle_state.a.a_long, self.a_long_std, sigma_max=self.n_bound) if self.a_long_std is not None else vehicle_state.a.a_long
        sim_car_a_tran = self.add_white_noise(vehicle_state.a.a_tran, self.a_tran_std, sigma_max=self.n_bound) if self.a_tran_std is not None else vehicle_state.a.a_tran

        # These states aren't simulated in planar dynamics but we can still add noise
        sim_car_z = self.add_white_noise(0.0, self.z_std, sigma_max=self.n_bound) if self.z_std is not None else 0.0
        sim_car_v_vert = self.add_white_noise(0.0, self.v_vert_std, sigma_max=self.n_bound) if self.v_vert_std is not None else 0.0
        sim_car_a_vert = self.add_white_noise(0.0, self.a_vert_std, sigma_max=self.n_bound) if self.a_vert_std is not None else 0.0
        sim_car_roll = self.add_white_noise(0.0, self.roll_std, sigma_max=self.n_bound) if self.roll_std is not None else 0.0
        sim_car_pitch = self.add_white_noise(0.0, self.pitch_std, sigma_max=self.n_bound) if self.pitch_std is not None else 0.0
        sim_car_roll_dot = self.add_white_noise(0.0, self.roll_dot_std, sigma_max=self.n_bound) if self.roll_dot_std is not None else 0.0
        sim_car_pitch_dot = self.add_white_noise(0.0, self.pitch_dot_std, sigma_max=self.n_bound) if self.pitch_dot_std is not None else 0.0

        # Pose of tracker in simulator global frame
        sim_t265_x = sim_car_x + (self.t265_offset_long*np.cos(sim_car_yaw) - self.t265_offset_tran*np.sin(sim_car_yaw))
        sim_t265_y = sim_car_y + (self.t265_offset_long*np.sin(sim_car_yaw) + self.t265_offset_tran*np.cos(sim_car_yaw))
        sim_t265_z = sim_car_z + self.t265_offset_vert
        sim_t265_yaw = sim_car_yaw + self.t265_offset_yaw

        x_bar = sim_t265_x - self.t265_origin_x
        y_bar = sim_t265_y - self.t265_origin_y

        theta_bar = np.arctan2(y_bar, x_bar)
        r = np.sqrt(x_bar**2 + y_bar**2)
        phi = theta_bar - self.t265_origin_yaw

        # Pose of tracker in t265 global frame
        cam_t265_x = r*np.cos(phi)
        cam_t265_y = r*np.sin(phi)
        cam_t265_yaw = sim_t265_yaw - self.t265_origin_yaw

        self.t265_pose_meas.x = cam_t265_x
        self.t265_pose_meas.y = cam_t265_y
        self.t265_pose_meas.z = sim_t265_z
        self.t265_pose_meas.v_long = sim_car_v_long
        self.t265_pose_meas.v_tran = sim_car_v_tran
        self.t265_pose_meas.v_vert = sim_car_v_vert
        self.t265_pose_meas.roll = sim_car_roll
        self.t265_pose_meas.pitch = sim_car_pitch
        self.t265_pose_meas.yaw = cam_t265_yaw
        self.t265_pose_meas.roll_dot = sim_car_roll_dot
        self.t265_pose_meas.pitch_dot = sim_car_pitch_dot
        self.t265_pose_meas.yaw_dot = sim_car_yaw_dot

        rot = Rotation.from_euler('ZYX', [cam_t265_yaw, sim_car_pitch, sim_car_roll])
        quat = rot.as_quat()

        self.t265_imu_meas.linear_acceleration.x = sim_car_a_long
        self.t265_imu_meas.linear_acceleration.y = sim_car_a_tran
        self.t265_imu_meas.linear_acceleration.z = sim_car_a_vert
        self.t265_imu_meas.angular_velocity.x = sim_car_roll_dot
        self.t265_imu_meas.angular_velocity.y = sim_car_pitch_dot
        self.t265_imu_meas.angular_velocity.z = sim_car_yaw_dot
        self.t265_imu_meas.orientation.x = quat[0]
        self.t265_imu_meas.orientation.y = quat[1]
        self.t265_imu_meas.orientation.z = quat[2]
        self.t265_imu_meas.orientation.w = quat[3]

        return {'pose': self.t265_pose_meas, 'imu': self.t265_imu_meas}

if __name__ == '__main__':
    sim = T265Simulator()
    t0 = time.time()
    for i in range(10):
        print(sim.step((0.1,0.1)))
        print('\n')
    print('10 steps in %f seconds'%(time.time() - t0))

#!/usr/bin/env python3

import numpy as np
from scipy.spatial.transform import Rotation

from mpclab_simulation.abstractSensor import abstractSensor
from mpclab_simulation.sim_types import OptiTrackSimConfig

from mpclab_common.pytypes import VehicleState, Position, OrientationEuler, BodyAngularVelocity, BodyLinearVelocity, BodyLinearAcceleration
from mpclab_common.models.model_types import PoseVelMeasurement

import time
import pdb

class OptiTrackSimulator(abstractSensor):
    '''
    Class for creating and running a simulated OptiTrack system
    '''

    def __init__(self, params: OptiTrackSimConfig = OptiTrackSimConfig()):
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

        # Offset of optitrack rigid body frame with respect to com aligned body frame of car
        self.cf2rf_p = np.array([params.offset_long, params.offset_tran, params.offset_vert])

        # Transformation from car body frame to Optitrack rigid body frame
        self.cf2rf_R = Rotation.from_euler('ZYX', [params.offset_yaw, params.offset_pitch, params.offset_roll])

        # Location of the origin of the track frame with respect to the OptiTrack frame
        self.go2to_p = np.array([params.origin_x, params.origin_y, params.origin_z])

        # Transformation from OptiTrack global origin frame to track origin frame
        self.go2to_R = Rotation.from_euler('ZYX', [params.origin_yaw, params.origin_pitch, params.origin_roll])

        # Vive global frame positions, tracker body frame velocities
        self.meas = PoseVelMeasurement()

        self.initialized = True

    def initialize(self, init_vehicle_state: VehicleState):
        self.initialized = True

    def step(self, vehicle_state: VehicleState):
        if not self.initialized:
            raise RuntimeError('OptiTrack simulator not initialized')

        x = self.add_white_noise(vehicle_state.x.x, self.x_std, sigma_max=self.n_bound) if self.x_std is not None else vehicle_state.x.x
        y = self.add_white_noise(vehicle_state.x.y, self.y_std, sigma_max=self.n_bound) if self.y_std is not None else vehicle_state.x.y
        z = self.add_white_noise(vehicle_state.x.z, self.z_std, sigma_max=self.n_bound) if self.z_std is not None else vehicle_state.x.z
        cf_p_to = np.array([x, y, z])

        roll    = self.add_white_noise(vehicle_state.e.phi, self.roll_std, sigma_max=self.n_bound) if self.roll_std is not None else vehicle_state.e.phi
        pitch   = self.add_white_noise(vehicle_state.e.theta, self.pitch_std, sigma_max=self.n_bound) if self.pitch_std is not None else vehicle_state.e.theta      
        yaw     = self.add_white_noise(vehicle_state.e.psi, self.yaw_std, sigma_max=self.n_bound) if self.yaw_std is not None else vehicle_state.e.psi
        cf_e_to = np.array([yaw, pitch, roll])

        v_long  = self.add_white_noise(vehicle_state.v.v_long, self.v_long_std, sigma_max=self.n_bound) if self.v_long_std is not None else vehicle_state.v.v_long
        v_tran  = self.add_white_noise(vehicle_state.v.v_tran, self.v_tran_std, sigma_max=self.n_bound) if self.v_tran_std is not None else vehicle_state.v.v_tran
        v_vert  = self.add_white_noise(vehicle_state.v.v_n, self.v_vert_std, sigma_max=self.n_bound) if self.v_vert_std is not None else vehicle_state.v.v_n
        cf_v    = np.array([v_long, v_tran, v_vert])

        roll_dot    = self.add_white_noise(vehicle_state.w.w_phi, self.roll_dot_std, sigma_max=self.n_bound) if self.roll_dot_std is not None else vehicle_state.w.w_phi
        pitch_dot   = self.add_white_noise(vehicle_state.w.w_theta, self.pitch_dot_std, sigma_max=self.n_bound) if self.pitch_dot_std is not None else vehicle_state.w.w_theta
        yaw_dot     = self.add_white_noise(vehicle_state.w.w_psi, self.yaw_dot_std, sigma_max=self.n_bound) if self.yaw_dot_std is not None else vehicle_state.w.w_psi
        cf_w        = np.array([roll_dot, pitch_dot, yaw_dot])

        # Angular velocities in optitrack rigid body frame
        rf_w = self.cf2rf_R.inv().apply(cf_w)

        # Linear velocities in optitrack rigid body frame
        rf_v = self.cf2rf_R.inv().apply(cf_v + np.cross(cf_w, self.cf2rf_p))

        # Attitude of car frame in OptiTrack global origin frame
        cf_e_go = cf_e_to + self.go2to_R.as_euler('ZYX')

        # Transformation from global origin frame to car body frame
        go2cf_R = Rotation.from_euler('ZYX', cf_e_go)

        # Attitude of OptiTrack rigid body frame
        rf_e_go = cf_e_go + self.cf2rf_R.as_euler('ZYX')

        # Position of car frame in the OptiTrack global frame
        cf_p_go = self.go2to_R.apply(cf_p_to) + self.go2to_p

        # Position of OptiTrack rigid body frame in the OptiTrack global frame
        rf_p_go = cf_p_go + go2cf_R.apply(self.cf2rf_p)

        self.meas.x         = rf_p_go[0]
        self.meas.y         = rf_p_go[1]
        self.meas.z         = rf_p_go[2]

        self.meas.v_long    = rf_v[0]
        self.meas.v_tran    = rf_v[1]
        self.meas.v_vert    = rf_v[2]

        self.meas.yaw       = rf_e_go[0]
        self.meas.roll      = rf_e_go[1]
        self.meas.pitch     = rf_e_go[2]

        self.meas.roll_dot  = rf_w[0]
        self.meas.pitch_dot = rf_w[1]
        self.meas.yaw_dot   = rf_w[2]

        return {'pose': self.meas}

if __name__ == '__main__':
    conf = OptiTrackSimConfig(origin_x=1.0, 
                                origin_y=2.0, 
                                origin_yaw=0.0, 
                                offset_long=0.2, 
                                offset_tran=0.1, 
                                offset_yaw=np.pi/6)
    sim = OptiTrackSimulator(conf)

    state = VehicleState(x=Position(x = 1.0, y = 0.5), 
                            e=OrientationEuler(psi=np.pi/4), 
                            v=BodyLinearVelocity(v_long=1.0, v_tran=0.0), 
                            w=BodyAngularVelocity(w_psi=0.1))
    meas = sim.step(state)

    print(meas)

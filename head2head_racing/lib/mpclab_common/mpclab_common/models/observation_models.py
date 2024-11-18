#!/usr/bin/env python3

import casadi as ca
import numpy as np
from scipy.linalg import sqrtm

import copy

from mpclab_common.pytypes import VehicleState
from mpclab_common.models.abstract_model import AbstractModel
from mpclab_common.models.model_types import ObserverConfig, PoseVelMeasurement
from mpclab_common.track import get_track

class CasadiObservationModel(AbstractModel):
    '''
    Base class for observation models that use casadi for their models.
    Implements common functions for linearizing models, integrating models, etc...
    '''
    def __init__(self, model_config, track=None):
        super().__init__(model_config)

        # Load track by name first, then by track object passed into model init
        if model_config.track_name is not None:
            self.track = get_track(model_config.track_name)
        else:
            self.track = track

    def precompute_model(self):
        '''
        wraps up model initialization
        require the following fields to be initialized:
        self.sym_q: ca.SX with elements of state vector q
        self.sym_u: ca.SX with elements of control vector u
        self.sym_z: ca.SX with observed quantities (z = h(q,u))
        '''
        meas_inputs = [self.sym_q, self.sym_u]
        if self.model_config.noise:
            meas_inputs += [self.sym_n]

        # Measurement function
        self.h = ca.Function('h', meas_inputs, [self.sym_z], self.options('h'))

        # symbolic jacobians
        self.sym_H = ca.jacobian(self.sym_z, self.sym_q)
        self.sym_G = ca.jacobian(self.sym_z, self.sym_u)
        self.sym_D = self.sym_z

        #numpy lambda functions for linearized continuous time system
        self.fH = ca.Function('fH', meas_inputs, [self.sym_H], self.options('fH'))
        self.fG = ca.Function('fG', meas_inputs, [self.sym_G], self.options('fG'))
        self.fD = ca.Function('fD', meas_inputs, [self.sym_D], self.options('fD'))

        if self.model_config.noise:
            self.sym_N = ca.jacobian(self.sym_z, self.sym_n) # zeros if no noise is added
            self.fN = ca.Function('fN', meas_inputs, [self.sym_N], self.options('fN'))

        if self.code_gen and not self.jit:
            so_fns = [self.h, self.fH, self.fG, self.fD]
            if self.model_config.noise:
                so_fns += [self.fN]
            self.install_dir = self.build_shared_object(so_fns)

        return

    def step(self, vehicle_state: VehicleState, measurement_noise: np.ndarray = None) -> np.ndarray:
        '''
        Make an observation based on current vehicle state
        '''
        q, u = self.state2qu(vehicle_state)
        if self.model_config.noise:
            if measurement_noise is None:
                measurement_noise = np.zeros(self.n_n)
            z = self.h(q, u, measurement_noise).toarray().squeeze()
        else:
            z = self.h(q, u).toarray().squeeze()
        return z

class CasadiDynamicBicycleFullStateObserver(CasadiObservationModel):
    '''
    Full state observer for the dynamic bicycle model with the option to include
    additive Gaussian measurement noise
    '''
    def __init__(self, model_config=ObserverConfig(), track=None):
        super().__init__(model_config, track=track)
        self.noise = model_config.noise
        if self.noise:
            if model_config.noise_cov is None:
                raise RuntimeError('Noise covariance matrix not provided to DynamicBicycleFullStateObserver')
            noise_cov = np.array(model_config.noise_cov)
            self.noise_cov = np.diag(noise_cov) if noise_cov.ndim == 1 else noise_cov

        self.n_z = 6
        self.n_q = 6
        self.n_u = 2
        self.n_n = 6

        # Symbolic state variables
        self.sym_vx     = ca.SX.sym('v_long') # body frame vx, vy (vx>0 points in direction car points, vy>0 points to left hand side)
        self.sym_vy     = ca.SX.sym('v_tran')
        self.sym_psidot = ca.SX.sym('psidot')
        self.sym_x      = ca.SX.sym('x')
        self.sym_y      = ca.SX.sym('y')
        self.sym_psi    = ca.SX.sym('psi')

        # Symbolic input variables
        self.sym_u_s    = ca.SX.sym('gamma')
        self.sym_u_a    = ca.SX.sym('a')

        # Symbolic variables for measurement noise components for each state
        self.sym_n_vx     = ca.SX.sym('n_v_long') # body frame vx, vy (vx>0 points in direction car points, vy>0 points to left hand side)
        self.sym_n_vy     = ca.SX.sym('n_v_tran')
        self.sym_n_psidot = ca.SX.sym('n_psidot')
        self.sym_n_x      = ca.SX.sym('n_x')
        self.sym_n_y      = ca.SX.sym('n_y')
        self.sym_n_psi    = ca.SX.sym('n_psi')

        self.sym_q = ca.vertcat(self.sym_vx,  self.sym_vy,  self.sym_psidot,  self.sym_x,  self.sym_y,  self.sym_psi)
        self.sym_u = ca.vertcat(self.sym_u_a, self.sym_u_s)
        self.sym_n = ca.vertcat(self.sym_n_vx,  self.sym_n_vy,  self.sym_n_psidot,  self.sym_n_x,  self.sym_n_y,  self.sym_n_psi)

        if self.noise:
            self.sym_z = ca.vertcat(self.sym_vx,  self.sym_vy,  self.sym_psidot,  self.sym_x,  self.sym_y,  self.sym_psi) \
                            + ca.mtimes(np.linalg.cholesky(self.noise_cov), self.sym_n)
        else:
            self.sym_z = ca.vertcat(self.sym_vx,  self.sym_vy,  self.sym_psidot,  self.sym_x,  self.sym_y,  self.sym_psi)

        self.precompute_model()
        return

    def state2qu(self, state: VehicleState):
        q = np.array([state.v.v_long, state.v.v_tran, state.w.w_psi, state.x.x, state.x.y, state.e.psi])
        u = np.array([state.u.u_a, state.u.u_steer])
        return q,u

    def meas2z(self, meas: PoseVelMeasurement):
        z = np.array([meas.v_long, meas.v_tran, meas.yaw_dot, meas.x, meas.y, meas.yaw])
        return z

class CasadiDynamicBicycleCLFullStateObserver(CasadiObservationModel):
    '''
    Full state observer for the dynamic bicycle model with the option to include
    additive Gaussian measurement noise
    '''
    def __init__(self, model_config=ObserverConfig(), track=None):
        super().__init__(model_config, track=track)
        self.noise = model_config.noise
        if self.noise:
            if model_config.noise_cov is None:
                raise RuntimeError('Noise covariance matrix not provided to DynamicBicycleFullStateObserver')
            noise_cov = np.array(model_config.noise_cov)
            self.noise_cov = np.diag(noise_cov) if noise_cov.ndim == 1 else noise_cov

        self.n_z = 6
        self.n_q = 6
        self.n_u = 2
        self.n_n = 6

        self.last_s = None
        self.lap_no = 0

        # Symbolic state variables
        self.sym_vx     = ca.SX.sym('v_long') # body frame vx, vy (vx>0 points in direction car points, vy>0 points to left hand side)
        self.sym_vy     = ca.SX.sym('v_tran')
        self.sym_psidot = ca.SX.sym('psidot')
        self.sym_epsi   = ca.SX.sym('epsi')
        self.sym_s      = ca.SX.sym('s')
        self.sym_xtran  = ca.SX.sym('xtran')

        # Symbolic input variables
        self.sym_u_s    = ca.SX.sym('gamma')
        self.sym_u_a    = ca.SX.sym('a')

        # Symbolic variables for measurement noise components for each state
        self.sym_n_vx     = ca.SX.sym('n_v_long') # body frame vx, vy (vx>0 points in direction car points, vy>0 points to left hand side)
        self.sym_n_vy     = ca.SX.sym('n_v_tran')
        self.sym_n_psidot = ca.SX.sym('n_psidot')
        self.sym_n_epsi      = ca.SX.sym('n_epsi')
        self.sym_n_s      = ca.SX.sym('n_s')
        self.sym_n_xtran    = ca.SX.sym('n_xtran')

        self.sym_q = ca.vertcat(self.sym_vx,  self.sym_vy,  self.sym_psidot,  self.sym_epsi,  self.sym_s,  self.sym_xtran)
        self.sym_u = ca.vertcat(self.sym_u_a, self.sym_u_s)
        self.sym_n = ca.vertcat(self.sym_n_vx,  self.sym_n_vy,  self.sym_n_psidot,  self.sym_n_epsi,  self.sym_n_s,  self.sym_n_xtran)

        if self.noise:
            self.sym_z = ca.vertcat(self.sym_vx,  self.sym_vy,  self.sym_psidot,  self.sym_epsi,  self.sym_s,  self.sym_xtran) \
                            + ca.mtimes(np.linalg.cholesky(self.noise_cov), self.sym_n)
        else:
            self.sym_z = ca.vertcat(self.sym_vx,  self.sym_vy,  self.sym_psidot,  self.sym_epsi,  self.sym_s,  self.sym_xtran)

        self.precompute_model()
        return

    def state2qu(self, state: VehicleState):
        q = np.array([state.v.v_long, state.v.v_tran, state.w.w_psi, state.p.e_psi, state.p.s, state.p.x_tran])
        u = np.array([state.u.u_a, state.u.u_steer])
        return q,u

    def meas2z(self, meas: PoseVelMeasurement, lap_no: int = None):
        s, x_tran, e_psi = self.track.global_to_local([meas.x, meas.y, meas.yaw])
        if self.last_s is None:
            self.last_s = copy.copy(s)

        if self.last_s > self.track.track_length * 0.8 and s < self.track.track_length * 0.2:
            self.lap_no += 1
        elif s > self.track.track_length * 0.8 and self.last_s < self.track.track_length * 0.2:
            self.lap_no -= 1
        self.last_s = copy.copy(s)

        s += self.lap_no*self.track.track_length
        z = np.array([meas.v_long, meas.v_tran, meas.yaw_dot, e_psi, s, x_tran])
        return z

class CasadiKinematicBicycleFullStateObserver(CasadiObservationModel):
    '''
    Full state observer for the kinematic bicycle model with the option to include
    additive Gaussian measurement noise
    '''
    def __init__(self, model_config=ObserverConfig(), track=None):
        super().__init__(model_config, track=track)
        self.noise = model_config.noise
        if self.noise:
            if model_config.noise_cov is None:
                raise RuntimeError('Noise covariance matrix not provided to DynamicBicycleFullStateObserver')
            noise_cov = np.array(model_config.noise_cov)
            self.noise_cov = np.diag(noise_cov) if noise_cov.ndim == 1 else noise_cov

        self.n_z = 4
        self.n_q = 4
        self.n_u = 2
        self.n_n = 4

        # Symbolic state variables
        self.sym_x      = ca.SX.sym('x')
        self.sym_y      = ca.SX.sym('y')
        self.sym_vx     = ca.SX.sym('v_long') # body frame vx, vy (vx>0 points in direction car points, vy>0 points to left hand side)
        self.sym_psi    = ca.SX.sym('psi')

        # Symbolic input variables
        self.sym_u_s    = ca.SX.sym('gamma')
        self.sym_u_a    = ca.SX.sym('a')

        # Symbolic variables for measurement noise components for each state
        self.sym_n_x      = ca.SX.sym('n_x')
        self.sym_n_y      = ca.SX.sym('n_y')
        self.sym_n_vx     = ca.SX.sym('n_v_long') # body frame vx, vy (vx>0 points in direction car points, vy>0 points to left hand side)
        self.sym_n_psi    = ca.SX.sym('n_psi')

        self.sym_q = ca.vertcat(self.sym_x,  self.sym_y, self.sym_vx, self.sym_psi)
        self.sym_u = ca.vertcat(self.sym_u_a, self.sym_u_s)
        self.sym_n = ca.vertcat(self.sym_n_x,  self.sym_n_y, self.sym_n_vx, self.sym_n_psi)

        if self.noise:
            self.sym_z = ca.vertcat(self.sym_x, self.sym_y, self.sym_vx, self.sym_psi) \
                            + ca.mtimes(np.linalg.cholesky(self.noise_cov), self.sym_n)
        else:
            self.sym_z = ca.vertcat(self.sym_x, self.sym_y, self.sym_vx, self.sym_psi)

        self.precompute_model()
        return

    def state2qu(self, state: VehicleState):
        q = np.array([state.x.x, state.x.y, state.v.v_long, state.e.psi])
        u = np.array([state.u.u_a, state.u.u_steer])
        return q, u

    def meas2z(self, meas: PoseVelMeasurement):
        z = np.array([meas.x, meas.y, meas.v_long, meas.yaw])
        return z

class CasadiKinematicBicycleCLFullStateObserver(CasadiObservationModel):
    '''
    Full state observer for the kinematic bicycle model with the option to include
    additive Gaussian measurement noise
    '''
    def __init__(self, model_config=ObserverConfig(), track=None):
        super().__init__(model_config, track=track)
        self.noise = model_config.noise
        if self.noise:
            if model_config.noise_cov is None:
                raise RuntimeError('Noise covariance matrix not provided to DynamicBicycleFullStateObserver')
            noise_cov = np.array(model_config.noise_cov)
            self.noise_cov = np.diag(noise_cov) if noise_cov.ndim == 1 else noise_cov

        self.n_z = 4
        self.n_q = 4
        self.n_u = 2
        self.n_n = 4

        self.last_s = None
        self.lap_no = 0

        # Symbolic state variables
        self.sym_vx     = ca.SX.sym('v_long') # body frame vx, vy (vx>0 points in direction car points, vy>0 points to left hand side)
        self.sym_epsi   = ca.SX.sym('epsi')
        self.sym_s      = ca.SX.sym('s')
        self.sym_xtran  = ca.SX.sym('xtran')

        # Symbolic input variables
        self.sym_u_s    = ca.SX.sym('gamma')
        self.sym_u_a    = ca.SX.sym('a')

        # Symbolic variables for measurement noise components for each state
        self.sym_n_vx     = ca.SX.sym('n_v_long') # body frame vx, vy (vx>0 points in direction car points, vy>0 points to left hand side)
        self.sym_n_epsi   = ca.SX.sym('n_epsi')
        self.sym_n_s      = ca.SX.sym('n_s')
        self.sym_n_xtran  = ca.SX.sym('n_xtran')

        self.sym_q = ca.vertcat(self.sym_vx,  self.sym_epsi,  self.sym_s,  self.sym_xtran)
        self.sym_u = ca.vertcat(self.sym_u_a, self.sym_u_s)
        self.sym_n = ca.vertcat(self.sym_n_vx,  self.sym_n_epsi,  self.sym_n_s,  self.sym_n_xtran)

        if self.noise:
            self.sym_z = ca.vertcat(self.sym_vx,  self.sym_epsi,  self.sym_s,  self.sym_xtran) \
                            + ca.mtimes(np.linalg.cholesky(self.noise_cov), self.sym_n)
        else:
            self.sym_z = ca.vertcat(self.sym_vx,  self.sym_epsi,  self.sym_s,  self.sym_xtran)

        self.precompute_model()
        return

    def state2qu(self, state: VehicleState):
        q = np.array([state.v.v_long, state.p.ths, state.p.s, state.p.y])
        u = np.array([state.u.u_a, state.u.u_steer])
        return q,u

    def meas2z(self, meas: PoseVelMeasurement, lap_no: int = None):
        s, x_tran, e_psi = self.track.global_to_local([meas.x, meas.y, meas.yaw])
        if self.last_s is None:
            self.last_s = copy.copy(s)

        if self.last_s > self.track.track_length * 0.8 and s < self.track.track_length * 0.2:
            self.lap_no += 1
        elif s > self.track.track_length * 0.8 and self.last_s < self.track.track_length * 0.2:
            self.lap_no -= 1
        self.last_s = copy.copy(s)

        s += self.lap_no*self.track.track_length
        z = np.array([meas.v_long, e_psi, s, x_tran])
        return z

def get_observation_model(model_config, track=None) -> CasadiObservationModel:
    '''
    Helper function for getting a vehicle model class from a text string
    Should be used anywhere vehicle models may be changed by configuration
    '''
    if model_config.model_name == 'dynamic_bicycle_full_state':
        return CasadiDynamicBicycleFullStateObserver(model_config, track=track)
    elif model_config.model_name == 'dynamic_bicycle_cl_full_state':
        return CasadiDynamicBicycleCLFullStateObserver(model_config, track=track)
    elif model_config.model_name == 'kinematic_bicycle_full_state':
        return CasadiKinematicBicycleFullStateObserver(model_config, track=track)
    elif model_config.model_name == 'kinematic_bicycle_cl_full_state':
        return CasadiKinematicBicycleCLFullStateObserver(model_config, track=track)
    else:
        raise ValueError('Unrecognized vehicle model name: %s' % model_config.model_name)

def main():
    import pdb

    config = ObserverConfig(model_name='dynamic_bicycle_full_state', noise=True, noise_cov=np.eye(6), code_gen=True, opt_flag='O3', verbose=True)
    observer = CasadiDynamicBicycleFullStateObserver(config)
    # H = observer.fH(np.ones(6), np.ones(2)).toarray()

    pdb.set_trace()

if __name__ == '__main__':
    main()

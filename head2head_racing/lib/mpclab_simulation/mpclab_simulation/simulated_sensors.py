import numpy as np
import time
import pdb

from abc import abstractmethod

from mpclab_common.pytypes import VehicleState
from mpclab_simulation.sim_types import GPSConfig, T265SimConfig, EncConfig, EncMsg


class BaseSimSensor():
    @abstractmethod
    def step(self, vehicle_state):
        '''
        When applicable, should return an object of type VehicleCoords
        otherwise, should return a standard message specific to the sensor type (e.g. encoder, lidar, ...)
        '''
        return None

class SimGPSClass(BaseSimSensor):
    """ Object simulating an indoor GPS system """
    def __init__(self, params = None):
        if params is None: params = GPSConfig()

        self.params = params
        self.x_std      = params.x_std
        self.y_std      = params.y_std
        self.z_std      = params.z_std
        self.n_bound    = params.n_bound


        #NOTE: Initial state of the sensor is taken w.r.t. reference, so initial state can be nonero
        # with this in mind, the actual sensor output is (r_ref) + (r_vehicle_com) + (rotated r_offset)
        # note that this is indepdenent of where the vehicle starts and what the initialization of the sensor is
        # which may not be true for some sensors (e.g. they may zero themselves to start)

        # implicitly assumed here is that the x,y axis of the GPS sensor and the track are aligned
        #  - mismatch here is extremely difficult to observe and correct algorithmically and usually needs a fixed reference
        #    (e.g. physicalwalls you can look at and compare to the track definition)


        self.r_offset = np.array([[params.offset_x,     params.offset_y,     params.offset_z]]).T           # offset from vehicle COM to sensor center
        self.r_ref    = np.array([[params.ref_offset_x, params.ref_offset_y, params.ref_offset_z]]).T       # offset from vehicle starting position to sensor zero point
        self.msg = VehicleState()

    def step(self, vehicle_state):

        r_com = np.array([[vehicle_state.x.x, vehicle_state.x.y, vehicle_state.x.z]]).T
        r_sensor_offset = vehicle_state.get_R() @ self.r_offset
        rm = r_com + self.r_ref  + r_sensor_offset
        self.x = rm[0]
        self.y = rm[1]
        self.z = rm[2]


        self.msg.t = vehicle_state.t
        self.msg.x.x = self.x
        self.msg.x.y = self.y
        self.msg.x.z = self.z
        return self.msg

class SimT265Class(BaseSimSensor):
    """ Object simulating a realsense t265 camera """
    #TODO: Implement sensor heading drift and resulting integration errors in position estimate
    def __init__(self, params = None):
        if params is None: params = T265SimConfig()
        self.params = params

        #limit on how many sigma of noise can be added (e.g. 1 will cap noise at +/- 1 standard deviation)
        #set to None for no bound
        self.n_bound    = params.n_bound

        #NOTE: Initial output of the sensor is coerced to zero
        # since the T265 is an inertial measurement unit, it only has good measurements of angular velocity and angular/linear acceleration
        # so long as there are visual features present, it can measure linear velocity accurately as well

        # It is assumed in this model that measurements of angular and linear velocity are accurate but corrupted with noise
        # As a result, the heading and position (integrated heading and velocity) measurements drift substantially over time.
        # Like most IMU systems, the sensor velocity output is in the body frame, and the position output is in the global frame.

        # However, the T265 is not located at the center of mass of the vehicle. As such, the linear velocities are modified by the angular velocity of the vehicle.

        # implicitly assumed here is that the x,y axis of the T265 sensor and the track are aligned at the start of an experiment
        #  - mismatch here could potentially be corrected via GPS


        self.x         = 0.0 # relative to origin of the track (centerline at starting line (s = 0))
        self.y         = 0.0
        self.v_long    = 0.0 # in body frame of the vehicle (x - direction vehicle is heading, y - probably left)
        self.v_tran    = 0.0
        self.a_long    = 0.0
        self.a_tran    = 0.0
        self.psi       = 0.0 # relative to initial heading of the track
        self.psidot    = 0.0

        self.msg = VehicleState()
        self.last_t = None
        return

    def step(self, vehicle_state):
        self.psidot = add_white_noise(vehicle_state.psidot, self.params.psidot_std, sigma_max = self.n_bound)
        self.a_long = add_white_noise(vehicle_state.a_long, self.params.along_std,  sigma_max = self.n_bound)
        self.a_tran = add_white_noise(vehicle_state.a_tran, self.params.atran_std,  sigma_max = self.n_bound)

        # corrupted with white noise after adding offset
        self.v_long = add_white_noise(vehicle_state.v_long - self.params.offset_y * self.psidot, self.params.vlong_std, sigma_max = self.n_bound)
        self.v_tran = add_white_noise(vehicle_state.v_tran + self.params.offset_x * self.psidot, self.params.vtran_std, sigma_max = self.n_bound)

        if self.last_t is None:
            dt = 0
        else:
            dt = vehicle_state.t - self.last_t
        self.last_t = vehicle_state.t
        self.x = self.x + dt * (np.cos(self.psi) * self.v_long - np.sin(self.psi) * self.v_tran)
        self.y = self.y + dt * (np.cos(self.psi) * self.v_tran + np.sin(self.psi) * self.v_long)

        self.msg.t = vehicle_state.t

        self.msg.x.x = self.x
        self.msg.x.y = self.y
        self.msg.v_long = self.v_long  #
        self.msg.v_tran = self.v_tran
        self.msg.a_long = self.a_long
        self.msg.a_tran = self.a_tran
        self.msg.psi = self.psi
        self.msg.psidot = self.psidot
        return self.msg


class SimEncClass(BaseSimSensor):
    # Simulates the velocity estimate from the encoders on the car
    def __init__(self, params = None):
        if params is None: params = EncConfig()

        self.v = 0.0

        self.v_std      = params.v_std
        self.n_bound    = params.n_bound

        self.msg = EncMsg()


    def step(self, vehicle_state):

        self.v = add_white_noise(vehicle_state.v.v_long, self.v_std, sigma_max = self.n_bound)# Velocity along body longitudinal direction
        self.msg.t = vehicle_state.t
        self.msg.fl = self.v
        self.msg.fr = self.v
        self.msg.bl = self.v
        self.msg.br = self.v
        self.msg.ds = self.v
        return self.msg


def add_white_noise(x,std,sigma_max = None):
        if sigma_max == None:
            return x + np.random.normal(0,std)
        else:
            return x + std * np.clip(np.random.normal(), -sigma_max, sigma_max)

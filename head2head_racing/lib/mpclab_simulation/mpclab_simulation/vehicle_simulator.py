import numpy as np
import time
from abc import abstractmethod

from mpclab_simulation.simulated_sensors import SimGPSClass, SimT265Class, SimEncClass

from mpclab_common.models.dynamics_models import get_model

from mpclab_common.pytypes import VehicleState, Position, BodyAngularVelocity, BodyLinearVelocity, BodyLinearAcceleration
from mpclab_common.models.dynamics_models import DynamicBicycleConfig

import pdb


class BaseVehicleSimulator():
    '''
    Base class for any vehicle simulator

    All vehicle simulators must implement a vehicle model from mpclab_common.dynamics_models

    Additionally, vehicles may have vehicle-specific sensors - gps, T265, Lidar, etc...
    which should be output in a dictionary format

    Vehicle simulators take as input the previous simulated vehicle state with modified controller signals
    However, hidden variables may be employed by vehicle and sensor models so a separate object should be made for each vehicle.

    '''

    @abstractmethod
    def __init__(self,vehicle_model = DynamicBicycleConfig()):
        raise NotImplementedError('Cannot call base class')
        return

    @abstractmethod
    def step(self):
        raise NotImplementedError('Cannot call base class')
        return



class BarcSimulator(BaseVehicleSimulator):
    '''
    Class for creating and running a simulated barc vehicle, default settings are those for BARC

    '''
    def __init__(self, vehicle_model: DynamicBicycleConfig,
        gps_params = None,
        t265_params = None,
        enc_params = None):

        self.dt = vehicle_model.dt
        self.vehicle_model = vehicle_model
        self.vehicle_simulator = get_model(vehicle_model)
        self.track = self.vehicle_simulator.track

        self.gps = SimGPSClass(params=gps_params)
        self.t265 = SimT265Class(params=t265_params)
        self.enc = SimEncClass(params=enc_params)
        return

    def step(self,vehicle_state : VehicleState) -> dict:
        '''
        Step a vehicle simulator on the provided vehicle state (kinematic, dynamic state, and well as control signals)

        Also step simulated sensors.

        Vehicle state is modified by reference,
        Each simulated sensor data is returned in a dictionary
        '''
        self.vehicle_simulator.step(vehicle_state)

        gps_out  = self.gps.step(vehicle_state)
        t265_out = self.t265.step(vehicle_state)
        enc_out  = self.enc.step(vehicle_state)

        return {'GPS':gps_out,'T265':t265_out,'Encoder':enc_out}

class VehicleSimulator(BarcSimulator):
    '''
    Flex class that creates a different vehicle simulator depending on settings in mpclab_common
    '''
    def __init__(self, vehicle_model: DynamicBicycleConfig, gps_params = None, t265_params = None, enc_params = None):
        #TODO: Implement standard method for initializing entire code base to BARC or full-size vehicle(s)
        if True:
            BarcSimulator.__init__(self,vehicle_model, gps_params, t265_params, enc_params)
        else:
            raise TypeError('Unrecognized default vehicle type: %s'%'foo')
        return


def time_vehicle_model():
    sim = BarcSimulator(DynamicBicycleConfig())
    state = VehicleState(t = 0, x = Position(x = 0, y = 0), v = BodyLinearVelocity(v_tran = 0, v_long = 0), a = BodyLinearAcceleration( a_tran = 0, a_long = 0), e = OrientationEuler(psi = 0), w = BodyAngularVelocity(w_psi = 0), u =VehicleActuation(u_a = 1, u_steer = 0.1))
    state.update_body_velocity_from_global()
    sim.track.global_to_local_typed(state)
    N = 1000
    t0 = time.time()
    for i in range(N):
        sim.step(state)
    print('%d steps in %f seconds'%(N,(time.time() - t0)))



if __name__ == '__main__':
    time_vehicle_model()

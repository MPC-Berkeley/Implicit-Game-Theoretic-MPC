import numpy as np
import time

# from mpclab_simulation.simulated_sensors import SimGPSClass, SimT265Class, SimEncClass

from mpclab_common.models.dynamics_models import get_dynamics_model
from mpclab_common.pytypes import VehicleState

import pdb


class GPDynamicsSimulator():
    '''
    Class for simulating vehicle dynamics, default settings are those for BARC

    '''
    def __init__(self, t0: float, dynamics_config):
        self.model = get_dynamics_model(t0, dynamics_config)

        return

    def step(self, state: VehicleState):
        #update the vehicle state
        self.model.step(state)
        s = state.p.s
        for i in range(state.lookahead.n):
            state.lookahead.curvature.__setitem__(i, self.model.track.get_curvature(s + state.lookahead.dl * i))
            pass


if __name__ == '__main__':
    sim = GPDynamicsSimulator()
    t0 = time.time()
    for i in range(10):
        print(sim.step((0.1,0.1)))
        print('\n')
    print('10 steps in %f seconds'%(time.time() - t0))

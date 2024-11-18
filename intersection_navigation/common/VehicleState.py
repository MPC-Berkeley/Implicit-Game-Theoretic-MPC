import numpy as np

class VehicleState():
    def __init__(self,state_dict: dict):
        self.x = state_dict['x']
        self.y = state_dict['y']
        self.heading = state_dict['heading']
        self.v = state_dict['v']

        if 's' in state_dict.keys():
            self.s = state_dict['s']
            self.ey = state_dict['ey']
            self.epsi = state_dict['epsi']

    def update(self,state_dict: dict):
        self.x = state_dict['x']
        self.y = state_dict['y']
        self.heading = state_dict['heading']
        self.v = state_dict['v']

        if 's' in state_dict.keys():
            self.s = state_dict['s']
            self.ey = state_dict['ey']
            self.epsi = state_dict['epsi']
import numpy as np

class VehicleAction():
    def __init__(self,action_dict: dict):
        self.a = action_dict['a']
        self.df = action_dict['df']

    def update(self,action_dict: dict):
        self.a = action_dict['a']
        self.df = action_dict['df']


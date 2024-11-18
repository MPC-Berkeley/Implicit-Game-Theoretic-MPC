from VehicleState import VehicleState

class VehicleReference():
    def __init__(self,state_dict: dict):
        self.x = state_dict['x']
        self.y = state_dict['y']
        self.heading = state_dict['heading']
        self.v = state_dict['v']
        self.s = state_dict['s']
        self.K = state_dict['K']
        
        self.ey = state_dict['ey']
        self.epsi = state_dict['epsi']

    def update(self,state_dict: dict):
        self.x = state_dict['x']
        self.y = state_dict['y']
        self.heading = state_dict['heading']
        self.v = state_dict['v']

        self.s = state_dict['s']
        self.K = state_dict['K']
        self.ey = state_dict['ey']
        self.epsi = state_dict['epsi']

    def get_state_array(self):
        return [self.x,self.y,self.s,self.ey,self.epsi,self.v] # No heading

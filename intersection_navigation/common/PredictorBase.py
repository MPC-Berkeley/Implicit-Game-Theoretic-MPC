from infrastructure.VehicleReference import VehicleReference

class PredictorBase():

    def __init__(self, N, dt):
        self.N = N # Predicition horizon
        self.dt = dt # Time step

    def predict(self,agents,routes):
        '''
        Given agents' current states and routes, predict the agents' for the next N steps
        Output: List of List of VehicleReferences
        '''
        return [VehicleReference()]
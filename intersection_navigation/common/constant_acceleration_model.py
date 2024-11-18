from PredictorBase import PredictorBase
from VehicleReference import VehicleReference
from utils import frenet2global
import numpy as np
import casadi as ca
import os
import yaml

class ConstantAccelerationModel(PredictorBase):
    def __init__(self,N = 10, dt=0.1,constant_speed=False):
        super().__init__(N,dt)
        # Load the environment config
        with open(os.getcwd()+'/common/fourwayint.yaml') as f:
            self.env_config = yaml.load(f, Loader=yaml.FullLoader)

        self.constant_speed = constant_speed

    def predict(self,agents,agent_cur_inputs,routes,refs):
        '''
        Constant acceleration model.
        1) Forecast s and v for the next N steps given current a
        2) Get the corresponding x, y, heading
        3) Return a list of VehicleReference objects
        '''
        #Initialize the prediction list
        if self.constant_speed:
            a_arr = [0 for i in range(len(agents))]
        else:
            a_arr = [agent_cur_inputs[i].a for i in range(len(agents))] 
        preds = []
        for i, agent in enumerate(agents):
           
            # Initial state
            ey = 0; epsi = 0
            K = None
            s_cur = agent['state'].s
            v = agent['state'].v
            
            # Append the current state
            preds.append([VehicleReference({'x':agent['state'].x,'y':agent['state'].y,'heading':agent['state'].heading,'v':v,'s':s_cur,'K':K, 'ey':ey,'epsi':epsi})])


            v_max = self.env_config['v_max']
            v_min = self.env_config['v_min']
            
            # Psi ref function. Specific for fourwayint.py
            if np.all(refs[i]['K']==0):
                def straight(s_cur):
                    return refs[i]['heading'][0]
                s = ca.SX.sym('s')
                psi_ref = straight
            else:
                # Create a symbolic variable
                s = ca.SX.sym('s')

                if routes[i] in ['12','23','34','41']:
                    breakpoints = ca.DM([0,(self.env_config['road_length']-self.env_config['road_width'])/2,(self.env_config['road_length']-self.env_config['road_width'])/2 + max(abs(1/refs[i]['K'][np.nonzero(refs[i]['K'])])) * np.pi/2,1000])  # Breakpoints where the function values change
                else:
                    breakpoints = ca.DM([0,(self.env_config['road_length']-self.env_config['road_width'])/2 - (self.env_config['road_width'] - self.env_config['ca_radius']),(self.env_config['road_length']-self.env_config['road_width'])/2 - (self.env_config['road_width'] - self.env_config['ca_radius']) + max(abs(1/refs[i]['K'][np.nonzero(refs[i]['K'])])) * np.pi/2,1000])

                if routes[i] in ['32','41']:
                    values = [abs(refs[i]['heading'][0]), abs(refs[i]['heading'][0]), abs(refs[i]['heading'][-1]), abs(refs[i]['heading'][-1])]
                else:
                    values = [refs[i]['heading'][0], refs[i]['heading'][0], refs[i]['heading'][-1], refs[i]['heading'][-1]]
                pw_linear = ca.pw_lin(s, breakpoints, values)
                psi_ref = ca.Function('psi_ref', [s], [pw_linear])
            
            # Predict the next N steps
            for k in range(self.N):
                s_cur += v * self.dt + 0.5 * a_arr[i] * self.dt**2 # Double integrator model
                v = np.clip(v + a_arr[i] * self.dt, v_min, v_max)

                # Get the corresponding x, y, heading
                theta = psi_ref(s_cur)
                pos = frenet2global(s_cur=s_cur,ref=refs[i],route=routes[i],road_length=self.env_config['road_length'],road_width=self.env_config['road_width'],ca_radius=self.env_config['ca_radius'])
                x = pos[0,0]
                y = pos[1,0]

                # We don't need K, ey, epsi predictions for surrounding vehicles
                preds[i].append(VehicleReference({'x':x,'y':y,'heading':theta,'v':v,'s':s_cur,'K':K, 'ey':ey,'epsi':epsi}))

        return preds


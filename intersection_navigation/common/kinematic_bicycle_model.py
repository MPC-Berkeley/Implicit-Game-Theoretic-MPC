import numpy as np
import casadi as ca
from VehicleState import VehicleState
from VehicleReference import VehicleReference

class KinematicBicycleModel():
    def __init__(self,l_r,l_f,width, dt, discretization='euler',mode='numpy'):
        self.l_r = l_r
        self.l_f = l_f
        self.width = width
        self.delta_t = dt
        self.discretization = discretization
        self.mode = mode
    
    def __call__(self,state,action):
        if self.discretization == 'euler':
            if self.mode == 'numpy':
                # Assume state is of class VehicleReference
                x = state.x
                y = state.y
                psi = state.heading
                v = state.v

                a = action.a
                delta_f = action.df
                
                beta = np.arctan(((self.l_r/(self.l_f + self.l_r))*np.tan(delta_f)))
                x_new = x + self.delta_t*v*np.cos(psi+beta)
                y_new = y + self.delta_t*v*np.sin(psi+beta)
                psi_new = psi + self.delta_t*(v*np.cos(beta)/(self.l_r+self.l_f)*np.tan(delta_f))
                v_new = v + self.delta_t*a
            elif self.mode == 'casadi':
                x = state.x
                y = state.y
                psi = state.heading
                v = state.v

                a = action.a
                delta_f = action.df
                
                beta = ca.atan((self.l_r/(self.l_f+self.l_r))*ca.tan(delta_f))
                x_new = x + self.delta_t*v*ca.cos(psi+beta)
                y_new = y + self.delta_t*v*ca.sin(psi+beta)
                psi_new = psi + self.delta_t*(v*ca.cos(beta)/(self.l_r+self.l_f)*ca.tan(delta_f))
                v_new = v + self.delta_t*a
        else:
            raise NotImplementedError
        

        return VehicleReference({'x':x_new,'y':y_new,'heading':psi_new,'v':v_new, 's':0, 'ey':0, 'epsi':0, 'K':0})

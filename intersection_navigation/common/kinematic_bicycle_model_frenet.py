import numpy as np
import casadi as ca
from VehicleState import VehicleState
from VehicleReference import VehicleReference

class KinematicBicycleModelFrenet():
    def __init__(self,l_r,l_f,width, dt, discretization='euler',mode = 'numpy', num_rk4_steps = 10):
        self.l_r = l_r
        self.l_f = l_f
        self.width = width
        self.delta_t = dt
        self.discretization = discretization
        self.mode = mode
        self.num_rk4_steps = num_rk4_steps
    
    def __call__(self,state,action):
        if self.discretization == 'euler':
            if self.mode=='numpy':
                # Assume state is of class VehicleState
                s = state.s
                ey = state.ey
                epsi = state.epsi
                v = state.v
                K = state.K

                a = action.a
                delta_f = action.df
                
                beta = np.arctan((self.l_r/(self.l_f+self.l_r))*np.tan(delta_f))
                dsdt = v * np.cos(beta + epsi) / (1 - K(s) * ey )
                dyawdt = v * np.sin(beta) / self.l_r

                s_new = s + self.delta_t*dsdt
                ey_new = ey + self.delta_t*v*np.sin(beta + epsi)
                epsi_new = epsi + self.delta_t*(dyawdt - dsdt * K(s))
                v_new = v + self.delta_t*a

                x_new = x + self.delta_t*v*np.cos(psi+beta)
                y_new = y + self.delta_t*v*np.sin(psi+beta)
                psi_new = psi + self.delta_t*v*np.sin(beta) /self.l_r

            elif self.mode == 'casadi':
                s = state.s
                ey = state.ey
                epsi = state.epsi
                v = state.v
                x = state.x
                y = state.y
                psi = state.heading
                K = state.K

                a = action.a
                delta_f = action.df
                
                beta = ca.atan((self.l_r/(self.l_f+self.l_r))*ca.tan(delta_f))
                dsdt = v * ca.cos(beta + epsi) / (1 - K(s) * ey )
                dyawdt = v * ca.sin(beta) / self.l_r

                s_new = s + self.delta_t*dsdt
                ey_new = ey + self.delta_t*v*ca.sin(beta + epsi)
                epsi_new = epsi + self.delta_t*(dyawdt - dsdt * K(s))
                v_new = v + self.delta_t*a

                x_new = x + self.delta_t*v*ca.cos(psi+beta)
                y_new = y + self.delta_t*v*ca.sin(psi+beta)
                psi_new = psi + self.delta_t*v*ca.sin(beta) /self.l_r
            else:
                raise ValueError('Invalid mode')
        elif self.discretization == 'rk4':
            if self.mode == 'numpy':
                def dsdt(s,ey,epsi,v,K,a,delta_f):
                    beta = np.arctan((self.l_r/(self.l_f+self.l_r))*np.tan(delta_f))
                    return v * np.cos(beta + epsi) / (1 - K(s) * ey )
                def deydt(s,ey,epsi,v,K,a,delta_f):
                    beta = np.arctan((self.l_r/(self.l_f+self.l_r))*np.tan(delta_f))
                    return v*np.sin(beta + epsi)
                def depsidt(s,ey,epsi,v,K,a,delta_f):
                    beta = np.arctan((self.l_r/(self.l_f+self.l_r))*np.tan(delta_f))
                    return v * np.sin(beta) / self.l_r - float(dsdt(s,ey,epsi,v,K,a,delta_f) * K(s) )
                def dvdt(s,ey,epsi,v,K,a,delta_f):
                    return a
                def dxdt(s,ey,epsi,v,K,a,delta_f,psi_):
                    beta = np.arctan((self.l_r/(self.l_f+self.l_r))*np.tan(delta_f))
                    return v*np.cos(psi_+beta)
                def dydt(s,ey,epsi,v,K,a,delta_f,psi_):
                    beta = np.arctan((self.l_r/(self.l_f+self.l_r))*np.tan(delta_f))
                    return v*np.sin(psi_+beta)
                def dheadingdt(s,ey,epsi,v,K,a,delta_f):
                    beta = np.arctan((self.l_r/(self.l_f+self.l_r))*np.tan(delta_f))
                    return v*np.sin(beta) / self.l_r
                

                h = self.delta_t/self.num_rk4_steps

                s = state.s
                ey = state.ey
                epsi = state.epsi
                v = state.v
                x = state.x
                y = state.y
                psi = state.heading
                K = state.K

                a = action.a
                delta_f = action.df

                for i in range(self.num_rk4_steps):
                    k1 = np.array([float(dsdt(s,ey,epsi,v,K,a,delta_f)),deydt(s,ey,epsi,v,K,a,delta_f),depsidt(s,ey,epsi,v,K,a,delta_f),dvdt(s,ey,epsi,v,K,a,delta_f),dxdt(s,ey,epsi,v,K,a,delta_f,psi),dydt(s,ey,epsi,v,K,a,delta_f,psi),dheadingdt(s,ey,epsi,v,K,a,delta_f)])
                    k2 = np.array([float(dsdt(s+h/2*k1[0],ey+h/2*k1[1],epsi+h/2*k1[2],v+h/2*k1[3],K,a,delta_f)),deydt(s+h/2*k1[0],ey+h/2*k1[1],epsi+h/2*k1[2],v+h/2*k1[3],K,a,delta_f),depsidt(s+h/2*k1[0],ey+h/2*k1[1],epsi+h/2*k1[2],v+h/2*k1[3],K,a,delta_f),dvdt(s+h/2*k1[0],ey+h/2*k1[1],epsi+h/2*k1[2],v+h/2*k1[3],K,a,delta_f),dxdt(s+h/2*k1[0],ey+h/2*k1[1],epsi+h/2*k1[2],v+h/2*k1[3],K,a,delta_f,psi+h/2*k1[6]),dydt(s+h/2*k1[0],ey+h/2*k1[1],epsi+h/2*k1[2],v+h/2*k1[3],K,a,delta_f,psi+h/2*k1[6]),dheadingdt(s+h/2*k1[0],ey+h/2*k1[1],epsi+h/2*k1[2],v+h/2*k1[3],K,a,delta_f)])
                    k3 = np.array([float(dsdt(s+h/2*k2[0],ey+h/2*k2[1],epsi+h/2*k2[2],v+h/2*k2[3],K,a,delta_f)),deydt(s+h/2*k2[0],ey+h/2*k2[1],epsi+h/2*k2[2],v+h/2*k2[3],K,a,delta_f),depsidt(s+h/2*k2[0],ey+h/2*k2[1],epsi+h/2*k2[2],v+h/2*k2[3],K,a,delta_f),dvdt(s+h/2*k2[0],ey+h/2*k2[1],epsi+h/2*k2[2],v+h/2*k2[3],K,a,delta_f),dxdt(s+h/2*k2[0],ey+h/2*k2[1],epsi+h/2*k2[2],v+h/2*k2[3],K,a,delta_f,psi+h/2*k2[6]),dydt(s+h/2*k2[0],ey+h/2*k2[1],epsi+h/2*k2[2],v+h/2*k2[3],K,a,delta_f,psi+h/2*k2[6]),dheadingdt(s+h/2*k2[0],ey+h/2*k2[1],epsi+h/2*k2[2],v+h/2*k2[3],K,a,delta_f)])
                    k4 = np.array([float(dsdt(s+h*k3[0],ey+h*k3[1],epsi+h*k3[2],v+h*k3[3],K,a,delta_f)),deydt(s+h*k3[0],ey+h*k3[1],epsi+h*k3[2],v+h*k3[3],K,a,delta_f),depsidt(s+h*k3[0],ey+h*k3[1],epsi+h*k3[2],v+h*k3[3],K,a,delta_f),dvdt(s+h*k3[0],ey+h*k3[1],epsi+h*k3[2],v+h*k3[3],K,a,delta_f),dxdt(s+h*k3[0],ey+h*k3[1],epsi+h*k3[2],v+h*k3[3],K,a,delta_f,psi+h/2*k3[6]),dydt(s+h*k3[0],ey+h*k3[1],epsi+h*k3[2],v+h*k3[3],K,a,delta_f,psi+h/2*k3[6]),dheadingdt(s+h*k3[0],ey+h*k3[1],epsi+h*k3[2],v+h*k3[3],K,a,delta_f)])

                    s_new = s + h/6*(k1[0] + 2*k2[0] + 2*k3[0] + k4[0])
                    ey_new = ey + h/6*(k1[1] + 2*k2[1] + 2*k3[1] + k4[1])
                    epsi_new = epsi + h/6*(k1[2] + 2*k2[2] + 2*k3[2] + k4[2])
                    v_new = v + h/6*(k1[3] + 2*k2[3] + 2*k3[3] + k4[3])
                    x_new = x + h/6*(k1[4] + 2*k2[4] + 2*k3[4] + k4[4])
                    y_new = y + h/6*(k1[5] + 2*k2[5] + 2*k3[5] + k4[5])
                    psi_new = psi + h/6*(k1[6] + 2*k2[6] + 2*k3[6] + k4[6])

                    s = s_new
                    ey = ey_new
                    epsi = epsi_new
                    v = v_new
                    x = x_new
                    y = y_new
                    psi = psi_new

            elif self.mode == 'casadi':
                def dsdt(s,ey,epsi,v,K,a,delta_f):
                    beta = ca.atan((self.l_r/(self.l_f+self.l_r))*ca.tan(delta_f))
                    return v * ca.cos(beta + epsi) / (1 - K(s) * ey )
                def deydt(s,ey,epsi,v,K,a,delta_f):
                    beta = ca.atan((self.l_r/(self.l_f+self.l_r))*ca.tan(delta_f))
                    return v*ca.sin(beta + epsi)
                def depsidt(s,ey,epsi,v,K,a,delta_f):
                    beta = ca.atan((self.l_r/(self.l_f+self.l_r))*ca.tan(delta_f))
                    return v * ca.sin(beta) / self.l_r - dsdt(s,ey,epsi,v,K,a,delta_f) * K(s)
                def dvdt(s,ey,epsi,v,K,a,delta_f):
                    return a
                def dxdt(s,ey,epsi,v,K,a,delta_f,psi_):
                    beta = ca.atan((self.l_r/(self.l_f+self.l_r))*ca.tan(delta_f))
                    return v*ca.cos(psi_+beta)
                def dydt(s,ey,epsi,v,K,a,delta_f,psi_):
                    beta = ca.atan((self.l_r/(self.l_f+self.l_r))*ca.tan(delta_f))
                    return v*ca.sin(psi_+beta)
                def dheadingdt(s,ey,epsi,v,K,a,delta_f):
                    beta = ca.atan((self.l_r/(self.l_f+self.l_r))*ca.tan(delta_f))
                    return v*ca.sin(beta) / self.l_r
                
                h = self.delta_t/self.num_rk4_steps

                s = state.s
                ey = state.ey
                epsi = state.epsi
                v = state.v
                x = state.x
                y = state.y
                psi = state.heading
                K = state.K

                a = action.a
                delta_f = action.df

                for _ in range(self.num_rk4_steps):
                    k1 = ca.vec(ca.vertcat(dsdt(s,ey,epsi,v,K,a,delta_f),deydt(s,ey,epsi,v,K,a,delta_f),depsidt(s,ey,epsi,v,K,a,delta_f),dvdt(s,ey,epsi,v,K,a,delta_f),dxdt(s,ey,epsi,v,K,a,delta_f,psi),dydt(s,ey,epsi,v,K,a,delta_f,psi),dheadingdt(s,ey,epsi,v,K,a,delta_f)))
                    k2 = ca.vec(ca.vertcat(dsdt(s+h/2*k1[0],ey+h/2*k1[1],epsi+h/2*k1[2],v+h/2*k1[3],K,a,delta_f),deydt(s+h/2*k1[0],ey+h/2*k1[1],epsi+h/2*k1[2],v+h/2*k1[3],K,a,delta_f),depsidt(s+h/2*k1[0],ey+h/2*k1[1],epsi+h/2*k1[2],v+h/2*k1[3],K,a,delta_f),dvdt(s+h/2*k1[0],ey+h/2*k1[1],epsi+h/2*k1[2],v+h/2*k1[3],K,a,delta_f),dxdt(s+h/2*k1[0],ey+h/2*k1[1],epsi+h/2*k1[2],v+h/2*k1[3],K,a,delta_f,psi+h/2*k1[6]),dydt(s+h/2*k1[0],ey+h/2*k1[1],epsi+h/2*k1[2],v+h/2*k1[3],K,a,delta_f,psi+h/2*k1[6]),dheadingdt(s+h/2*k1[0],ey+h/2*k1[1],epsi+h/2*k1[2],v+h/2*k1[3],K,a,delta_f)))
                    k3 = ca.vec(ca.vertcat(dsdt(s+h/2*k2[0],ey+h/2*k2[1],epsi+h/2*k2[2],v+h/2*k2[3],K,a,delta_f),deydt(s+h/2*k2[0],ey+h/2*k2[1],epsi+h/2*k2[2],v+h/2*k2[3],K,a,delta_f),depsidt(s+h/2*k2[0],ey+h/2*k2[1],epsi+h/2*k2[2],v+h/2*k2[3],K,a,delta_f),dvdt(s+h/2*k2[0],ey+h/2*k2[1],epsi+h/2*k2[2],v+h/2*k2[3],K,a,delta_f),dxdt(s+h/2*k2[0],ey+h/2*k2[1],epsi+h/2*k2[2],v+h/2*k2[3],K,a,delta_f,psi+h/2*k2[6]),dydt(s+h/2*k2[0],ey+h/2*k2[1],epsi+h/2*k2[2],v+h/2*k2[3],K,a,delta_f,psi+h/2*k2[6]),dheadingdt(s+h/2*k2[0],ey+h/2*k2[1],epsi+h/2*k2[2],v+h/2*k2[3],K,a,delta_f)))
                    k4 = ca.vec(ca.vertcat(dsdt(s+h*k3[0],ey+h*k3[1],epsi+h*k3[2],v+h*k3[3],K,a,delta_f),deydt(s+h*k3[0],ey+h*k3[1],epsi+h*k3[2],v+h*k3[3],K,a,delta_f),depsidt(s+h*k3[0],ey+h*k3[1],epsi+h*k3[2],v+h*k3[3],K,a,delta_f),dvdt(s+h*k3[0],ey+h*k3[1],epsi+h*k3[2],v+h*k3[3],K,a,delta_f),dxdt(s+h*k3[0],ey+h*k3[1],epsi+h*k3[2],v+h*k3[3],K,a,delta_f,psi+h/2*k3[6]),dydt(s+h*k3[0],ey+h*k3[1],epsi+h*k3[2],v+h*k3[3],K,a,delta_f,psi+h/2*k3[6]),dheadingdt(s+h*k3[0],ey+h*k3[1],epsi+h*k3[2],v+h*k3[3],K,a,delta_f)))

                    s_new = s + h/6*(k1[0] + 2*k2[0] + 2*k3[0] + k4[0])
                    ey_new = ey + h/6*(k1[1] + 2*k2[1] + 2*k3[1] + k4[1])
                    epsi_new = epsi + h/6*(k1[2] + 2*k2[2] + 2*k3[2] + k4[2])
                    v_new = v + h/6*(k1[3] + 2*k2[3] + 2*k3[3] + k4[3])
                    x_new = x + h/6*(k1[4] + 2*k2[4] + 2*k3[4] + k4[4])
                    y_new = y + h/6*(k1[5] + 2*k2[5] + 2*k3[5] + k4[5])
                    psi_new = psi + h/6*(k1[6] + 2*k2[6] + 2*k3[6] + k4[6])

                    s = s_new
                    ey = ey_new
                    epsi = epsi_new
                    v = v_new
                    x = x_new
                    y = y_new
                    psi = psi_new

            else:
                raise ValueError('Invalid mode')
        else:
            raise ValueError('Invalid discretization method')

        return VehicleReference({'s':s_new,'ey':ey_new,'epsi':epsi_new,'v':v_new,'x':x_new,'y':y_new,'heading':psi_new, 'K':K})

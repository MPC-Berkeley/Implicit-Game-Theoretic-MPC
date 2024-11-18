import casadi as ca
import numpy as np
import polytope as pt
import os
import pickle
import time
import sys
import yaml, torch
import scipy as sp
from model import mlp

sys.path.insert(0, os.getcwd()+'/common')
from utils import rotation_translation
from VehicleAction import VehicleAction
from VehicleReference import VehicleReference
from kinematic_bicycle_model_frenet import KinematicBicycleModelFrenet
from utils import route_encoding, Cinf, scenario_encoding

class MPC_Planner():

    def __init__(self,
                 N  = 10,
                 dt = 0.1,
                 agents = None,
                 goals = None,
                 ca_radius = 2.8,
                 ref = None,
                 road_dim = (10,50),
                 routes = None,
                 ds_right = None,
                 index = None,
                 num_rk4_steps=7,
                 solver='ipopt',
                 ca_type = 'circle',
                 nn_config_dir = None,
                 use_NN_cost2go = False,
                 weights = [1,1,1]):
        self.N = N
        self.dt = dt
        self.ca_type = ca_type
        self.x_sol_prev  = None
        if self.ca_type == 'obca':
            self.d_min = 0
        else:
            self.d_min = 2*ca_radius # + index

        # Vehicle parameters
        self.l = 4.47
        self.l_f = 4.47/2
        self.l_r = 4.47/2
        self.width = 2.0

        # System limitations
        self.routes = routes
        self.steering_rate_limit = 0.7   # rad/s
        self.jerk_limit = 0.9   # m/s^3 (jerk limit)
        self.v_min = 0
        self.v_max = 5 
        self.a_min = -4
        self.a_max = 3
        self.ey_lim = 0.2 
        self.max_steering = 1
        self.use_NN_cost2go = use_NN_cost2go

        self.road_width = road_dim[0]
        self.road_length = road_dim[1]

        self.ref = ref
        self.model = KinematicBicycleModelFrenet(self.l_r,self.l_f,self.width,self.dt,discretization='rk4',mode='casadi',num_rk4_steps=num_rk4_steps)
        self.ds_right = ds_right
        self.weights = weights
        if use_NN_cost2go:
            with open(nn_config_dir, 'rb') as f:
                self.NN_config = yaml.load(f, Loader=yaml.FullLoader)
        self.nx = 7
        self.nu = 2
        self.agents = agents
        self.ind = index
        self.initial_agent = agents[self.ind]
        assert index is not None
        self.goals = goals  
        self.M = len(ref)
        self.num_obstacles = self.M - 1
        self.offset = 0
        self.pred_ind = [i for i in range(self.M)]
        del self.pred_ind[self.ind]

        X = pt.Polytope(np.array([[1, 0],
                          [0, 1],
                          [-1, 0],
                          [0, -1]]),
                np.array([[5],
                          [3],
                          [1],
                          [4]]))
        # Input constraint
        U = pt.Polytope(np.array([[1],
                                [-1]]),
                        np.array([[self.dt*self.jerk_limit],
                                [self.dt*self.jerk_limit]]))

        A = np.array([[1,self.dt],[0,1]])
        B = np.array([[0],[1]])
        self.C_inf = Cinf(A, B, X, U)
        if self.ca_type == 'obca':
            self.G, self.g = rotation_translation([0,0],0,self.l,self.width,mode='casadi') # Ego vehicle (H-representation)

        if self.use_NN_cost2go:
            with open(os.getcwd()+self.NN_config['data_path'], 'rb') as f:
                D = pickle.load(f)
            h_size = self.NN_config['hidden_size']
            self.feature_mean = D['train'].feature_mean
            self.feature_cov = D['train'].feature_cov
            reg_value = 1e-6
            self.feature_cov += reg_value * np.eye(self.feature_cov.shape[0])
            self.feature_cov_inv = np.real(sp.linalg.inv(sp.linalg.sqrtm(D['train'].feature_cov)))
            self.target_mean = D['train'].target_mean
            self.target_cov = np.sqrt(D['train'].target_cov)
            self.mlp = mlp(input_layer_size=len(D['test'][0][0]),
                output_layer_size=len(D['test'][0][1]),
                hidden_layer_sizes=[h_size for _ in range(self.NN_config['num_layers'])], 
                activation='tanh', 
                batch_norm=False)
            self.mlp.load_state_dict(torch.load(os.getcwd()+self.NN_config['model_path']))
            self.V_ca = self.mlp.get_casadi_mlp()
            self.mlp.to(torch.device('cuda'))
            self.mlp.eval()

        self.opti = ca.Opti()
        p_opts = {'expand': True, 'print_time':0, 'verbose' :False, 'error_on_fail':0} # 'nlp_scaling_method': 'gradient-based'
        s_opts = {'print_level': 0,
                  'output_file':'ipopt_output.txt',
                  'tol':1e-3,
                  'dual_inf_tol':1e-3,
                  'constr_viol_tol':1e-3,
                  'max_wall_time': self.N,
                  'max_iter': self.N*100, 
                  'acceptable_obj_change_tol': 1e-4,
                  'nlp_scaling_method':'gradient-based',
                  'warm_start_init_point': 'yes',
                  'warm_start_bound_push': 1e-9,
                  'warm_start_bound_frac': 1e-9,
                  'warm_start_slack_bound_frac': 1e-9,
                  'warm_start_slack_bound_push': 1e-9,
                  'warm_start_mult_bound_push': 1e-9}
        self.opti.solver(solver,p_opts,s_opts)
        self.add_decision_variables()
        self.set_reference(ref)
        self.add_state_and_input_constraints()
        self.add_initial_constraints()
        self.add_input_rate_constraints()
        self.add_dynamic_constraints()
        self.add_ey_constraints()
        self.add_terminal_constraints()
        assert agents is not None, 'Agents are not defined'
        if self.ca_type == 'obca':
            self.add_obca_collision_avoidance_constraints()
        else:
            self.add_collision_avoidance_constraints()
        self.opti.minimize(self.cost_function())

    def add_decision_variables(self):
        self.x = self.opti.variable(self.nx,self.N+1) # [x,y,s,ey,epsi,v, psi]
        self.u = self.opti.variable(self.nu,self.N)
        self.u_prev = self.opti.parameter(self.nu)
        self.x0 = self.opti.parameter(self.nx)
        self.preds = self.opti.parameter(self.num_obstacles*(self.nx), self.N+1) # Obstacles (x,y,psi,v) along the prediction horizon
        self.raw_preds = self.opti.parameter(1,self.M*(self.nx))

        if self.ca_type == 'obca':
            self.lambda_vars = []
            self.mu_vars = []
            for _ in range(self.num_obstacles):
                self.lambda_vars.append(self.opti.variable(4,self.N+1)) # 4 linear constraints compose the rectangular polytope to represent the obstacle
                self.mu_vars.append(self.opti.variable(4,self.N+1))

    def add_terminal_constraints(self):
        temp = self.C_inf.A @ np.array([[self.x[5,self.N-1]], [self.u[0,self.N-1]]]) 
        for m in range(temp.shape[0]):
            self.opti.subject_to(temp[m,0] <= self.C_inf.b[m])

    def add_dynamic_constraints(self):
        if np.all(self.ref[self.ind]['K']==0):
            def straight(s_cur):
                return 0
            s = ca.SX.sym('s')
            K = ca.Function('K',[s],[straight(s)])
        else:
            if self.routes[self.ind] in ['12','23','34','41']: # Left turn
                breakpoints = ca.DM([(self.road_length-self.road_width)/2,(self.road_length-self.road_width)/2 + max(abs(1/self.ref[self.ind]['K'][np.nonzero(self.ref[self.ind]['K'])])) * np.pi/2])  # Breakpoints where the function values change
            else:
                breakpoints = ca.DM([(self.road_length-self.road_width)/2 - self.ds_right,(self.road_length-self.road_width)/2 - self.ds_right + max(abs(1/self.ref[self.ind]['K'][np.nonzero(self.ref[self.ind]['K'])])) * np.pi/2])
            values = ca.DM([0.0, self.ref[self.ind]['K'][np.nonzero(self.ref[self.ind]['K'])[0][0]], 0.0])  # Function values on each segment

            # Create a symbolic variable
            s = ca.SX.sym('s')

            # Create a piecewise constant function for curvature
            curvature_func = ca.pw_const(s, breakpoints, values)
            K = ca.Function('K', [s], [curvature_func])
        for k in range(self.N):
            state_kp1 = self.model(VehicleReference({'s':self.x[2,k],'ey':self.x[3,k],'epsi':self.x[4,k],'v':self.x[5,k],'K':K,'x':self.x[0,k], 'y':self.x[1,k], 'heading': self.x[6,k]}),VehicleAction({'a':self.u[0,k],'df':self.u[1,k]}))
            self.opti.subject_to(self.x[0,k+1] == state_kp1.x)
            self.opti.subject_to(self.x[1,k+1] == state_kp1.y)
            self.opti.subject_to(self.x[2,k+1] == state_kp1.s)
            self.opti.subject_to(self.x[3,k+1] == state_kp1.ey)
            self.opti.subject_to(self.x[4,k+1] == state_kp1.epsi)
            self.opti.subject_to(self.x[5,k+1] == state_kp1.v)
            self.opti.subject_to(self.x[6,k+1] == state_kp1.heading)

    def add_obca_collision_avoidance_constraints(self):
        for k in range(1,self.N+1):
            for m in range(self.num_obstacles):
                A, b = rotation_translation([self.preds[(self.nx)*m+0,k],self.preds[(self.nx)*m+1,k]],self.preds[(self.nx)*m+6,k],self.l,self.width,mode='casadi')
                Rk = ca.vertcat(ca.horzcat(ca.cos(self.x[6,k]), -ca.sin(self.x[6,k])), ca.horzcat(ca.sin(self.x[6,k]), ca.cos(self.x[6,k])))
                self.opti.subject_to(-self.g.T @ self.mu_vars[m][:,k] + (A @ self.x[0:2,k] - b).T @ self.lambda_vars[m][:,k] >= self.d_min + 1e-6)
                self.opti.subject_to(self.G.T @ self.mu_vars[m][:,k] + Rk.T @ A.T @ self.lambda_vars[m][:,k] == 0)
                self.opti.subject_to(ca.bilin(ca.DM.eye(2),A.T @ self.lambda_vars[m][:,k]) <= 1)
                for n in range(4):
                    self.opti.subject_to(self.lambda_vars[m][n,k] >= 0)
                    self.opti.subject_to(self.mu_vars[m][n,k] >= 0) 

    def add_collision_avoidance_constraints(self):
        for k in range(1,self.N+1):  
            for i in range(self.num_obstacles):
                self.opti.subject_to( self.d_min**2 - ca.bilin(ca.DM.eye(2), self.x[0:2,k]-self.preds[(self.nx)*i+0:(self.nx)*i+2,k], self.x[0:2,k]-self.preds[(self.nx)*i+0:(self.nx)*i+2,k]) <= 0)

    def add_initial_constraints(self):
        self.opti.subject_to(self.x[0,0] == self.x0[0]) # xs
        self.opti.subject_to(self.x[1,0] == self.x0[1]) # y
        if self.routes[self.ind] in ['32','41']:
            self.opti.subject_to(self.x[6,0] == ca.fabs(self.x0[6]))
        else:
            self.opti.subject_to(self.x[6,0] == self.x0[6])

        self.opti.subject_to(self.x[2,0] == self.x0[2]) # s
        self.opti.subject_to(self.x[3,0] == self.x0[3]) # ey
        self.opti.subject_to(self.x[4,0] == self.x0[4]) # epsi
        self.opti.subject_to(self.x[5,0] == self.x0[5]) # v

    def update_predictions(self,preds,raw_preds=None):
        assert len(preds) == self.M, ValueError('Invalid number of predictions')
        m = 0
        self.pred_ind = []
        for i, pred in enumerate(preds):  
            if i != self.ind:
                self.pred_ind.append(i)
                assert len(pred) == self.N+1, ValueError('Invalid prediction length (Horizon)')
                for k in range(self.N+1):               
                    if self.routes[i] in ['32','41']:
                        heading = abs(pred[k].heading)
                    else:
                        heading = pred[k].heading
                    self.opti.set_value(self.preds[(self.nx)*m+0,k],pred[k].x) #assumes pred[k] is a VehicleReference or VehicleState object that has x and y attributes
                    self.opti.set_value(self.preds[(self.nx)*m+1,k],pred[k].y)
                    self.opti.set_value(self.preds[(self.nx)*m+2,k],pred[k].s)
                    self.opti.set_value(self.preds[(self.nx)*m+3,k],pred[k].ey)
                    self.opti.set_value(self.preds[(self.nx)*m+4,k],pred[k].epsi)
                    self.opti.set_value(self.preds[(self.nx)*m+5,k],pred[k].v)
                    self.opti.set_value(self.preds[(self.nx)*m+6,k],heading)
                m += 1
        # Raw preds are needed for local quadratic approximation of the neural network model
        if raw_preds is not None:
            self.raw_preds_np = np.zeros((1,self.nx*self.M))
            for i, pred in enumerate(raw_preds):
                self.opti.set_value(self.raw_preds[:,(self.nx)*i+0],pred[-1].x)
                self.opti.set_value(self.raw_preds[:,(self.nx)*i+1],pred[-1].y)
                self.opti.set_value(self.raw_preds[:,(self.nx)*i+2],pred[-1].s)
                self.opti.set_value(self.raw_preds[:,(self.nx)*i+3],pred[-1].ey)
                self.opti.set_value(self.raw_preds[:,(self.nx)*i+4],pred[-1].epsi)
                self.opti.set_value(self.raw_preds[:,(self.nx)*i+5],pred[-1].v)
                self.raw_preds_np[:,(self.nx)*i:(self.nx)*(i+1)] = np.array([[pred[-1].x,pred[-1].y,pred[-1].s,pred[-1].ey,pred[-1].epsi,pred[-1].v,np.float64(pred[-1].heading)]])   
                if self.routes[i] in ['32','41']:
                    self.opti.set_value(self.raw_preds[:,(self.nx)*i+6],abs(pred[-1].heading))
                else:
                    self.opti.set_value(self.raw_preds[:,(self.nx)*i+6],pred[-1].heading)
        # NN
        self.NN_query_time = -1 # Not queried

    def update_initial_condition(self,agent,u_prev):
        self.u_prev_raw = u_prev
        if self.routes[self.ind] in ['32','41']:
           self.opti.set_value(self.x0[6],abs(agent['state'].heading))
        else:
           self.opti.set_value(self.x0[6],agent['state'].heading)
        self.opti.set_value(self.u_prev,[u_prev.a,u_prev.df])
        self.opti.set_value(self.x0[0],agent['state'].x) 
        self.opti.set_value(self.x0[1],agent['state'].y) 
        self.opti.set_value(self.x0[2],agent['state'].s)
        self.opti.set_value(self.x0[3],agent['state'].ey)
        self.opti.set_value(self.x0[4],agent['state'].epsi)
        self.opti.set_value(self.x0[5],agent['state'].v)

        self.initial_agent = agent

    def add_ey_constraints(self):
        for k in range(self.N+1):
            self.opti.subject_to(self.x[3,k] <= self.ey_lim) # ey
            self.opti.subject_to(self.x[3,k] >= -self.ey_lim) # ey
                    
    def add_input_rate_constraints(self):
        for k in range(self.N):
            if k == 0:
                self.opti.subject_to((self.u[0,k] - self.u_prev[0]) >= -self.dt*self.jerk_limit)
                self.opti.subject_to((self.u[0,k] - self.u_prev[0]) <= self.dt*self.jerk_limit)
                self.opti.subject_to((self.u[1,k] - self.u_prev[1]) >= -self.dt*self.steering_rate_limit)
                self.opti.subject_to((self.u[1,k] - self.u_prev[1]) <= self.dt*self.steering_rate_limit)
            else:
                self.opti.subject_to((self.u[0,k] - self.u[0,k-1]) >= -self.dt*self.jerk_limit)
                self.opti.subject_to((self.u[0,k] - self.u[0,k-1]) <= self.dt*self.jerk_limit)
                self.opti.subject_to((self.u[1,k] - self.u[1,k-1]) >= -self.dt*self.steering_rate_limit)
                self.opti.subject_to((self.u[1,k] - self.u[1,k-1]) <= self.dt*self.steering_rate_limit)

    def add_state_and_input_constraints(self):
        for k in range(self.N):
            self.opti.subject_to(self.x[5,k] >= self.v_min)
            self.opti.subject_to(self.x[5,k] <= self.v_max)
            self.opti.subject_to(self.u[0,k] >= self.a_min)
            self.opti.subject_to(self.u[0,k] <= self.a_max)
            self.opti.subject_to(self.u[1,k] >= -self.max_steering)
            self.opti.subject_to(self.u[1,k] <= self.max_steering)

    def set_reference(self,ref):
        self.ref = ref

    def get_xN(self,x,mode='casadi',include_route=True):
        if mode == 'casadi':
            ego_term = ca.vertcat(x[2,self.N],x[5,self.N]) # s_N, v_N
            j = self.pred_ind[0]
            tv_term = ca.vertcat(self.raw_preds[:,self.nx*j+2],self.raw_preds[:,self.nx*j+5]) # s_N, v_N
            if include_route:  
                ego_term = ca.vertcat(ego_term, route_encoding(self.routes)[self.ind])
                tv_term = ca.vertcat(tv_term, route_encoding(self.routes)[j])
            else:
                # Scenario encoding
                ego_term = ca.vertcat(ego_term, scenario_encoding(self.routes)[self.ind])
                tv_term = ca.vertcat(tv_term, scenario_encoding(self.routes)[j])
            x_N = ca.vertcat(tv_term, ego_term-tv_term) # [TV_state, ego_state - TV_state]
            
        else:
            ego_term = np.array([[x[2,-1]],[x[5,-1]]])
            j = self.pred_ind[0]
            tv_term = np.array([[self.raw_preds_np[0,self.nx*j+2]],[self.raw_preds_np[0,self.nx*j+5]]])
            if include_route:
            
                ego_term = np.concatenate((ego_term, np.array([[route_encoding(self.routes)[self.ind]]])))
                tv_term = np.concatenate((tv_term, np.array([[route_encoding(self.routes)[j]]])))
            else:
                #scenario encoding
                ego_term = np.concatenate((ego_term, np.array([[scenario_encoding(self.routes)[self.ind]]])))
                tv_term = np.concatenate((tv_term, np.array([[scenario_encoding(self.routes)[j]]])))
            
            x_N = np.concatenate((tv_term, ego_term-tv_term)) # [TV_state, ego_state - TV_state]
        return x_N
    
    def CAV_utility(self):
        sum_cost = 0

        # Tracking
        for k in range(self.N+1):
            if k < self.N:
                sum_cost += 0.05 * (self.u[0,k]**2 + self.u[1,k]**2)
            sum_cost += self.x[4,k]**2 # epsi
            sum_cost += self.x[3,k]**2 # ey

        # cost2go
        if self.use_NN_cost2go:
            x_N = self.get_xN(self.x, include_route=self.NN_config['include_route'])
            sum_cost -= self.V_ca(self.feature_cov_inv @ (x_N - self.feature_mean))*float(self.target_cov) + float(self.target_mean)
        else:
            # Terminal tracking only
            sum_cost -= (self.x[2,self.N]-self.x[2,0]) # -s_N - s_0 (local objective if not proposed)
        return sum_cost
    
    def cost_function(self):
        return self.CAV_utility()
    
    def value_function(self,x_N):
        x_norm = sp.linalg.solve(np.real(sp.linalg.sqrtm(self.feature_cov)), x_N.flatten() - self.feature_mean, assume_a='pos')  # Input normalization
        V = float(float(self.mlp(torch.tensor(x_norm.reshape((1,-1))).cuda()).cpu())*self.target_cov + self.target_mean) # Target unnormalization
        return V
    
    def solve(self,x_sol_prev=None, u_sol_prev = None):
        try:       
            self.opti.minimize(self.cost_function()) 
            if x_sol_prev is not None:
                # Use previous MPC solution as the initial guess      
                self.opti.set_initial(self.x, x_sol_prev)
                self.opti.set_initial(self.u, u_sol_prev)
                
            t_start = time.time()
            self.sol = self.opti.solve()
            t_end = time.time()
            self.x_sol_prev = self.sol.value(self.x)
            if self.use_NN_cost2go:
                x_N = self.get_xN(self.sol.value(self.x),include_route=self.NN_config['include_route'],mode='numpy')
                print(x_N)
                print(self.value_function(x_N))
            self.solve_time = t_end - t_start
            is_opt = True
            return (self.sol.value(self.x), self.sol.value(self.u), is_opt)
        except:
            print('NLP SOLVE FAILED'.center(80,'*'))
         
            is_opt = False        
            return (None, None, is_opt)
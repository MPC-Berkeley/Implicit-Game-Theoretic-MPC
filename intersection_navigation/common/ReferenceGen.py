from kinematic_bicycle_model import KinematicBicycleModel
from VehicleAction import VehicleAction
from VehicleState import VehicleState
from VehicleReference import VehicleReference
import numpy as np
import copy

class ReferenceGenerator():
    def __init__(self,
                 N = 100,
                 dt = 0.1,
                 model = None,
                 env = None,
                 initial_state = None,
                 goals = None,
                 routes = None,
                 road_width = 10,
                 road_length = 50,
                 radius = 2.8,
                 target_velocity = 2.0,
                 fillet_radius=2.8,
                 mode = 'cartesian'
                 ):
        
        self.N = N
        self.dt = dt
        if model is None:
            self.model = KinematicBicycleModel(l_r=4.47/2,l_f=4.47/2,width=2,dt=dt)
        else:
            self.model = model
        self.mode = mode
        self.road_width = road_width
        self.road_length = road_length
        self.ca_radius = radius
        self.target_velocity = target_velocity
        self.fillet_radius = fillet_radius
        turn_types = {'left': ['12','23','34','41'], 'right': ['14','21','32','43'], 'straight': ['13','24','31','42']}
        self.reference = self.generate_reference(initial_state,goals,turn_types,routes)
        

    def generate_reference(self,initial_state,goals,turn_types,routes):
        reference = []
        road_width = self.road_width
        road_length = self.road_length

        def project_point_onto_circle(point, circle_center, circle_radius):
            # Compute the vector from the circle center to the point
            vector = np.array(point) - np.array(circle_center)

            # Normalize the vector
            normalized_vector = vector / np.linalg.norm(vector)

            # Scale the vector to the circle radius
            scaled_vector = normalized_vector * circle_radius

            # Compute the projected point
            projected_point = circle_center + scaled_vector

            return projected_point
        def get_circle_params(route,agent,road_width,road_length):
            if route == '12' or route == '21':
                h = (road_length - road_width)/2
                k = road_width
                if route == '12':
                    r = k - agent['state'].y 
                elif route == '21':
                    h,k = ((road_length - road_width)/2-self.fillet_radius,road_width+self.fillet_radius)
                    r = self.fillet_radius + self.ca_radius
            if route == '14' or route == '41':
                h = (road_length - road_width)/2
                k = 0
                if route == '14':
                    h,k = ((road_length - road_width)/2-self.fillet_radius,0-self.fillet_radius)
                    r = self.fillet_radius + self.ca_radius
                elif route == '41':
                    r = agent['state'].x - h
            if route == '23' or route == '32':
                h = (road_length + road_width)/2
                k = road_width
                if route == '23':
                    r = h - agent['state'].x
                elif route == '32':
                    h,k = ((road_length + road_width)/2+self.fillet_radius,road_width+self.fillet_radius)
                    r = self.fillet_radius + self.ca_radius
            if route == '34' or route == '43':
                h = (road_length + road_width)/2
                k = 0
                if route == '34':
                    r = agent['state'].y
                elif route == '43':
                    h,k = ((road_length + road_width)/2+self.fillet_radius,0-self.fillet_radius)
                    r = self.fillet_radius + self.ca_radius
            return (h,k,r)
        def check_in_intersection(route,state,center):
            (h,k) = center
            if route in ['12','21']:
                if state.x >= h and state.y <= k:
                    return True
                else:
                    return False              
            elif route in ['14','41']:
                if state.x >= h and state.y >= k:
                    return True
                else:
                    return False
            elif route in ['23','32']:
                if state.x <= h and state.y <= k:
                    return True
                else:
                    return False
            elif route in ['34','43']:
                if state.x <= h and state.y >= k:
                    return True
                else:
                    return False
            else:
                raise BaseException('Invalid route')
            
        for i, agent in enumerate(initial_state):
            ref = []
            state_copy = copy.deepcopy(agent['state'])
            state_copy.v = self.target_velocity
            s = 0
            curvature = 0
            state_copy = VehicleReference({'x':state_copy.x,'y':state_copy.y,'heading':abs(state_copy.heading) if routes[i] in ['32','41'] else state_copy.heading,'v':state_copy.v,'K':curvature,'s':s,'ey':0,'epsi':0})
            ref.append(state_copy)
  
            if routes[i] in turn_types['straight']:
                for t in range(self.N):
                    # Straight line trajectory
                    new_state = self.model(state_copy,VehicleAction({'a':0,'df':0}))
                    if self.mode == 'frenet':
                        s += state_copy.v*self.dt
                        new_state_copy = VehicleReference({'x':new_state.x,'y':new_state.y,'heading':new_state.heading,'v':new_state.v,'K':curvature,'s':s,'ey':0,'epsi':0})
                        state_copy = new_state_copy
                        ref.append(new_state_copy)
                    else:
                        state_copy = new_state
                        ref.append(new_state)

            elif routes[i] in turn_types['left'] or routes[i] in turn_types['right']:
            # Left turn
                # Center of circle
                (h,k,r) = get_circle_params(routes[i],agent,road_width,road_length)
                pass_intersection = False
                for t in range(self.N):
                    if check_in_intersection(routes[i],state_copy,(h,k)):
                        if routes[i] == '12' or routes[i]=='23' or routes[i] == '41' or routes[i]=='34':
                            curvature = 1/r
                        else:
                            curvature = -1/r
                        pass_intersection = True
                        weight = 0.7 if routes[i] in turn_types['right'] else 0.9
                        state_copy.v = self.target_velocity*weight
                        new_state = self.model(state_copy,VehicleAction({'a':0,'df':0}))
                        
                        # Projection onto the circle
                        [x,y] = project_point_onto_circle([new_state.x, new_state.y], [h,k], r)

                        if routes[i] in turn_types['left']:
                            if (x-h) >0 and (y-k) >0 : # First quadrant
                                theta = np.arctan((y-k)/(x-h))
                                psi = np.pi/2 + abs(theta)
                            elif (x-h) <0 and (y-k) >0: # Second quadrant
                                theta = np.arctan((y-k)/(x-h))
                                assert theta < 0
                                psi = (np.pi/2 - abs(theta)) + np.pi
                            elif (x-h) <0 and (y-k) <0: # Third quadrant
                                theta = np.arctan((y-k)/(x-h))
                                assert theta > 0
                                psi = -(np.pi/2 - abs(theta))
                            elif (x-h) > 0 and (y-k) <0: # Fourth quadrant
                                theta = np.arctan((y-k)/(x-h))
                                psi = np.pi/2 - abs(theta)
                        else: # Right turn
                            psi = np.arctan2(-(x-h), (y-k))
                        
                        new_state = VehicleState({'x':x,'y':y,'heading':psi,'v':self.target_velocity*weight})                        
                        if not check_in_intersection(routes[i],new_state,(h,k)):
                            # Set heading to the goal heading
                            state_copy.heading = abs(goals[i].heading) if routes[i] in ['32','41'] else goals[i].heading
                            new_state = self.model(state_copy,VehicleAction({'a':0,'df':0}))
                    else:
                        curvature = 0 # Straight line
                        if pass_intersection:
                            state_copy.heading = abs(goals[i].heading) if routes[i] in ['32','41'] else goals[i].heading
                        state_copy.v = self.target_velocity
                        new_state = self.model(state_copy,VehicleAction({'a':0,'df':0}))

                    if self.mode == 'frenet':
                        s += np.sqrt((new_state.x - state_copy.x)**2 + (new_state.y - state_copy.y)**2 )
                        new_state_copy = VehicleReference({'x':new_state.x,'y':new_state.y,'heading':new_state.heading,'v':new_state.v,'K':curvature,'s':s,'ey':0,'epsi':0})
                        state_copy = new_state_copy
                        ref.append(new_state_copy)
                    else:
                        state_copy = new_state
                        ref.append(new_state)
            else:
                raise BaseException('Invalid route') 
            reference.append(ref)
        return reference
    
    def get_reference(self,N,initial_states = None,output_type=np.ndarray):
        if self.mode == 'cartesian':
            nx = 4
        else:
            nx = 6
        ref_arr = self.state2array()
        ref_arr_pp = np.zeros((nx*len(self.reference),N+1))
        if N == self.N:
            if output_type is dict:
                pass
                return NotImplementedError
            else:   
                return ref_arr_pp
        else:
            # Assume N < self.N:
            for i in range(len(self.reference)):
                # Projection of the current initial_states onto the new reference
                ind = np.argmin( (ref_arr[i*nx,:] - initial_states[i]['state'].x)**2 + (ref_arr[i*nx+1,:] - initial_states[i]['state'].y)**2)
                end = np.min([ind+N+1,self.N+1])
                ref_arr_pp[i*nx+0:i*nx+nx,:] = ref_arr[i*nx+0:i*nx+nx,ind:end]
            if output_type is dict:
                if nx ==4:
                    ref_dict_list = []
                    for i in range(len(self.reference)):
                        ref_dict_list.append({'x':ref_arr_pp[i*nx+0,:],'y':ref_arr_pp[i*nx+1,:],'heading':ref_arr_pp[i*nx+2,:],'v':ref_arr_pp[i*nx+3,:]})
                    return ref_dict_list
                else:
                    ref_dict_list = []
                    for i in range(len(self.reference)):
                        ref_dict_list.append({'x':ref_arr_pp[i*nx+0,:],'y':ref_arr_pp[i*nx+1,:],'heading':ref_arr_pp[i*nx+2,:],'v':ref_arr_pp[i*nx+3,:],'s':ref_arr_pp[i*nx+4,:],'K':ref_arr_pp[i*nx+5,:]})
                    return ref_dict_list
            else:   
                return ref_arr_pp
    def state2array(self):
        if self.mode == 'cartesian':
            nx = 4
            ref_array = np.zeros((nx*len(self.reference),self.N+1))
            for i,ref in enumerate(self.reference):
                for k,state in enumerate(ref):
                    ref_array[i*nx+0:i*nx+nx,k] = np.array([state.x,state.y,state.heading,state.v])
            return ref_array
        else:
            nx = 6
            ref_array = np.zeros((nx*len(self.reference),self.N+1))
            for i,ref in enumerate(self.reference):
                for k,state in enumerate(ref):
                    ref_array[i*nx+0:i*nx+nx,k] = np.array([state.x,state.y,state.heading,state.v,state.s,state.K])
            return ref_array

import numpy as np
import casadi as ca
import math as m
from typing import List
import math
import matplotlib.ticker as tkr
import polytope as pt
import torch
import copy
from VehicleAction import VehicleAction
from VehicleReference import VehicleReference
import torch as th
import random
import pickle
import scipy as sp
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import warnings
import polytope as pt
import matplotlib as mpl


class SceneHistoryBuffer:
    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.buffer = []
        self.buffer_index = 0

    def add(self, scene):
        if len(self.buffer) < self.buffer_size:
            self.buffer.append(scene)
        else:
            self.buffer[self.buffer_index] = scene
        self.buffer_index = (self.buffer_index + 1) % self.buffer_size

    def get(self):
        if len(self.buffer) == self.buffer_size:
            return self.buffer
        else:
            return [self.buffer[0]]*(self.buffer_size-len(self.buffer)) + self.buffer

def successor_set(A,B,X,U):
    if isinstance(X,pt.Polytope):
        V1 = pt.extreme(X)
    else:
        V1 = X
    
    if isinstance(U,pt.Polytope):
        V2 = pt.extreme(U)
    else:
        V2 = U
    x = pt.qhull((A@V1.T).T)
    temp = (B@V2.T).T
    eps = 1e-4
    temp = np.vstack([temp,temp,temp,temp])
    temp += np.array([[eps,eps,0],[eps,eps,0],[-eps,-eps,0],[-eps,-eps,0],[eps,-eps,0],[eps,-eps,0],[-eps,eps,0],[-eps,eps,0]])
    y = pt.qhull(temp)
    return minkowski_sum(x,y)

def compute_reachable_set(A,B,X,U,x0,N):
    R = x0  
    for k in range(N):
        R = successor_set(A,B,R,U).intersect(X)
    return R

def rotation_translation(x0,theta, h, w):
# % x0 is a 2x1 vector [x,y]' which represents the coordinates of the center of the vehicle
# % theta is the heading angle in radian
# % h is the total length of the vehicle
# % w is the width of the vehicle
    R = np.array([[np.cos(theta), -np.sin(theta)],
    [np.sin(theta), np.cos(theta)]])

    b = np.zeros((4,1))
    b[0] = h/2; b[1] = w/2; b[2] = h/2; b[3] = w/2
    A = np.vstack([R.T,-R.T])
    x0 = np.array(x0).reshape((len(x0),1))
    b_new = b + A@x0

    A_new = A

    return A_new, b_new

def scenario_index(route: List[str],m: int):
    turn_types = {'left': ['12','23','34','41'], 'right': ['14','21','32','43'], 'straight': ['13','24','31','42']}
    # For 2 cars
    vh1 = route[0][0]
    vh2 = route[1][0]
    if m == 1:
        # Scenario 1
        if route[0] == '42':
            vh1 = '0'
        elif route[1] == '42':
            vh2 = '0'
    elif m == 2:
        # Scenario 2
        if route[0] == '12' and route[1] == '41':
            vh1 = '5'
        elif route[1] == '12' and route[0] == '41':
            vh2 = '5'
    elif m == 3:
        # Scenario 3
        if (route[0] == '13' and route[1] == '42'):
            vh2 = '0'
        elif(route[1] == '13' and route[0] == '42'):
            vh1 = '0'
    if m < 4:
        if int(vh1) < int(vh2):  
            return [m,-m]
        else:
            return [-m,m]

    if m == 4:
        if route[0] in turn_types['left']:
            return [m,-m]
        else:
            return [-m,m]
    elif m == 5:
        if route[0] in turn_types['straight']:
            return [m,-m]
        else:
            return [-m,m]
    elif m == 6:
        if route[0] in turn_types['left']:
            return [m,-m]
        else:
            return [-m,m]
    elif m == 7:
        if int(vh1) < int(vh2):  
            return [m,-m]
        else:
            return [-m,m]    
    elif m == 8:
        if route[0] in turn_types['left']:
            return [m,-m]
        else:
            return [-m,m]
    else:
        raise ValueError('Scenario not found')    

def scenario_encoding(route: List):
    scenario1 = [{'13','23'},{'24','34'},{'31','41'},{'42','12'}] # Left and straight (same destination) [v1, 24 34]
    scenario2 = [{'12','41'},{'23','12'},{'34','23'},{'41','34'}] # Both left orgin==destination [v2, 34 41]
    scenario3 = [{'13','24'},{'24','31'},{'31','42'},{'42','13'}] # Straight and straight (cross) [v3, 13 24]
    scenario4 = [{'12','32'},{'23','43'},{'34','14'},{'41','21'}] # Left and right (same destination) [v4, 12 32]
    scenario5 = [{'13','43'},{'24','14'},{'31','21'},{'42','32'}] # Straight and right [v6, 13 43]
    scenario6 = [{'13','41'},{'24','12'},{'31','23'},{'42','34'}] # Left and straight (origin==destination) [v7, 13 41]
    scenario7 = [{'12','34'},{'23','41'},{'34','12'},{'41','23'}] # Left and left cross [v8, 12 34]
    scenario8 = [{'12','31'},{'23','42'},{'34','13'},{'41','24'}] # Left and straight (Not same dest) [v9, 12 31] 

    cur_route_set = {route[0],route[1]}
    if cur_route_set in scenario1:
        return scenario_index(route,1)
    elif cur_route_set in scenario2:
        return scenario_index(route,2)
    elif cur_route_set in scenario3:
        return scenario_index(route,3)
    elif cur_route_set in scenario4:
        return scenario_index(route,4)
    elif cur_route_set in scenario5:
        return scenario_index(route,5)
    elif cur_route_set in scenario6:
        return scenario_index(route,6)
    elif cur_route_set in scenario7:
        return scenario_index(route,7)
    elif cur_route_set in scenario8:
        return scenario_index(route,8)
    else:
        raise ValueError('Scenario not found')

def get_scenario_encoding(routes: List[str]):
    encoding = []
    for route in routes:
        encoding.append(scenario_encoding(route))
    return encoding

def get_route_from_scenario(scenario: int):
    route_list = [[{'13','23'},{'24','34'},{'31','41'},{'42','12'}],[{'12','41'},{'23','12'},{'34','23'},{'41','34'}],[{'13','24'},{'24','31'},{'31','42'},{'42','13'}],[{'12','32'},{'23','43'},{'34','14'},{'41','21'}],[{'13','43'},{'24','14'},{'31','21'},{'42','32'}],[{'13','41'},{'24','12'},{'31','23'},{'42','34'}],[{'12','34'},{'23','41'},{'34','12'},{'41','23'}],[{'12','31'},{'23','42'},{'34','13'},{'41','24'}]]
    return list(random.choice(route_list[scenario-1]))

def trajectory_graph(solutions):
    s_arr = []
    x_arr  = []
    y_arr =  []
    
    for sol in solutions:
        x_sol = sol[0]
        u_sol = sol[1]
        s_arr.append(x_sol[2,:])
        x_arr.append(x_sol[0,:])
        y_arr.append(x_sol[1,:])
    
    # Compute distance
    d_arr = np.sqrt((x_arr[0] - x_arr[1])**2 + (y_arr[0] - y_arr[1])**2)
    print(d_arr)
    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(s_arr[0],s_arr[1],'-*b',label='Trajectory')
    plt.xlabel('$s_1$')
    plt.ylabel('$s_2$')
    s1min =  np.asin(((50-19.3) - (2.8*2 + 22.1))/(8.6)) * 8.6 + 19.3
    y = 8.6 - (8.6 - 8.6*np.cos(np.asin(((50-19.3) - (2.8*2 + 22.1))/8.6)))
    s2max = 61.4/2 - y

    r = 2.8
    R = 8.6 + r
    theta = np.asin((R-r)/(R+r))
    s1max = 19.3 + 8.6*theta

    y_temp = np.cos(theta)*(R+r)
    y = np.sqrt((R+r)**2 - (R-r)**2)
    assert y == y_temp
    s2min = 61.4/2 - y
    print(s1min,s1max)
    print(s2min,s2max)
    # Plot a box rectangle
    plt.plot([s1min,s1max,s1max,s1min,s1min],[s2min,s2min,s2max,s2max,s2min],'--r',label='Deadlock Region')
    plt.legend()
    plt.show()

def compute_quadratic_form(x, Q, q, c,mode='torch'):
    if mode=='torch':
        # Quadratic term
        xQx = torch.einsum('ij,ijk,ik->i', x, Q, x)
        
        # Linear term
        qTx = torch.einsum('ij,ij->i', q, x)
        
        # Sum all terms
        result = xQx + qTx + c.squeeze()
    elif mode=='casadi':
        x = x.T
        # Quadratic term
        xQx = ca.mtimes(ca.mtimes(x.T,Q),x)
        
        # Linear term
        qTx = ca.mtimes(q,x)
        
        # Sum all terms
        result = xQx + qTx + c
    elif mode=='numpy':
        x = x.T
        # Quadratic term
        xQx = np.einsum('ij,ijk,ik->i', x, Q, x)
        
        # Linear term
        qTx = np.einsum('ij,ij->i', q, x)
        
        # Sum all terms
        result = xQx + qTx + c.squeeze()
    return result

def colored_line_between_pts(x, y, c, ax, **lc_kwargs):
    """
    Plot a line with a color specified between (x, y) points by a third value.

    It does this by creating a collection of line segments between each pair of
    neighboring points. The color of each segment is determined by the
    made up of two straight lines each connecting the current (x, y) point to the
    midpoints of the lines connecting the current point with its two neighbors.
    This creates a smooth line with no gaps between the line segments.

    Parameters
    ----------
    x, y : array-like
        The horizontal and vertical coordinates of the data points.
    c : array-like
        The color values, which should have a size one less than that of x and y.
    ax : Axes
        Axis object on which to plot the colored line.
    **lc_kwargs
        Any additional arguments to pass to matplotlib.collections.LineCollection
        constructor. This should not include the array keyword argument because
        that is set to the color argument. If provided, it will be overridden.

    Returns
    -------
    matplotlib.collections.LineCollection
        The generated line collection representing the colored line.
    """
    if "array" in lc_kwargs:
        warnings.warn('The provided "array" keyword argument will be overridden')

    # Check color array size (LineCollection still works, but values are unused)
    if len(c) != len(x) - 1:
        warnings.warn(
            "The c argument should have a length one less than the length of x and y. "
            "If it has the same length, use the colored_line function instead."
        )

    ''' 
    Create a set of line segments so that we can color them individually
    This creates the points as an N x 1 x 2 array so that we can stack points
    together easily to get the segments. The segments array for line collection
    needs to be (numlines) x (points per line) x 2 (for x and y)
    '''
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    lc = LineCollection(segments, **lc_kwargs)

    # Set the values used for colormapping
    lc.set_array(c)

    return ax.add_collection(lc)

def plot_scene(ax,cur_agent, road_length=50,road_width=11.4,fillet_radius=2.8):
    w = 2
    l = 4.47
    ax.clear()  # Clear the axis before drawing new artists
    ax.add_patch(plt.Rectangle((0,(road_width-road_length)/2), road_length, road_length, color='gray'))
    patch1_width = (road_length - road_width) /2
    patch1_length = patch1_width - fillet_radius

    ax.add_patch(plt.Rectangle((0,(road_width-road_length)/2), patch1_width, patch1_length, color='white',linewidth=0))
    ax.add_patch(plt.Rectangle((0,(road_width-road_length)/2), patch1_length, patch1_width, color='white',linewidth=0))
    ax.add_patch(plt.Circle(((road_length - road_width)/2-fillet_radius,0-fillet_radius),fillet_radius, color='white',linewidth=0))

    ax.add_patch(plt.Rectangle((0,(road_length+road_width)/2), patch1_width, -patch1_length, color='white',linewidth=0))
    ax.add_patch(plt.Rectangle((0,(road_length+road_width)/2), patch1_length, -patch1_width, color='white',linewidth=0))
    ax.add_patch(plt.Circle(((road_length - road_width)/2-fillet_radius,road_width+fillet_radius),fillet_radius, color='white',linewidth=0))
    
    ax.add_patch(plt.Rectangle((road_length,(road_length+road_width)/2), -patch1_width, -patch1_length, color='white',linewidth=0))
    ax.add_patch(plt.Rectangle((road_length,(road_length+road_width)/2), -patch1_length, -patch1_width, color='white',linewidth=0))
    ax.add_patch(plt.Circle(((road_length + road_width)/2+fillet_radius,road_width+fillet_radius),fillet_radius, color='white',linewidth=0))
    
    ax.add_patch(plt.Rectangle((road_length,(road_width-road_length)/2), -patch1_width, patch1_length, color='white',linewidth=0))
    ax.add_patch(plt.Rectangle((road_length,(road_width-road_length)/2), -patch1_length, patch1_width, color='white',linewidth=0))
    ax.add_patch(plt.Circle(((road_length + road_width)/2+fillet_radius,0-fillet_radius),fillet_radius, color='white',linewidth=0))

    for i in range(len(cur_agent)):
        color = 'green' if cur_agent[i]['type'] == 'CAV' else 'red'
        ax.add_patch(plt.Rectangle(transform_Rectangle(cur_agent[i]['state'].x, cur_agent[i]['state'].y,w,l/2, 0),angle=cur_agent[i]['state'].heading*180/np.pi,rotation_point='center',width=l,height=w,color=color))
        ax.arrow(cur_agent[i]['state'].x,cur_agent[i]['state'].y, 2*np.cos(cur_agent[i]['state'].heading), 2*np.sin(cur_agent[i]['state'].heading), head_width=0.5, head_length=0.5, fc='k', ec='k')

    ax.set_xlim(0,road_length)
    ax.set_ylim(road_width/2-road_length/2,road_length/2+road_width/2)
    ax.set_aspect('equal')

def share_motion_forecasts(predictor,preds,cav_sols,cav_index_opt,refs_dict,routes,config):
    K_arr = config['K']
    preds4CAV = preds
    for m in range(len(cav_sols)):
        temp = []
        for k in range(1,config['N']+1):
            #convert mpc solution to VehicleReference
            temp.append(VehicleReference({'x':cav_sols[m][0][0,k],'y':cav_sols[m][0][1,k],'v':cav_sols[m][0][5,k],'s':cav_sols[m][0][2,k],'heading':cav_sols[m][0][6,k],'ey':cav_sols[m][0][3,k],'epsi':cav_sols[m][0][4,k],'K':K_arr[cav_index_opt[m]]}))
        cav_pred_aug = predictor.predict(agents=[{'type':'CAV','state': VehicleReference({'x':cav_sols[j][0][0,-1],'y':cav_sols[j][0][1,-1],'v':cav_sols[j][0][5,-1],'s':cav_sols[j][0][2,-1],'heading':cav_sols[j][0][6,-1],'ey':cav_sols[j][0][3,-1],'epsi':cav_sols[j][0][4,-1],'K':K_arr[cav_index_opt[j]]})} for j in range(len(cav_sols))],agent_cur_inputs=[VehicleAction({'a':cav_sols[j][1][0,-1],'df':cav_sols[j][1][1,-1]}) for j in range(len(cav_sols))],routes=[routes[j] for j in cav_index_opt],refs=[refs_dict[j] for j in cav_index_opt])
        if cav_pred_aug[m][1].v > 5: 
            cav_pred_aug = predictor.predict(agents=[{'type':'CAV','state': VehicleReference({'x':cav_sols[j][0][0,-1],'y':cav_sols[j][0][1,-1],'v':cav_sols[j][0][5,-1],'s':cav_sols[j][0][2,-1],'heading':cav_sols[j][0][6,-1],'ey':cav_sols[j][0][3,-1],'epsi':cav_sols[j][0][4,-1],'K':K_arr[cav_index_opt[j]]})} for j in range(len(cav_sols))],agent_cur_inputs=[VehicleAction({'a':cav_sols[j][1][0,-1] if j!=m else 0,'df':cav_sols[j][1][1,-1]}) for j in range(len(cav_sols))],routes=[routes[j] for j in cav_index_opt],refs=[refs_dict[j] for j in cav_index_opt])
        temp.append(cav_pred_aug[m][1])
        preds4CAV[cav_index_opt[m]] = temp
    return preds4CAV

def augment_prev_sol(data,model,K):
    x_sol_prev = data[0]
    u_sol_prev = data[1]

    next_state = model(VehicleReference({'s':x_sol_prev[2,-1],'ey':x_sol_prev[3,-1],'epsi':x_sol_prev[4,-1],'v':x_sol_prev[5,-1],'K':K,'x':x_sol_prev[0,-1], 'y':x_sol_prev[1,-1], 'heading': x_sol_prev[6,-1]}),VehicleAction({'a':u_sol_prev[0,-1],'df':u_sol_prev[1,-1]}))
    if next_state.v > 5: 
        next_state = model(VehicleReference({'s':x_sol_prev[2,-1],'ey':x_sol_prev[3,-1],'epsi':x_sol_prev[4,-1],'v':x_sol_prev[5,-1],'K':K,'x':x_sol_prev[0,-1], 'y':x_sol_prev[1,-1], 'heading': x_sol_prev[6,-1]}),VehicleAction({'a':0,'df':u_sol_prev[1,-1]}))
    x_sol_prev = np.hstack([x_sol_prev[:,1:], np.array([[next_state.x],[next_state.y],[next_state.s],[next_state.ey],[next_state.epsi],[np.clip(next_state.v,-1,5)],[next_state.heading]])])
    u_sol_prev = np.hstack([u_sol_prev[:,1:], u_sol_prev[:,[-1]]])
    return (x_sol_prev,u_sol_prev)

def filter_preds(preds_input, ego_id):
    preds = copy.deepcopy(preds_input)
    x = preds[ego_id][0].x
    y = preds[ego_id][0].y
    heading = preds[ego_id][0].heading
    for i, pred in enumerate(preds):
        if i != ego_id:
            dx = pred[0].x - x
            dy = pred[0].y - y

            dp = np.array([[dx], [dy]])

            dx_ = np.cos(heading)
            dy_ = np.sin(heading)
            dp_ = np.array([[dx_], [dy_]])
            # Dot product
            dot = np.dot(dp.T, dp_)
            # Sign of dot product
            if dot < 0: # Means the vehicle is not in the 90 degree cone in front of the ego vehicle
                for pred_ in preds[i]:
                    # Arbitrarily set x and y to 100000 (i.e. really far away from the ego vehicle)
                    pred_.x = -20
                    pred_.y = -20
    return preds

def label_smoothing(y_hot,a=0,K=15):
    return (1 - a) * y_hot + a / K

def route_encoding(routes: List):
    route_dict = {
        '12': 0, '13': 1, '14': 2,
        '21': 3, '23': 4, '24': 5,
        '31': 6, '32': 7, '34': 8,
        '41': 9, '42': 10, '43': 11
    }

    encoded_routes = [route_dict[route] for route in routes]
    return encoded_routes

def transform_Rectangle(x,y,vehicle_width,l_r,psi):
    s = np.sqrt((vehicle_width/2)**2 + (l_r)**2)
    x_new = x - s*np.cos(psi+np.arctan(vehicle_width/l_r))
    y_new = y - s*np.sin(psi+np.arctan(vehicle_width/l_r))
    xy = (x_new,y_new)
    xy = (x-l_r, y-vehicle_width/2)
    return xy    

def rotation_translation(x0,theta, h, w, mode='numpy'):
    '''
    x0 is a 2x1 vector [x,y]' which represents the coordinates of the center of the vehicle
    theta is the heading angle in radian
    h is the total length of the vehicle
    w is the width of the vehicle
    '''
    if mode=='numpy':
        R = np.array([[np.cos(theta), -np.sin(theta)],
        [np.sin(theta), np.cos(theta)]])

        b = np.zeros((4,1))
        b[0] = h/2; b[1] = w/2; b[2] = h/2; b[3] = w/2
        A = np.vstack([R.T,-R.T])
        x0 = np.array(x0).reshape((len(x0),1))
        b_new = b + A@x0

        A_new = A

        return A_new, b_new
    else:
        R = ca.vertcat(ca.horzcat(ca.cos(theta), -ca.sin(theta)), ca.horzcat(ca.sin(theta), ca.cos(theta)))

        b = ca.DM.zeros(4,1)
        b[0] = h/2; b[1] = w/2; b[2] = h/2; b[3] = w/2
        A = ca.vertcat(R.T,-R.T)
        x0 = ca.vertcat(*x0)
        b_new = b + A@x0

        A_new = A
        return A_new, b_new

def kinematic_bicycle_model(x,y,psi,v,delta_t,l_f,l_r,a,delta_f):
    '''
     x is position in longitudinal direction
     y is position in lateral direction
     psi is heading angle
     v is velocity (norm of velocity in x and y directions)
     delta_t is sampling time
     l_f is the length of the car from center of gravity to the front end
     l_r is the length of the car from center of gravity to the rear end
     a is acceleration which is control input
     delta_f is steering angle which is control input
     '''

    beta = np.arctan(((l_r/(l_f+l_r))*np.tan(delta_f)))
    x_new = x + delta_t*v*np.cos(psi+beta)
    y_new = y + delta_t*v*np.sin(psi+beta)
    psi_new = psi + delta_t*(v*np.cos(beta)/(l_r+l_f)*np.tan(delta_f))
    v_new = v + delta_t*a

    return x_new,y_new,psi_new,v_new

def make_ca_fun(s, x, y, psi, v):
    x_ca= ca.interpolant("f2gx", "linear", [s], x)
    y_ca= ca.interpolant("f2gy", "linear", [s], y)
    psi_ca= ca.interpolant("f2gpsi", "linear", [s], psi)
    v_ca= ca.interpolant("f2gv", "linear", [s], v)
    s_sym=ca.MX.sym("s",1)

    glob_fun=ca.Function("fx",[s_sym], [ca.vertcat(x_ca(s_sym), y_ca(s_sym), psi_ca(s_sym), v_ca(s_sym))])
    return glob_fun

def make_jac_fun(pos_fun):
    s_sym=ca.MX.sym("s",1)
    pos_jac=ca.jacobian(pos_fun(s_sym), s_sym)
    return ca.Function("pos_jac",[s_sym], [pos_jac]) 


def global2frenet(ref_path, x_cur, y_cur, psi_cur):
    s = ref_path['s']
    x = ref_path['x']
    y = ref_path['y']
    K = ref_path['K']
    psi = ref_path['heading']

    norm_array = (x-x_cur)**2+(y-y_cur)**2
    idx_min = np.argmin(norm_array)

    s_cur = s.item(idx_min)

    # Unsigned ey
    e_y_cur = np.sqrt(norm_array.item(idx_min))

    # ey sign
    delta_x = x.item(idx_min) - x_cur
    delta_y = y.item(idx_min) - y_cur
    delta_vec = np.array([[delta_x], [delta_y]])
    R = np.array([[np.cos(psi.item(idx_min)), np.sin(psi.item(idx_min))], [- np.sin(psi.item(idx_min)), np.cos(psi.item(idx_min))]])
    C = np.array([0,1]).reshape((1,-1))
    ey_dir = - np.sign(C @ R @ (delta_vec)) # - (delta_vec.T @ unit_vec) * unit_vec))

    # Signed ey
    e_y_cur = e_y_cur.flatten()[0] * ey_dir.flatten()[0]

    e_psi_cur = psi_cur - psi.item(idx_min)

    return s_cur, e_y_cur, e_psi_cur, idx_min

def curvatures4mpc(ref_path, idx_min, N_MPC, Ts, s_cur, vx_cur, ax_cur):
    s = ref_path['s']
    K = ref_path['K']
    K_array = np.ones((1,N_MPC))

    # s to K array
    mock_t_array = np.arange(N_MPC) * Ts
    mock_s_array = s_cur + vx_cur * mock_t_array + 0.5 * ax_cur * mock_t_array ** 2
    mock_s_array = s

    for ind in range(N_MPC):
        mock_s = mock_s_array.item(ind)
        s_err_array = np.abs(s - mock_s)
        idx_min = np.argmin(s_err_array)

        K_array[0, ind] = K.item(idx_min)

    K_array.reshape((1,-1))

    return K_array

def frenet2global(s_cur,ref,route,road_length,road_width,ca_radius,verbose=False):
    turn_types = {'left': ['12','23','34','41'], 'right': ['14','21','32','43'], 'straight': ['13','24','31','42']}
    ey = 0
    if route in turn_types['straight']:
        if abs(ref['x'][-1] - ref['x'][0]) < 1e-4:
            if route == '24':
                y0 = (road_length+road_width)/2
            elif route == '42':
                y0 = (road_width-road_length)/2
            else: 
                y0 = ref['y'][0]
            x_cur = ref['x'][0]
            y_cur = y0 + np.sign(ref['y'][-1] - ref['y'][0]) * s_cur
        elif abs(ref['y'][-1] - ref['y'][0]) < 1e-4:
            if route == '13':
                x0 = 0
            elif route == '31':
                x0 = road_length
            else: 
                x0 = ref['x'][0]
            x_cur = x0 + np.sign(ref['x'][-1] - ref['x'][0]) * s_cur
            y_cur = ref['y'][0]

    elif route in turn_types['left']:
        radius = max(abs(1/ref['K'][np.nonzero(ref['K'])]))
        if route == '12':
            x_cur = ca.if_else(s_cur < (road_length-road_width)/2, s_cur, ca.if_else(s_cur > (road_length-road_width)/2 + max(abs(1/ref['K'][np.nonzero(ref['K'])])) * np.pi/2, ref['x'][-1], (road_length-road_width)/2+ radius*(ca.sin((s_cur - (road_length-road_width)/2)/radius) )))  
            y_cur = ca.if_else(s_cur < (road_length-road_width)/2, ref['y'][0], ref['y'][0] + ca.if_else(s_cur > (road_length-road_width)/2 + max(abs(1/ref['K'][np.nonzero(ref['K'])])) * np.pi/2, s_cur - (road_length-road_width)/2 - (max(abs(1/ref['K'][np.nonzero(ref['K'])])) * np.pi/2) + radius, radius*(1-ca.cos((s_cur - (road_length-road_width)/2)/radius) )))
        elif route == '23':
            x_cur = ca.if_else(s_cur < (road_length-road_width)/2, ref['x'][0], ref['x'][0] + ca.if_else(s_cur > (road_length-road_width)/2 + max(abs(1/ref['K'][np.nonzero(ref['K'])])) * np.pi/2, s_cur - (road_length-road_width)/2 - (max(abs(1/ref['K'][np.nonzero(ref['K'])])) * np.pi/2) + radius, radius*(1-ca.cos((s_cur - (road_length-road_width)/2)/radius) )))
            y_cur = ca.if_else(s_cur < (road_length-road_width)/2, (road_length+road_width)/2 - s_cur, ca.if_else(s_cur > (road_length-road_width)/2 + max(abs(1/ref['K'][np.nonzero(ref['K'])])) * np.pi/2, ref['y'][-1], (road_length+road_width)/2 - (road_length-road_width)/2-radius*(ca.sin((s_cur - (road_length-road_width)/2)/radius) )))  
        elif route == '34':
            x_cur = ca.if_else(s_cur < (road_length-road_width)/2, road_length - s_cur, ca.if_else(s_cur > (road_length-road_width)/2 + max(abs(1/ref['K'][np.nonzero(ref['K'])])) * np.pi/2, ref['x'][-1], road_length - (road_length-road_width)/2 - radius*(ca.sin((s_cur - (road_length-road_width)/2)/radius) )))  
            y_cur = ca.if_else(s_cur < (road_length-road_width)/2, ref['y'][0], ref['y'][0] - ca.if_else(s_cur > (road_length-road_width)/2 + max(abs(1/ref['K'][np.nonzero(ref['K'])])) * np.pi/2, s_cur - (road_length-road_width)/2 - (max(abs(1/ref['K'][np.nonzero(ref['K'])])) * np.pi/2) + radius, radius*(1-ca.cos((s_cur - (road_length-road_width)/2)/radius) )))
        elif route == '41':
            x_cur = ca.if_else(s_cur < (road_length-road_width)/2, ref['x'][0], ref['x'][0] - ca.if_else(s_cur > (road_length-road_width)/2 + max(abs(1/ref['K'][np.nonzero(ref['K'])])) * np.pi/2, s_cur - (road_length-road_width)/2 - (max(abs(1/ref['K'][np.nonzero(ref['K'])])) * np.pi/2) + radius, radius*(1-ca.cos((s_cur - (road_length-road_width)/2)/radius) )))
            y_cur = ca.if_else(s_cur < (road_length-road_width)/2, (road_width-road_length)/2 + s_cur, ca.if_else(s_cur > (road_length-road_width)/2 + max(abs(1/ref['K'][np.nonzero(ref['K'])])) * np.pi/2, ref['y'][-1], (road_width-road_length)/2 + (road_length-road_width)/2+radius*(ca.sin((s_cur - (road_length-road_width)/2)/radius) )))  

    elif route in turn_types['right']:
        radius = max(abs(1/ref['K'][np.nonzero(ref['K'])]))
        if route == '14':
            x_cur = ca.if_else(s_cur < (road_length-road_width)/2- (road_width - ca_radius), s_cur, ca.if_else(s_cur > (road_length-road_width)/2 - (road_width - ca_radius) + max(abs(1/ref['K'][np.nonzero(ref['K'])])) * np.pi/2, ref['x'][-1], (road_length-road_width)/2 - (road_width - ca_radius) + radius*(ca.sin((s_cur - (road_length-road_width)/2 + (road_width - ca_radius))/radius) )))  
            y_cur = ca.if_else(s_cur < (road_length-road_width)/2- (road_width - ca_radius), ref['y'][0], ref['y'][0] - ca.if_else(s_cur > (road_length-road_width)/2 - (road_width - ca_radius) + max(abs(1/ref['K'][np.nonzero(ref['K'])])) * np.pi/2, s_cur - (road_length-road_width)/2 + (road_width - ca_radius)- (max(abs(1/ref['K'][np.nonzero(ref['K'])])) * np.pi/2) + radius, radius*(1-ca.cos((s_cur - (road_length-road_width)/2 + (road_width - ca_radius))/radius) )))
        elif route == '21':
            x_cur = ca.if_else(s_cur < (road_length-road_width)/2- (road_width - ca_radius), ref['x'][0], ref['x'][0] - ca.if_else(s_cur > (road_length-road_width)/2 - (road_width - ca_radius) + max(abs(1/ref['K'][np.nonzero(ref['K'])])) * np.pi/2, s_cur - (road_length-road_width)/2 + (road_width - ca_radius)- (max(abs(1/ref['K'][np.nonzero(ref['K'])])) * np.pi/2) + radius, radius*(1-ca.cos((s_cur - (road_length-road_width)/2 + (road_width - ca_radius))/radius) )))
            y_cur = ca.if_else(s_cur < (road_length-road_width)/2- (road_width - ca_radius), (road_length+road_width)/2 - s_cur, ca.if_else(s_cur > (road_length-road_width)/2 - (road_width - ca_radius) + max(abs(1/ref['K'][np.nonzero(ref['K'])])) * np.pi/2, ref['y'][-1], (road_length+road_width)/2 - (road_length-road_width)/2 + (road_width - ca_radius) -radius*(ca.sin((s_cur - (road_length-road_width)/2 + (road_width - ca_radius))/radius) )))  
        elif route == '32':
            x_cur = ca.if_else(s_cur < (road_length-road_width)/2- (road_width - ca_radius), road_length - s_cur, ca.if_else(s_cur > (road_length-road_width)/2 - (road_width - ca_radius) + max(abs(1/ref['K'][np.nonzero(ref['K'])])) * np.pi/2, ref['x'][-1], road_length - (road_length-road_width)/2 + (road_width - ca_radius) - radius*(ca.sin((s_cur - (road_length-road_width)/2 + (road_width - ca_radius))/radius) )))  
            y_cur = ca.if_else(s_cur < (road_length-road_width)/2- (road_width - ca_radius), ref['y'][0], ref['y'][0] + ca.if_else(s_cur > (road_length-road_width)/2 - (road_width - ca_radius) + max(abs(1/ref['K'][np.nonzero(ref['K'])])) * np.pi/2, s_cur - (road_length-road_width)/2 + (road_width - ca_radius) - (max(abs(1/ref['K'][np.nonzero(ref['K'])])) * np.pi/2) + radius, radius*(1-ca.cos((s_cur - (road_length-road_width)/2 + (road_width - ca_radius))/radius) )))
        elif route == '43':     
            x_cur = ca.if_else(s_cur < (road_length-road_width)/2- (road_width - ca_radius), ref['x'][0], ref['x'][0] + ca.if_else(s_cur > (road_length-road_width)/2 - (road_width - ca_radius) + max(abs(1/ref['K'][np.nonzero(ref['K'])])) * np.pi/2, s_cur - (road_length-road_width)/2 + (road_width - ca_radius) - (max(abs(1/ref['K'][np.nonzero(ref['K'])])) * np.pi/2) + radius, radius*(1-ca.cos((s_cur - (road_length-road_width)/2 + (road_width - ca_radius))/radius) )))
            y_cur = ca.if_else(s_cur < (road_length-road_width)/2- (road_width - ca_radius), (road_width-road_length)/2 + s_cur, ca.if_else(s_cur > (road_length-road_width)/2 - (road_width - ca_radius) + max(abs(1/ref['K'][np.nonzero(ref['K'])])) * np.pi/2, ref['y'][-1], (road_width-road_length)/2 + (road_length-road_width)/2 - (road_width - ca_radius) + radius*(ca.sin((s_cur - (road_length-road_width)/2 + (road_width - ca_radius))/radius) )))  
    else:
        raise ValueError('Invalid route')
    return  np.vstack((x_cur, y_cur))

def Cinf(A, B, Xset, Uset):

    Omega = Xset
    k = 0
    Omegap = precursor(Omega, A, Uset, B).intersect(Omega)
    while not Omegap == Omega:
        k += 1
        Omega = Omegap
        Omegap = precursor(Omega, A, Uset, B).intersect(Omega)
    return Omegap

def precursor(Xset, A, Uset=pt.Polytope(), B=np.array([])):
        if not B.any():
            return pt.Polytope(Xset.A @ A, Xset.b)
        else:
            tmp  = minkowski_sum( Xset, pt.extreme(Uset) @ -B.T )
        return pt.Polytope(tmp.A @ A, tmp.b)

def minkowski_sum(X, Y):
    '''
    Minkowski sum between two polytopes based on
    vertex enumeration. So, it's not fast for the
    high dimensional polytopes with lots of vertices
    '''
    V_sum = []
    if isinstance(X, pt.Polytope):
        V1 = pt.extreme(X)
    else:
        # Assuming vertices are in (N x d) shape. N # of vertices, d dimension
        V1 = X

    if isinstance(Y, pt.Polytope):
        V2 = pt.extreme(Y)
    else:
        V2 = Y

    for i in range(V1.shape[0]):
        for j in range(V2.shape[0]):
            V_sum.append(V1[i,:] + V2[j,:])
    return pt.qhull(np.asarray(V_sum))

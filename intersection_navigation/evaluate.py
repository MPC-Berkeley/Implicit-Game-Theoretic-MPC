import numpy as np
import datetime
import casadi as ca
import csv
import copy
import os, pickle
import argparse
import yaml
from mpc import MPC_Planner
import sys
# Adding Folder_2/subfolder to the system path
sys.path.insert(0, os.getcwd()+'/common')
from VehicleState import VehicleState
from VehicleReference import VehicleReference
from VehicleAction import VehicleAction
from animate import animate_trajectory
from Goals import Goals
from ReferenceGen import ReferenceGenerator
from constant_acceleration_model import ConstantAccelerationModel
from kinematic_bicycle_model_frenet import KinematicBicycleModelFrenet
from utils import filter_preds, augment_prev_sol, share_motion_forecasts, get_route_from_scenario, scenario_encoding

def get_scenario_config(sc):
    return os.getcwd() + '/game_theoretic_NN/configs/sc' + str(sc)+ '_config.yaml'

def main(args):
    # Load environment configuration
    with open(os.getcwd() + '/common/fourwayint.yaml') as f:
        env_config = yaml.load(f, Loader=yaml.FullLoader)

    # Load policy configuration
    with open('mpc.yaml') as f:
        policy_config = yaml.load(f, Loader=yaml.FullLoader)

    seed = 2026
    env_initial_regions = {1,2,3,4} # W, N, E, S
    env_goals = {'1': {2,3,4}, '2': {1,3,4}, '3': {1,2,4}, '4': {1,2,3}}
    
    # Load Environment parameters
    road_width = env_config['road_width']
    road_length = env_config['road_length']
    ca_radius = env_config['ca_radius']
    dt = env_config['dt']
    v0 = env_config['v0']
    v_des = env_config['v_des']
    M = env_config['num_agents']

    fillet_radius = road_width - ca_radius
    max_start = (road_length-road_width)/2 - fillet_radius

    goals_states = {'1': VehicleState({'x':0,'y':road_width-ca_radius,'heading':-np.pi,'v':v_des}), 
                    '2':VehicleState({'x':road_length/2+road_width/2-ca_radius,'y':road_length/2+road_width/2,'heading':np.pi/2,'v':v_des}), 
                    '3':VehicleState({'x':road_length,'y':ca_radius,'heading':0,'v':v_des}),
                    '4':VehicleState({'x':road_length/2-road_width/2+ca_radius,'y':(road_width-road_length)/2,'heading':-np.pi/2,'v':v_des})}

    rng = np.random.default_rng(seed=seed)
    np.random.seed(seed)
    
    env_initial_states = {'1': VehicleState({'x':0,'y':ca_radius,'heading':0,'v':v0}), 
                        '2':VehicleState({'x':road_length/2-road_width/2+ca_radius,'y':road_length/2+road_width/2,'heading':-np.pi/2,'v':v0}), 
                        '3':VehicleState({'x':road_length,'y':road_width-ca_radius,'heading':-np.pi,'v':v0}), 
                        '4':VehicleState({'x':road_length/2+road_width/2-ca_radius,'y':(road_width-road_length)/2,'heading':np.pi/2,'v':v0})}

    num_deadlocks = 0
    timenow = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Policy parameter
    if policy_config['type'] == 'MPC':
        N = policy_config['N']
        if args.eval_mode == 'gt_mpc':
            use_NN_cost2go= True
            ca_type = policy_config['collision_avoidance_type']
        else:
            use_NN_cost2go= False
            ca_type = policy_config['collision_avoidance_type']
        if 'constant_acceleration' in policy_config['prediction_type']:
            predictor = ConstantAccelerationModel(N=N,dt=dt)
        elif 'constant_speed' in policy_config['prediction_type']:
            predictor = ConstantAccelerationModel(N=N,dt=dt,constant_speed=True)
        else:
            raise ValueError('Invalid prediction type')
        
    # Simulation parameters
    T_sim = 15 # Seconds
    M_sim = int(T_sim/dt)

    for it in range(args.num_samples):
        print(f'Running iteration: {it+1}')

        # Scenario Random Generation
        agent_initial_states = {'1': VehicleState({'x':rng.random()*max_start,'y':ca_radius,'heading':0,'v':v0}), 
                                '2':VehicleState({'x':road_length/2-road_width/2+ca_radius,'y':road_length/2+road_width/2-rng.random()*max_start,'heading':-np.pi/2,'v':v0}), 
                                '3':VehicleState({'x':road_length-rng.random()*max_start,'y':road_width-ca_radius,'heading':-np.pi,'v':v0}), 
                                '4':VehicleState({'x':road_length/2+road_width/2-ca_radius,'y':(road_width-road_length)/2+rng.random()*max_start,'heading':np.pi/2,'v':v0})}
        # Sample goals
        agent_initial_regions = np.random.choice(list(env_initial_regions),M,replace=False)
        routes = get_route_from_scenario(args.sc)
        agent_initial_regions = [str(rt[0]) for rt in routes]
        goals_gen = Goals(env=None, goal_sets = [env_goals[str(agent_initial_regions[i])] for i in range(M) ])
        goal_ind = goals_gen.sample()
        goal_ind = [int(rt[1]) for rt in routes]
        random_goals = [goals_states[str(i)] for i in goal_ind]
        goals =  np.array([[state.x for state in random_goals],[state.y for state in random_goals]])
        routes = [str(start_ind) + str(goal_ind[i]) for i, start_ind in enumerate(agent_initial_regions)]

        #Initial states
        agent_types = ['CAV' for _ in range(M)]
        weights = [1,1] 
        num_rk4_steps = 4

        # Get current scenario
        sc = abs(min(scenario_encoding(routes)))

        initial_default = [{'type':agent_types[ind],'state': env_initial_states[str(reg)]} for ind, reg in enumerate(agent_initial_regions)]
        initial_agents = [{'type':agent_types[ind],'state': agent_initial_states[str(reg)]} for ind, reg in enumerate(agent_initial_regions)]

        # Reference generator
        ref_generator = ReferenceGenerator(N=2*M_sim,dt=dt,initial_state=initial_default,goals=random_goals,env=None, radius=ca_radius,routes = routes, target_velocity=v_des, mode='frenet',road_width=road_width,road_length=road_length,fillet_radius=fillet_radius)
        refs_dict = ref_generator.get_reference(M_sim,initial_states=initial_agents,output_type=dict)
        ref = np.zeros((4,len(refs_dict[0]['x'])))
        planners = []

        if args.eval_mode == 'gt_mpc':
            '''
            Proposed method (IGT-MPC)
            '''
            #Route Curvature for each agent(vehicles)
            use_NN_cost2go= True
            K_arr = []
            for i in range(M):
                ref[0,:] = refs_dict[i]['s'] # s
                ref[1,:] = np.zeros(refs_dict[i]['s'].shape) # ey
                ref[2,:] = np.zeros(refs_dict[i]['s'].shape) # epsi
                ref[3,:] = refs_dict[i]['v'] # v

                if np.all(refs_dict[i]['K']==0):
                    def straight(s_cur):
                        return 0
                    s = ca.SX.sym('s')
                    K = ca.Function('K',[s],[straight(s)])
                else:
                    if routes[i] in ['12','23','34','41']: # Left turn
                        breakpoints = ca.DM([(road_length-road_width)/2,(road_length-road_width)/2 + max(abs(1/refs_dict[i]['K'][np.nonzero(refs_dict[i]['K'])])) * np.pi/2])  # Breakpoints where the function values change
                    else:
                        breakpoints = ca.DM([(road_length-road_width)/2 - fillet_radius,(road_length-road_width)/2 - fillet_radius + max(abs(1/refs_dict[i]['K'][np.nonzero(refs_dict[i]['K'])])) * np.pi/2])
                    values = ca.DM([0.0, refs_dict[i]['K'][np.nonzero(refs_dict[i]['K'])[0][0]], 0.0])  # Function values on each segment

                    # Create a symbolic variable
                    s = ca.SX.sym('s')

                    # Create a piecewise constant function for curvature
                    curvature_func = ca.pw_const(s, breakpoints, values)
                    K = ca.Function('K', [s], [curvature_func])
                K_arr.append(K)

            s_cur_arr = []
            for i in range(M):
                if routes[i] in ['12','13','14']:
                    s_cur = abs(initial_agents[i]['state'].x - env_initial_states[routes[i][0]].x)
                elif routes[i] in ['21','23','24']:
                    s_cur = abs(initial_agents[i]['state'].y - env_initial_states[routes[i][0]].y)
                elif routes[i] in ['32','31','34']:
                    s_cur = abs(initial_agents[i]['state'].x - env_initial_states[routes[i][0]].x)
                elif routes[i] in ['41','42','43']:
                    s_cur = abs(initial_agents[i]['state'].y - env_initial_states[routes[i][0]].y)
                else:
                    raise ValueError('Invalid route')
                s_cur_arr.append(s_cur)

            cur_agent_states = [{'type':agent_types[ind],'state': VehicleReference({'x':initial_agents[ind]['state'].x,'y':initial_agents[ind]['state'].y,'v':initial_agents[ind]['state'].v,'s':s_cur_arr[ind],'heading':initial_agents[ind]['state'].heading,'ey':0,'epsi':0,'K':K_arr[ind]})} for ind in range(M)]
            prev_agent_inputs = [VehicleAction({'a':0,'df':0}) for _ in range(M)]

            # Initialize closed-loop trajectories
            z_cl = np.zeros((7*M,M_sim+1)) # [x,y,s,ey,epsi,v,psi]
            u_cl = np.zeros((2*M,M_sim)) # [a,df]
            preds_cl = []
            # Update closed-loop trajectories with initial states
            print('INITIALIZING MPC Planners...')
            for i in range(M):
                z_cl[i*7+0,0] = cur_agent_states[i]['state'].x
                z_cl[i*7+1,0] = cur_agent_states[i]['state'].y
                z_cl[i*7+2,0] = cur_agent_states[i]['state'].s
                z_cl[i*7+3,0] = cur_agent_states[i]['state'].ey
                z_cl[i*7+4,0] = cur_agent_states[i]['state'].epsi
                z_cl[i*7+5,0] = cur_agent_states[i]['state'].v
                z_cl[i*7+6,0] = cur_agent_states[i]['state'].heading

                u_cl[i*2+0,0] = 0
                u_cl[i*2+1,0] = 0

                planners.append(MPC_Planner(N=N,dt=dt,ca_radius=ca_radius,agents=cur_agent_states,routes = routes,ref = refs_dict,goals=random_goals,road_dim=(road_width,road_length),ds_right=fillet_radius,index=i,num_rk4_steps=num_rk4_steps,use_NN_cost2go=use_NN_cost2go,ca_type=ca_type,nn_config_dir=get_scenario_config(sc),weights=weights))
                NN_query_time_arr = []
                cav_kin_model = KinematicBicycleModelFrenet(predictor.env_config['l_r'],predictor.env_config['l_f'],predictor.env_config['width'],dt,discretization='rk4',mode='numpy',num_rk4_steps=num_rk4_steps)
            print('FINISHED INITIALIZING MPC Planners!')              
            t_track = 0
            prev_cav_sols = []
            prev_cav_index_opt = []
            s_N_guides = [[] for _ in range(M)]
            num_infeasible = [0 for _ in range(M)]
            avg_solve_times = [[] for _ in range(M)]      

            for t in range(M_sim):
                    print(f'Time: {t}')
                    print(f'Routes: {routes}, Cur States: {[(cur_agent_states[i]["state"].x, cur_agent_states[i]["state"].y, cur_agent_states[i]["state"].v) for i in range(M)]}')
                    # Predict 
                    preds = predictor.predict(agents=cur_agent_states,agent_cur_inputs=prev_agent_inputs,routes=routes,refs=refs_dict)
                    if t == 0:
                        a0 = 0
                        jerk_limit = 0.09
                        preds = predictor.predict(agents=cur_agent_states,agent_cur_inputs= [VehicleAction({'a':a0+jerk_limit*(k+1),'df':0}) for k in range(M)],routes=routes,refs=refs_dict)
                    
                    # V2V communication (share motion forecasts with other CAVs)
                    preds4CAV = preds
                    if t > 0 and cav_sols:
                        preds4CAV = share_motion_forecasts(predictor,preds,cav_sols,cav_index_opt,refs_dict,routes,{'N':N,'K':K_arr})
                    preds_cl.append(preds4CAV)
                    # Update planners with current predictions and states
                    cur_agent_inputs = []
                    next_agent_states = []

                    cav_sols = []
                    cav_index_opt = []

                    for i in range(M):
                        print('AGENT: ',i)
                        print(f'prev: {prev_agent_inputs[i].a,prev_agent_inputs[i].df}')
                        planners[i].update_initial_condition(cur_agent_states[i],prev_agent_inputs[i])

                        if policy_config['type'] == 'MPC' and agent_types[i]=='CAV':
                            filtered_preds = filter_preds(preds4CAV,i)
                            planners[i].update_predictions(filtered_preds,raw_preds=preds4CAV)
                            if i in prev_cav_index_opt and t > 1 :
                                (x_sol_prev,u_sol_prev) = augment_prev_sol(prev_cav_sols[prev_cav_index_opt.index(i)],cav_kin_model,cur_agent_states[i]['state'].K)
                            else:
                                x_sol_prev, u_sol_prev = None, None 

                            (x_sol, u_sol, optimality) = planners[i].solve(x_sol_prev=x_sol_prev, u_sol_prev = u_sol_prev)
                            if optimality:
                                if N == 1:
                                    u_sol = u_sol[:,np.newaxis]
                                cav_sols.append([x_sol,u_sol])
                                cav_index_opt.append(i) 
                        else:
                            raise ValueError('Invalid policy type')
                        if optimality:
                            # Update current agent states (env update assuming perfect model)
                            print(f'optimal: {u_sol[0,0],u_sol[1,0]}')
                            next_agent_states.append({'type':agent_types[i],'state': VehicleReference({'x':x_sol[0,1],'y':x_sol[1,1],'v':x_sol[5,1],'s':x_sol[2,1],'heading':x_sol[6,1],'ey':x_sol[3,1],'epsi':x_sol[4,1],'K':cur_agent_states[i]['state'].K})})
                            cur_agent_inputs.append(VehicleAction({'a':u_sol[0,0],'df':u_sol[1,0]}))

                            # Update closed-loop trajectories
                            z_cl[i*7+0,t+1] = x_sol[0,1]
                            z_cl[i*7+1,t+1] = x_sol[1,1]
                            z_cl[i*7+2,t+1] = x_sol[2,1]
                            z_cl[i*7+3,t+1] = x_sol[3,1]
                            z_cl[i*7+4,t+1] = x_sol[4,1]
                            z_cl[i*7+5,t+1] = x_sol[5,1]
                            z_cl[i*7+6,t+1] = x_sol[6,1]
            
                            u_cl[i*2+0,0] = u_sol[0,0]
                            u_cl[i*2+1,0] = u_sol[1,0]
                            next_state = next_agent_states[i]['state']
                        else:    
                            num_infeasible[i] += 1                        
                            # Update current agent states (env update assuming perfect model)
                            temp_agent_inputs = copy.copy(prev_agent_inputs)
                            temp_agent_inputs[i] = VehicleAction({'a':policy_config['a_min'] if cur_agent_states[i]['state'].v > 0 else 0,'df':prev_agent_inputs[i].df}) #Slamming the brake
                            kin_model = cav_kin_model
                            
                            next_state = kin_model(cur_agent_states[i]['state'], temp_agent_inputs[i])

                            # Heuristics
                            if cur_agent_states[i]['state'].v < 0:
                                cur_agent_inputs.append(VehicleAction({'a':0,'df':prev_agent_inputs[i].df})) # a set to 0 for input rate feasibility
                                next_state = copy.copy(cur_agent_states[i]['state'])
                                next_state.v = 0 # Instantaneous stop
                            else:
                                cur_agent_inputs.append(temp_agent_inputs[i])
                            next_agent_states.append({'type':agent_types[i],'state': next_state})              

                            # Update closed-loop trajectories
                            z_cl[i*7+0,t+1] = next_state.x
                            z_cl[i*7+1,t+1] = next_state.y
                            z_cl[i*7+2,t+1] = next_state.s
                            z_cl[i*7+3,t+1] = next_state.ey
                            z_cl[i*7+4,t+1] = next_state.epsi
                            z_cl[i*7+5,t+1] = next_state.v
                            z_cl[i*7+6,t+1] = next_state.heading 
                            
                            u_cl[i*2+0,t] = cur_agent_inputs[-1].a
                            u_cl[i*2+1,t] = cur_agent_inputs[-1].df
                        # Log solve time
                        if optimality:
                            t_key = [key for key in planners[i].sol.stats().keys() if 't_' in key]
                            temp = sum(planners[i].sol.stats()[key] for key in t_key)
                            avg_solve_times[i].append(temp)          
                            NN_query_time_arr.append(planners[i].NN_query_time)

                    # Update prev_agent_inputs and cur_agent_states
                    cur_agent_states = next_agent_states
                    prev_agent_inputs = cur_agent_inputs
                
                    # Store previous solutions of all feasible solutions
                    prev_cav_sols = cav_sols
                    prev_cav_index_opt = cav_index_opt
                    del filtered_preds
                    t_track += 1

            # Deadlock:
            deadlock = False
            if np.sum([cur_agent_states[i]['state'].s <= 30 for i in range(M)]) >= 2:
                deadlock = True

            # Animate trajectory
            ref_data = np.zeros((6*M,M_sim+1))
            for i in range(M):
                ref_data[6*i+0,:] = refs_dict[i]['x'] # x
                ref_data[6*i+1,:] = refs_dict[i]['y'] # y
                ref_data[6*i+2,:] = refs_dict[i]['heading'] # psi(heading)
                ref_data[6*i+3,:] = refs_dict[i]['v'] # v
                ref_data[6*i+4,:] = refs_dict[i]['s'] # s
                ref_data[6*i+5,:] = refs_dict[i]['K'] # K
            
            if not os.path.isdir(args.save_dir +args.eval_mode + '_sc' + str(args.sc) +'_seed' + str(seed) + '_' + timenow + '/game_mpc/evaluation_videos'):
                os.makedirs(args.save_dir + args.eval_mode + '_sc' + str(args.sc) +'_seed' + str(seed) + '_' + timenow + '/game_mpc/evaluation_videos')
            if not os.path.isdir(args.save_dir + args.eval_mode + '_sc' + str(args.sc) +'_seed' + str(seed) + '_' + timenow + '/game_mpc/evaluation'):   
                os.makedirs(args.save_dir + args.eval_mode + '_sc' + str(args.sc) +'_seed' + str(seed) + '_' + timenow + '/game_mpc/evaluation')
            
            # Save model config and mpc yaml
            with open(args.save_dir + args.eval_mode +  '_sc' + str(args.sc) +'_seed' + str(seed) + '_' + timenow + '/game_mpc/evaluation/mpc.yaml', 'w') as f:
                yaml.dump(policy_config, f)

            animate_trajectory(initial_agents,ca_radius if ca_type=='circle' else 0,z_cl[:,:t_track+1],target=goals,roads=(road_length,road_width),fillet_radius=fillet_radius,filename = args.save_dir + args.eval_mode + '_sc' + str(args.sc)+ '_seed' + str(seed) + '_' + timenow + '/game_mpc/evaluation_videos/iter' + str(it) + '.mp4',preds_cl=preds_cl,vis_preds=True,s_N_guides=s_N_guides,refs=refs_dict,routes=routes)

            if not os.path.isfile(args.save_dir + args.eval_mode + '_sc' + str(args.sc) +'_seed' + str(seed) + '_' + timenow + '/game_mpc/evaluation/cl_traj.pkl'):
                with open(args.save_dir + args.eval_mode +  '_sc' + str(args.sc) +'_seed' + str(seed) + '_' + timenow + '/game_mpc/evaluation/cl_traj.pkl','wb') as f:
                    pickle.dump(z_cl[:,:t_track+1][np.newaxis,:,:],f)
                with open(args.save_dir + args.eval_mode + '_sc' + str(args.sc) +'_seed' + str(seed) + '_' + timenow + '/game_mpc/evaluation/u_cl.pkl','wb') as f:
                    pickle.dump(u_cl[np.newaxis,:,:],f)
            else:
                with open(args.save_dir + args.eval_mode +  '_sc' + str(args.sc) +'_seed' + str(seed) + '_' + timenow + '/game_mpc/evaluation/cl_traj.pkl','rb') as f:
                    cl_traj = pickle.load(f)
                with open(args.save_dir + args.eval_mode + '_sc' + str(args.sc) +'_seed' + str(seed) + '_' + timenow + '/game_mpc/evaluation/u_cl.pkl','rb') as f:
                    u_cl_traj = pickle.load(f)
                cl_traj = np.concatenate([cl_traj,z_cl[:,:t_track+1][np.newaxis,:,:]],axis=0)
                u_cl_traj = np.concatenate([u_cl_traj,u_cl[np.newaxis,:,:]],axis=0)
                with open(args.save_dir + args.eval_mode +  '_sc' + str(args.sc) +'_seed' + str(seed) + '_' + timenow + '/game_mpc/evaluation/cl_traj.pkl','wb') as f:
                    pickle.dump(cl_traj,f)
                with open(args.save_dir + args.eval_mode + '_sc' + str(args.sc) + '_seed' + str(seed) + '_' + timenow + '/game_mpc/evaluation/u_cl.pkl','wb') as f:
                    pickle.dump(u_cl_traj,f)

            # Get stats as np array and export as csv
            avg_solve_time = [np.mean(avg_solve_times[i]) for i in range(M)]
            std_solve_time = [np.std(avg_solve_times[i]) for i in range(M)]
            num_infeasible = [num_infeasible[i]/M_sim for i in range(M)]
            stat_dict = {'NN_query_time': np.array([-1]), 'avg_sol_times':np.array(avg_solve_time),'std_solve_times':std_solve_time,'infeasible_ratio':np.array(num_infeasible),'deadlock':deadlock}
            # Check if the csv file already exists
            csv_file = args.save_dir + args.eval_mode +  '_sc' + str(args.sc) + '_seed' + str(seed) + '_' + timenow + '/game_mpc/evaluation/stats.csv'
            if os.path.isfile(csv_file):
                # Append the stats to the existing csv file
                with open(csv_file, mode='a', newline='') as file:
                    writer = csv.DictWriter(file, fieldnames=stat_dict.keys())
                    writer.writerow(stat_dict)
            else:
                # Create a new csv file and write the stats
                with open(csv_file, mode='w', newline='') as file:
                    writer = csv.DictWriter(file, fieldnames=stat_dict.keys())
                    writer.writeheader()
                    writer.writerow(stat_dict)
        elif args.eval_mode == 'mpc':
            '''
            MPC (No NN)
            '''
            print('STARTING MPC EVALUATION'.center(80,'#'))
            planners = []
            use_NN_cost2go= False
            K_arr = []
            for i in range(M):
                ref[0,:] = refs_dict[i]['s'] # s
                ref[1,:] = np.zeros(refs_dict[i]['s'].shape) # ey
                ref[2,:] = np.zeros(refs_dict[i]['s'].shape) # epsi
                ref[3,:] = refs_dict[i]['v'] # v

                if np.all(refs_dict[i]['K']==0):
                    def straight(s_cur):
                        return 0
                    s = ca.SX.sym('s')
                    K = ca.Function('K',[s],[straight(s)])
                else:
                    if routes[i] in ['12','23','34','41']: # Left turn
                        breakpoints = ca.DM([(road_length-road_width)/2,(road_length-road_width)/2 + max(abs(1/refs_dict[i]['K'][np.nonzero(refs_dict[i]['K'])])) * np.pi/2])  # Breakpoints where the function values change
                    else:
                        breakpoints = ca.DM([(road_length-road_width)/2 - fillet_radius,(road_length-road_width)/2 - fillet_radius + max(abs(1/refs_dict[i]['K'][np.nonzero(refs_dict[i]['K'])])) * np.pi/2])
                    values = ca.DM([0.0, refs_dict[i]['K'][np.nonzero(refs_dict[i]['K'])[0][0]], 0.0])  # Function values on each segment

                    # Create a symbolic variable
                    s = ca.SX.sym('s')

                    # Create a piecewise constant function for curvature
                    curvature_func = ca.pw_const(s, breakpoints, values)
                    K = ca.Function('K', [s], [curvature_func])
                K_arr.append(K)

            s_cur_arr = []
            for i in range(M):
                if routes[i] in ['12','13','14']:
                    s_cur = abs(initial_agents[i]['state'].x - env_initial_states[routes[i][0]].x)
                elif routes[i] in ['21','23','24']:
                    s_cur = abs(initial_agents[i]['state'].y - env_initial_states[routes[i][0]].y)
                elif routes[i] in ['32','31','34']:
                    s_cur = abs(initial_agents[i]['state'].x - env_initial_states[routes[i][0]].x)
                elif routes[i] in ['41','42','43']:
                    s_cur = abs(initial_agents[i]['state'].y - env_initial_states[routes[i][0]].y)
                else:
                    raise ValueError('Invalid route')
                s_cur_arr.append(s_cur)

            cur_agent_states = [{'type':agent_types[ind],'state': VehicleReference({'x':initial_agents[ind]['state'].x,'y':initial_agents[ind]['state'].y,'v':initial_agents[ind]['state'].v,'s':s_cur_arr[ind],'heading':initial_agents[ind]['state'].heading,'ey':0,'epsi':0,'K':K_arr[ind]})} for ind in range(M)]
            prev_agent_inputs = [VehicleAction({'a':0.1,'df':0}) for _ in range(M)]

            # Initialize closed-loop trajectories
            z_cl = np.zeros((7*M,M_sim+1)) #[x,y,psi,v]
            u_cl = np.zeros((2*M,M_sim)) #[a,df]
            x_data = np.zeros((7*3,M_sim+1)) #(x,y,s,ey,epsi,v,heading)
            u_data = np.zeros((2*3,M_sim)) #(a,df)
            preds_cl = []
            
            # Update closed-loop trajectories with initial states
            for i in range(M):
                z_cl[i*7+0,0] = cur_agent_states[i]['state'].x
                z_cl[i*7+1,0] = cur_agent_states[i]['state'].y
                z_cl[i*7+2,0] = cur_agent_states[i]['state'].s
                z_cl[i*7+3,0] = cur_agent_states[i]['state'].ey
                z_cl[i*7+4,0] = cur_agent_states[i]['state'].epsi
                z_cl[i*7+5,0] = cur_agent_states[i]['state'].v
                z_cl[i*7+6,0] = cur_agent_states[i]['state'].heading
                u_cl[i*2+0,0] = 0.1
                u_cl[i*2+1,0] = 0

                planners.append(MPC_Planner(N=N,dt=dt,ca_radius=ca_radius,agents=cur_agent_states,routes = routes,ref = refs_dict,goals=random_goals,road_dim=(road_width,road_length),ds_right=fillet_radius,index=i,num_rk4_steps=num_rk4_steps))
                cav_kin_model = KinematicBicycleModelFrenet(predictor.env_config['l_r'],predictor.env_config['l_f'],predictor.env_config['width'],dt,discretization='rk4',mode='numpy',num_rk4_steps=num_rk4_steps)
                    
            t_track = 0
            prev_cav_sols = []
            prev_cav_index_opt = []
            s_N_guides = [[] for _ in range(M)]

            num_infeasible = [0 for _ in range(M)]
            avg_solve_times = [[] for _ in range(M)]      

            for t in range(M_sim):
                    print(f'Time: {t}')
                    print(f'Routes: {routes}, Cur States: {[(cur_agent_states[i]["state"].x, cur_agent_states[i]["state"].y, cur_agent_states[i]["state"].v) for i in range(M)]}')
                    # Predict 
                    preds = predictor.predict(agents=cur_agent_states,agent_cur_inputs=prev_agent_inputs,routes=routes,refs=refs_dict)

                    # V2V communication (share motion forecasts with other CAVs)
                    preds4CAV = copy.deepcopy(preds)
                    if t > 0 and cav_sols:
                        preds4CAV = share_motion_forecasts(predictor,preds,cav_sols,cav_index_opt,refs_dict,routes,{'N':N,'K':K_arr})
                    preds_cl.append(preds4CAV)
                    # Update planners with current predictions and states
                    cur_agent_inputs = []
                    next_agent_states = []

                    cav_sols = []
                    cav_index_opt = []

                    for i in range(M):
                        planners[i].update_initial_condition(cur_agent_states[i],prev_agent_inputs[i])
                        if t == 0:
                            x_data[7*i:(i+1)*7,t] = np.array([cur_agent_states[i]['state'].x,cur_agent_states[i]['state'].y,cur_agent_states[i]['state'].s,cur_agent_states[i]['state'].ey,cur_agent_states[i]['state'].epsi,cur_agent_states[i]['state'].v,cur_agent_states[i]['state'].heading])
                        if policy_config['type'] == 'MPC' and agent_types[i]=='CAV':
                            if policy_config['NN_type']=='s_sequence':
                                planners[i].update_predictions(filter_preds(preds4CAV,i),raw_preds=preds4CAV)
                            else:
                                planners[i].update_predictions(filter_preds(preds4CAV,i),raw_preds=preds4CAV)
                            if i in prev_cav_index_opt:
                                (x_sol_prev,u_sol_prev) = augment_prev_sol(prev_cav_sols[prev_cav_index_opt.index(i)],cav_kin_model,cur_agent_states[i]['state'].K)
                            else:
                                x_sol_prev, u_sol_prev = None, None       
                            (x_sol, u_sol, optimality) = planners[i].solve(x_sol_prev=x_sol_prev, u_sol_prev = u_sol_prev)
                            if optimality:
                                if N == 1:
                                    u_sol = u_sol[:,np.newaxis]
                                cav_sols.append([x_sol,u_sol])
                                cav_index_opt.append(i) 
                        else:
                            raise ValueError('Invalid policy type')

                        if optimality:
                            # Update current agent states (env update assuming perfect model)
                            next_agent_states.append({'type':agent_types[i],'state': VehicleReference({'x':x_sol[0,1],'y':x_sol[1,1],'v':x_sol[5,1],'s':x_sol[2,1],'heading':x_sol[6,1],'ey':x_sol[3,1],'epsi':x_sol[4,1],'K':cur_agent_states[i]['state'].K})})
                            cur_agent_inputs.append(VehicleAction({'a':u_sol[0,0],'df':u_sol[1,0]}))

                            # Update closed-loop trajectories
                            z_cl[i*7+0,t+1] = x_sol[0,1]
                            z_cl[i*7+1,t+1] = x_sol[1,1]
                            z_cl[i*7+2,t+1] = x_sol[2,1]
                            z_cl[i*7+3,t+1] = x_sol[3,1]
                            z_cl[i*7+4,t+1] = x_sol[4,1]
                            z_cl[i*7+5,t+1] = x_sol[5,1]
                            z_cl[i*7+6,t+1] = x_sol[6,1]

                            u_cl[i*2+0,0] = u_sol[0,0]
                            u_cl[i*2+1,0] = u_sol[1,0]
            
                            # Update x_data and u_data
                            x_data[i*7:(i+1)*7,t+1] = x_sol[:,1]
                            u_data[i*2:(i+1)*2,t] = u_sol[:,0]
                        else:    
                            num_infeasible[i] += 1                        
                            # Update current agent states (env update assuming perfect model)
                            temp_agent_inputs = copy.deepcopy(prev_agent_inputs)
                            temp_agent_inputs[i] = VehicleAction({'a':policy_config['a_min'] if cur_agent_states[i]['state'].v > 0 else 0,'df':prev_agent_inputs[i].df}) #Slamming the brake
                            
                            kin_model = cav_kin_model
                
                            
                            next_state = kin_model(cur_agent_states[i]['state'], temp_agent_inputs[i])

                            # Heuristics
                            if cur_agent_states[i]['state'].v < 0:
                                cur_agent_inputs.append(VehicleAction({'a':0,'df':prev_agent_inputs[i].df})) # a set to 0 for input rate feasibility
                                next_state = copy.deepcopy(cur_agent_states[i]['state'])
                                next_state.v = 0 # Instantaneous stop
                            else:
                                cur_agent_inputs.append(temp_agent_inputs[i])
                            next_agent_states.append({'type':agent_types[i],'state': next_state})              

                            # Update closed-loop trajectories
                            z_cl[i*7+0,t+1] = next_state.x
                            z_cl[i*7+1,t+1] = next_state.y
                            z_cl[i*7+2,t+1] = next_state.s
                            z_cl[i*7+3,t+1] = next_state.ey
                            z_cl[i*7+4,t+1] = next_state.epsi
                            z_cl[i*7+5,t+1] = next_state.v
                            z_cl[i*7+6,t+1] = next_state.heading 

                            u_cl[i*2+0,t] = cur_agent_inputs[-1].a
                            u_cl[i*2+1,t] = cur_agent_inputs[-1].df

                            # Update x_data and u_data
                            x_data[i*7:(i+1)*7,t+1] = [next_state.x,next_state.y,next_state.s,next_state.ey,next_state.epsi,next_state.v,next_state.heading]
                            u_data[i*2:(i+1)*2,t] = [cur_agent_inputs[-1].a,cur_agent_inputs[-1].df]

                        # Log solve time
                        if optimality:
                            t_key = [key for key in planners[i].sol.stats().keys() if 't_wall' in key]
                            temp = 0
                            for key in t_key:
                                temp += planners[i].sol.stats()[key]
                            temp = planners[i].solve_time
                            avg_solve_times[i].append(temp)          

                    # Update prev_agent_inputs and cur_agent_states
                    cur_agent_states = next_agent_states
                    prev_agent_inputs = cur_agent_inputs
                
                    # Store previous solutions of all feasible solutions
                    prev_cav_sols = cav_sols
                    prev_cav_index_opt = cav_index_opt

                    t_track += 1    

            # Deadlock:
            deadlock = False
            if np.sum([x_data[7*i+2,-1] <= 30 for i in range(M)]) >= 2:
                deadlock = True
                
            # Animate trajectory
            ref_data = np.zeros((6*M,M_sim+1))
            for i in range(M):
                ref_data[6*i+0,:] = refs_dict[i]['x'] # x
                ref_data[6*i+1,:] = refs_dict[i]['y'] # y
                ref_data[6*i+2,:] = refs_dict[i]['heading'] # psi(heading)
                ref_data[6*i+3,:] = refs_dict[i]['v'] # v
                ref_data[6*i+4,:] = refs_dict[i]['s'] # s
                ref_data[6*i+5,:] = refs_dict[i]['K'] # K
            
            if not os.path.isdir(args.save_dir + args.eval_mode + '_sc' + str(args.sc) +'_seed' + str(seed) + '_' + timenow + '/mpc'):
                os.makedirs(args.save_dir + args.eval_mode + '_sc' + str(args.sc) +'_seed' + str(seed) + '_' + timenow + '/mpc')
                os.makedirs(args.save_dir + args.eval_mode + '_sc' + str(args.sc) +'_seed' + str(seed) + '_' + timenow + '/mpc/evaluation_videos')
            
                # Save model config and mpc yaml
                with open(args.save_dir + args.eval_mode + '_sc' + str(args.sc) +'_seed' + str(seed) + '_' + timenow + '/mpc/mpc.yaml', 'w') as f:
                    yaml.dump(policy_config, f)
            videoname = args.save_dir + args.eval_mode + '_sc' + str(args.sc) +'_seed' + str(seed) + '_' + timenow + '/mpc/evaluation_videos/iter' + str(it) + '.mp4'
            animate_trajectory(initial_agents,ca_radius,z_cl[:,:t_track+1],target=goals,roads=(road_length,road_width),fillet_radius=fillet_radius,filename=videoname,preds_cl=preds_cl,vis_preds=True,s_N_guides=s_N_guides,refs=refs_dict,routes=routes)

            # Save the data
            data_filename = args.save_dir + args.eval_mode + '_sc' + str(args.sc) +'_seed' + str(seed) + '_' + timenow + '/mpc/evaluation_data.pkl'
            if os.path.isfile(data_filename):
                with open(data_filename, 'rb') as file:
                    paths = pickle.load(file)
                data = {'N': N, 'initial_default': initial_default,'refs':np.concatenate([paths['refs'], np.array([ref_data])],axis=0),'x_cl': np.concatenate([paths['x_cl'], np.array([x_data])],axis=0), 'u_cl': np.concatenate([paths['u_cl'], np.array([u_data])],axis=0), 'initial_agents': np.vstack([paths['initial_agents'], initial_agents]), 'agent_types': agent_types, 'weights': np.concatenate([paths['weights'],np.array([weights])],axis=0), "routes": np.concatenate([paths['routes'], np.array([routes])],axis=0), 'goals': np.concatenate([paths['goals'], np.array([goals])],axis=0),'deadlock':np.vstack([paths['deadlock'],np.array([deadlock])])}
                with open(data_filename, 'wb') as handle:
                    pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
            else:
                data = {'N': N,'refs':np.array([ref_data]),'x_cl': np.array([x_data]), 'u_cl': np.array([u_data]), 'initial_agents': initial_agents, 'agent_types': agent_types, 'weights': np.array([weights]), "routes": np.array([routes]), 'goals': np.array([goals]), 'initial_default': initial_default, 'deadlock': np.array([deadlock])}   
                with open(data_filename, 'wb') as handle:
                        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
                
            # Get stats as np array and export as csv
            avg_solve_times = [np.mean(avg_solve_times[i]) for i in range(M)]
            std_solve_times = [np.std(avg_solve_times[i]) for i in range(M)]
            num_infeasible = [num_infeasible[i]/M_sim for i in range(M)]
            stat_dict = {'avg_sol_times':np.array(avg_solve_times),'std_solve_times':std_solve_times,'infeasible_ratio':np.array(num_infeasible),'deadlock':deadlock}
            if not os.path.isfile(args.save_dir + args.eval_mode + '_sc' + str(args.sc) +'_seed' + str(seed) + '_' + timenow + '/mpc/cl_traj.pkl'):
                with open(args.save_dir + args.eval_mode +  '_sc' + str(args.sc) + '_seed' + str(seed) + '_' + timenow + '/mpc/cl_traj.pkl','wb') as f:
                    pickle.dump(z_cl[:,:t_track+1][np.newaxis,:,:],f)
                with open(args.save_dir + args.eval_mode + '_sc' + str(args.sc) +'_seed' + str(seed) + '_' + timenow + '/mpc/u_cl.pkl','wb') as f:
                    pickle.dump(u_cl[np.newaxis,:,:],f)
            else:
                with open(args.save_dir + args.eval_mode +  '_sc' + str(args.sc) + '_seed' + str(seed) + '_' + timenow + '/mpc/cl_traj.pkl','rb') as f:
                    cl_traj = pickle.load(f)
                with open(args.save_dir + args.eval_mode +  '_sc' + str(args.sc) + '_seed' + str(seed) + '_' + timenow + '/mpc/u_cl.pkl','rb') as f:
                    u_cl_traj = pickle.load(f)
                cl_traj = np.concatenate([cl_traj,z_cl[:,:t_track+1][np.newaxis,:,:]],axis=0)
                u_cl_traj = np.concatenate([u_cl_traj,u_cl[np.newaxis,:,:]],axis=0)

                with open(args.save_dir + args.eval_mode + '_sc' + str(args.sc) + '_seed' + str(seed) + '_' + timenow + '/mpc/cl_traj.pkl','wb') as f:
                    pickle.dump(cl_traj,f)
                with open(args.save_dir + args.eval_mode + '_sc' + str(args.sc) + '_seed' + str(seed) + '_' + timenow + '/mpc/u_cl.pkl','wb') as f:
                    pickle.dump(u_cl_traj,f)

            # Check if the csv file already exists
            csv_file = args.save_dir + args.eval_mode + '_sc' + str(args.sc) +'_seed' + str(seed) + '_' + timenow + '/mpc/eval_stats.csv'
            if os.path.isfile(csv_file):
                # Append the stats to the existing csv file
                with open(csv_file, mode='a', newline='') as file:
                    writer = csv.DictWriter(file, fieldnames=stat_dict.keys())
                    writer.writerow(stat_dict)
            else:
                # Create a new csv file and write the stats
                with open(csv_file, mode='w', newline='') as file:
                    writer = csv.DictWriter(file, fieldnames=stat_dict.keys())
                    writer.writeheader()
                    writer.writerow(stat_dict)
        else:
            raise ValueError('Invalid evaluation mode')

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir', type=str, required=True)
    parser.add_argument('--num_samples', type=int, default=1)
    parser.add_argument('--eval_mode', type=str, required=True)
    parser.add_argument('--sc', type=int, default=1,required=False)
    args = parser.parse_args()
    main(args)

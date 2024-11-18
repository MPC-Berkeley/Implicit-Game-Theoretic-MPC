#!/usr/bin/env python3

import pathlib
import os
from collections import deque
import copy
import yaml
import sys
import pickle

import pdb

import torch

import numpy as np
import scipy as sp
from scipy.interpolate import interp1d
from scipy.spatial.transform import Rotation
import casadi as ca

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.transforms import Affine2D
from matplotlib.animation import FFMpegWriter
from matplotlib.cm import ScalarMappable

from mpclab_common.pytypes import VehicleState, ParametricPose, OrientationEuler, BodyLinearVelocity
from mpclab_common.rosbag_utils import rosbagData
from mpclab_common.track import get_track, load_mpclab_raceline



experiment_dirs = ['barc2_dg_barc4_mp', 'barc2_mp_barc4_dg', 'barc2_dg_barc4_dg']
data_dirs = [['barc_run_race_h2h_12-22-2023_16-48-56', 
              'barc_run_race_h2h_12-22-2023_16-54-34',
              'barc_run_race_h2h_12-22-2023_17-06-15',
              'barc_run_race_h2h_12-22-2023_17-08-08',
              'barc_run_race_h2h_12-22-2023_17-10-04'],
             ['barc_run_race_h2h_12-22-2023_16-51-47', 
              'barc_run_race_h2h_12-22-2023_16-57-02',
              'barc_run_race_h2h_12-22-2023_16-59-23',
              'barc_run_race_h2h_12-22-2023_17-01-27',
              'barc_run_race_h2h_12-22-2023_17-03-59',
              'barc_run_race_h2h_12-22-2023_17-29-39',
              'barc_run_race_h2h_12-22-2023_17-31-49',
              'barc_run_race_h2h_12-22-2023_17-33-13',
              'barc_run_race_h2h_12-22-2023_17-35-17'],
             ['barc_run_race_h2h_12-22-2023_17-38-48',
              'barc_run_race_h2h_12-22-2023_17-40-53',
              'barc_run_race_h2h_12-22-2023_17-43-54',
              'barc_run_race_h2h_12-22-2023_17-45-49',
              'barc_run_race_h2h_12-22-2023_17-47-52']]
ego_cars = [['car1', 'car2'],
            ['car1', 'car2'],
            ['car1', 'car2']]

show_sol = True
show_pred = False
show_value = True
show_raceline = True

# Position of the track origin w.r.t. the OptiTrack origin
track_origin_x = -3.4669
track_origin_y = 1.9382
track_origin_z = 0.0

# Orientation of the track origin w.r.t. the OptiTrack origin
track_origin_roll = 0.0
track_origin_pitch = 0.0
track_origin_yaw = np.pi/2

# Offset of the OptiTrack measurement point w.r.t. car body center of mass
opti_body_offset_long = -0.16
opti_body_offset_tran = 0.0
opti_body_offset_vert = 0.0

opti_body_offset_roll = 0.0
opti_body_offset_pitch = 0.0
opti_body_offset_yaw = 0.0

# Define transformations
opti2track_p = np.array([track_origin_x, track_origin_y, track_origin_z])
opti2track_R = Rotation.from_euler('ZYX', [track_origin_yaw, track_origin_pitch, track_origin_roll])
car2opti_p = np.array([opti_body_offset_long, opti_body_offset_tran, opti_body_offset_vert])
car2opti_R = Rotation.from_euler('ZYX', [opti_body_offset_yaw, opti_body_offset_pitch, opti_body_offset_roll])

VL = 0.37
VW = 0.195

car1_alias = 'barc_2'
car2_alias = 'barc_4'

car1_c = 'b'
car2_c = 'g'

for k, (exp, d) in enumerate(zip(experiment_dirs, data_dirs)):
    for ego_car in ego_cars[k]:
        # Load controller parameters for ego vehicle
        if ego_car == 'car1':
            ego_alias = car1_alias
            ego_c = car1_c
            tar_c = car2_c
        elif ego_car == 'car2':
            ego_alias = car2_alias
            ego_c = car2_c
            tar_c = car1_c

        ego_params_path = pathlib.Path(exp, f'barc_run_race_h2h/{ego_alias}/controller.yaml')
        with open(ego_params_path, 'r') as f:
            ego_params = yaml.safe_load(f)[f'experiment/{ego_alias}/{ego_alias}_control']['ros__parameters']

        ego_sol_r = np.sqrt((VL/2)**2 + (VW/2)**2)*np.ones(ego_params['N'])
        ego_pred_r = np.sqrt((VL/2)**2 + (VW/2)**2)*np.ones(ego_params['N'])
        if ego_params['uncertain_predictions']:
            ego_pred_r += ego_params['uncertainty_increment']*np.arange(1, ego_params['N']+1)

        ego_terminal_cost = ego_params['value_function']
        if ego_terminal_cost == 'dynamic_game':
            module_path = os.path.expanduser(ego_params['module_path'])
            sys.path.append(module_path)
            from model import mlp

            # Construct learned value function neural network
            data_path = os.path.expanduser(ego_params['data_path'])
            with open(data_path, 'rb') as f:
                D = pickle.load(f)
            feature_mean = D['train'].feature_mean
            feature_cov_inv = sp.linalg.inv(sp.linalg.sqrtm(D['train'].feature_cov))
            target_mean = D['train'].target_mean
            target_cov = D['train'].target_cov

            model = mlp(input_layer_size=len(D['test'][0][0]),
                        output_layer_size=len(D['test'][0][1]),
                        hidden_layer_sizes=[48, 48], 
                        activation='tanh', 
                        batch_norm=False)
            model_path = os.path.expanduser(ego_params['model_path'])
            model.load_state_dict(torch.load(model_path))
            
            model.to(torch.device('cuda'))
            model.eval()
            def value_function(ego_state, tar_state):
                if len(ego_params['term_idx']) == 3:
                    tar_x = np.array([tar_state.v.v_long, tar_state.p.s, tar_state.p.x_tran])
                    ego_x = np.array([ego_state.v.v_long, ego_state.p.s, ego_state.p.x_tran])
                elif len(ego_params['term_idx']) == 4:
                    tar_x = np.array([tar_state.v.v_long, tar_state.e.psi, tar_state.p.s, tar_state.p.x_tran])
                    ego_x = np.array([ego_state.v.v_long, ego_state.e.psi, ego_state.p.s, ego_state.p.x_tran])
                x = np.concatenate((tar_x, ego_x-tar_x))
                x_norm = sp.linalg.solve(sp.linalg.sqrtm(D['train'].feature_cov), x - D['train'].feature_mean, assume_a='pos')
                V = float(float(model(torch.tensor(x_norm.reshape((1,-1))).cuda()).cpu())*D['train'].target_cov + D['train'].target_mean)
                return V
        elif ego_terminal_cost == 'max_progress':
            def value_function(ego_state, tar_state):
                V = ego_state.p.s - tar_state.p.s
                return V

        # Load track
        track_name = 'L_track_barc_reverse'
        track = get_track(track_name)

        if show_raceline:
            # Load raceline
            path = pathlib.Path(ego_params['reference_file']).expanduser()
            raceline, s2t, raceline_mat = load_mpclab_raceline(path, track_name, time_scale=ego_params['reference_scaling'])

        for _d in d:
            files = os.listdir(pathlib.Path(exp, _d))

            for f in files:
                if '.db3' in f:
                    file_name = f
                    break

            print(f'Generating video for {exp}/{_d}/{file_name} from perspective {ego_car}')

            rb_data = rosbagData(pathlib.Path(exp, _d, file_name))

            # Get raw data from topics
            car1_state_raw = rb_data.read_from_topic_to_dict(f'/experiment/{car1_alias}/optitrack/odom')
            car1_control_raw = rb_data.read_from_topic_to_dict(f'/experiment/{car1_alias}/barc_control')
            car1_sol_raw = rb_data.read_from_topic_to_dict(f'/experiment/{car1_alias}/pred')
            car1_opp_pred_raw = rb_data.read_from_topic_to_dict(f'/experiment/{car1_alias}/opponent_pred')

            car2_state_raw = rb_data.read_from_topic_to_dict(f'/experiment/{car2_alias}/optitrack/odom')
            car2_control_raw = rb_data.read_from_topic_to_dict(f'/experiment/{car2_alias}/barc_control')
            car2_sol_raw = rb_data.read_from_topic_to_dict(f'/experiment/{car2_alias}/pred')
            car2_opp_pred_raw = rb_data.read_from_topic_to_dict(f'/experiment/{car2_alias}/opponent_pred')

            # Get data into numpy array form and create interpolators
            car1_state = []
            for st in car1_state_raw:
                # Time stamp
                t = st['header']['stamp']['sec'] + st['header']['stamp']['nanosec']/1e9
                
                # Position
                p = np.array([t, st['pose']['pose']['position']['x'], st['pose']['pose']['position']['y'], st['pose']['pose']['position']['z']])
                
                # Orientation
                R = Rotation.from_quat([st['pose']['pose']['orientation']['x'],
                                        st['pose']['pose']['orientation']['y'],
                                        st['pose']['pose']['orientation']['z'],
                                        st['pose']['pose']['orientation']['w']])
                ez, ey, ex = R.as_euler('ZYX') # yaw, pitch, roll
                e = np.array([t, ez, ey, ex])
                
                # Linear velocity
                v = np.array([t, st['twist']['twist']['linear']['x'], st['twist']['twist']['linear']['y'], st['twist']['twist']['linear']['z']])
                
                # Angular velocity
                w = np.array([t, st['twist']['twist']['angular']['x'], st['twist']['twist']['angular']['y'], st['twist']['twist']['angular']['z']])

                # Get the position of the car frame in the track frame
                p[1:] = opti2track_R.inv().apply(p[1:] - opti2track_p)

                # Orientation w.r.t. track global frame
                e[1:] = e[1:] - car2opti_R.as_euler('ZYX') - opti2track_R.as_euler('ZYX')

                # Angular velocity in car body frame
                w[1:] = car2opti_R.apply(w[1:])

                # Linear velocity measured at CoM
                v[1:] = car2opti_R.apply(v[1:]) + np.cross(w[1:], -car2opti_p)

                s, ey, ep = track.global_to_local((p[1], p[2], e[1]))

                car1_state.append([t, p[1], p[2], e[1], v[1], v[2], w[3], ep, s, ey])
                
            car1_state = np.array(car1_state)
            car1_state_interp = interp1d(car1_state[:,0], car1_state[:,1:], kind='linear', axis=0, assume_sorted=True)
            car1_state_t_range = [car1_state[0,0], car1_state[-1,0]]

            car1_control = []
            for u in car1_control_raw:
                t = u['t']
                
                ua, us = u['u_a'], u['u_steer']
                
                car1_control.append([t, ua, us])
                
            car1_control =np.array(car1_control)
            car1_control_interp = interp1d(car1_control[:,0], car1_control[:,1:], kind='previous', axis=0, assume_sorted=True)
            car1_control_t_range = [car1_control[0,0], car1_control[-1,0]]

            car1_sol = []
            for pr in car1_sol_raw:
                t = pr['t']

                s, ey, ep = np.array(pr['s']), np.array(pr['x_tran']), np.array(pr['e_psi'])
                x, y, p = [], [], []
                for _s, _ey, _ep in zip(s, ey, ep):
                    _x, _y, _p = track.local_to_global((_s, _ey, _ep))
                    x.append(_x)
                    y.append(_y)
                    p.append(_p)
                x = np.array(x)
                y = np.array(y)
                p = np.array(p)

                vl, vt, w = np.array(pr['v_long']), np.array(pr['v_tran']), np.array(pr['psidot'])

                car1_sol.append([t, x, y, p, vl, vt, w])

            _car1_sol_t = np.array([p[0] for p in car1_sol])
            _car1_sol = np.array([np.array(p[1:]).T for p in car1_sol])
            _car1_sol_interp = interp1d(_car1_sol_t, _car1_sol, kind='previous', axis=0, assume_sorted=True)
            car1_sol_interp = lambda t: _car1_sol_interp(max(min(t, car1_sol[-1][0]), car1_sol[0][0]))
            car1_sol_t_range = [car1_sol[0][0], car1_sol[-1][0]]

            car2_state = []
            for st in car2_state_raw:
                # Time stamp
                t = st['header']['stamp']['sec'] + st['header']['stamp']['nanosec']/1e9
                
                # Position
                p = np.array([t, st['pose']['pose']['position']['x'], st['pose']['pose']['position']['y'], st['pose']['pose']['position']['z']])
                
                # Orientation
                R = Rotation.from_quat([st['pose']['pose']['orientation']['x'],
                                        st['pose']['pose']['orientation']['y'],
                                        st['pose']['pose']['orientation']['z'],
                                        st['pose']['pose']['orientation']['w']])
                ez, ey, ex = R.as_euler('ZYX') # yaw, pitch, roll
                e = np.array([t, ez, ey, ex])
                
                # Linear velocity
                v = np.array([t, st['twist']['twist']['linear']['x'], st['twist']['twist']['linear']['y'], st['twist']['twist']['linear']['z']])
                
                # Angular velocity
                w = np.array([t, st['twist']['twist']['angular']['x'], st['twist']['twist']['angular']['y'], st['twist']['twist']['angular']['z']])

                # Get the position of the car frame in the track frame
                p[1:] = opti2track_R.inv().apply(p[1:] - opti2track_p)

                # Orientation w.r.t. track global frame
                e[1:] = e[1:] - car2opti_R.as_euler('ZYX') - opti2track_R.as_euler('ZYX')

                # Angular velocity in car body frame
                w[1:] = car2opti_R.apply(w[1:])

                # Linear velocity measured at CoM
                v[1:] = car2opti_R.apply(v[1:]) + np.cross(w[1:], -car2opti_p)

                s, ey, ep = track.global_to_local((p[1], p[2], e[1]))

                car2_state.append([t, p[1], p[2], e[1], v[1], v[2], w[3], ep, s, ey])

            car2_state = np.array(car2_state)
            car2_state_interp = interp1d(car2_state[:,0], car2_state[:,1:], kind='linear', axis=0, assume_sorted=True)
            car2_state_t_range = [car2_state[0,0], car2_state[-1,0]]

            car2_control = []
            for u in car2_control_raw:
                t = u['t']
                
                ua, us = u['u_a'], u['u_steer']
                
                car2_control.append([t, ua, us])
                
            car2_control =np.array(car2_control)
            car2_control_interp = interp1d(car2_control[:,0], car2_control[:,1:], kind='previous', axis=0, assume_sorted=True)
            car2_control_t_range = [car2_control[0,0], car2_control[-1,0]]

            car2_sol = []
            for pr in car2_sol_raw:
                t = pr['t']
                
                s, ey, ep = np.array(pr['s']), np.array(pr['x_tran']), np.array(pr['e_psi'])
                x, y, p = [], [], []
                for _s, _ey, _ep in zip(s, ey, ep):
                    _x, _y, _p = track.local_to_global((_s, _ey, _ep))
                    x.append(_x)
                    y.append(_y)
                    p.append(_p)
                x = np.array(x)
                y = np.array(y)
                p = np.array(p)

                vl, vt, w = np.array(pr['v_long']), np.array(pr['v_tran']), np.array(pr['psidot'])

                car2_sol.append([t, x, y, p, vl, vt, w])

            _car2_sol_t = np.array([p[0] for p in car2_sol])
            _car2_sol = np.array([np.array(p[1:]).T for p in car2_sol])
            _car2_sol_interp = interp1d(_car2_sol_t, _car2_sol, kind='previous', axis=0, assume_sorted=True)
            car2_sol_interp = lambda t: _car2_sol_interp(max(min(t, car2_sol[-1][0]), car2_sol[0][0]))
            car2_sol_t_range = [car2_sol[0][0], car2_sol[-1][0]]
            
            if ego_car == 'car1':
                opp_pred_raw = car1_opp_pred_raw
            elif ego_car == 'car2':
                opp_pred_raw = car2_opp_pred_raw
            ego_pred = []
            for pr in opp_pred_raw:
                t = pr['t']
                
                x, y, p = np.array(pr['x']), np.array(pr['y']), np.array(pr['psi'])

                ego_pred.append([t, x, y, p])

            _ego_pred_t = np.array([p[0] for p in ego_pred])
            _ego_pred = np.array([np.array(p[1:]).T for p in ego_pred])
            _ego_pred_interp = interp1d(_ego_pred_t, _ego_pred, kind='previous', axis=0, assume_sorted=True)
            ego_pred_interp = lambda t: _ego_pred_interp(max(min(t, ego_pred[-1][0]), ego_pred[0][0]))
            ego_pred_t_range = [ego_pred[0][0], ego_pred[-1][0]]

            # Generate video
            fig = plt.figure(figsize=(20,10))
            ax_xy = fig.add_subplot(1,2,1)
            ax_v = fig.add_subplot(3,2,2)
            ax_a = fig.add_subplot(3,2,4)
            ax_s = fig.add_subplot(3,2,6)
            track.plot_map(ax_xy)
            if show_raceline:
                ax_xy.plot(raceline_mat[:,0], raceline_mat[:,1], 'r')
            ax_xy.set_aspect('equal')
            ax_xy.set_xlabel('X [m]', fontsize=15)
            ax_xy.set_ylabel('Y [m]', fontsize=15)
            ax_xy.tick_params(axis='both', which='major', labelsize=15)
            xy_lims = [ax_xy.get_xlim(), ax_xy.get_ylim()]
            ax_v.set_ylabel('vel [m/s]', fontsize=15)
            ax_v.tick_params(axis='y', which='major', labelsize=15)
            ax_v.xaxis.set_ticklabels([])
            ax_v.yaxis.set_label_coords(-0.12, 0.5)
            ax_a.set_ylabel('accel [m/s^2]', fontsize=15)
            ax_a.tick_params(axis='y', which='major', labelsize=15)
            ax_a.xaxis.set_ticklabels([])
            ax_a.yaxis.set_label_coords(-0.12, 0.5)
            ax_a.set_yticks([-1, 0, 1])
            ax_s.set_ylabel('steer [rad]', fontsize=15)
            ax_s.tick_params(axis='y', which='major', labelsize=15)
            ax_s.set_yticks([-0.436, 0, 0.436])
            ax_s.yaxis.set_label_coords(-0.12, 0.5)
            ax_s.set_xlabel('time [s]', fontsize=15)
            ax_s.tick_params(axis='x', which='major', labelsize=15)

            car1_l_v = ax_v.plot([], [], '-b')[0]
            car1_l_a = ax_a.plot([], [], 'b')[0]
            car1_l_s = ax_s.plot([], [], 'b')[0]

            car1_sol_rects = []
            for k in range(len(car1_sol[0][1])):
                _r = patches.Rectangle((0, 0), VL, VW, 
                                        linestyle='solid', 
                                        edgecolor=car1_c, 
                                        facecolor='none', 
                                        alpha=0.5, 
                                        rotation_point='center')
                _r.set(xy=(car1_state[0][1]-VL/2, car1_state[0][2]-VW/2), angle=np.rad2deg(car1_state[0][3]))
                ax_xy.add_patch(_r)
                car1_sol_rects.append(_r)

            # car1_l_pred = ax_xy.plot([], [], '-bo', markersize=3)[0]
            # car1_l_opp_pred = ax_xy.plot([], [], '--ro', markersize=3)[0]

            # car1_l_opp_pred_cov = [ax_xy.plot([], [], '-r')[0] for _ in range(len(car1_opp_pred[0][-1]))]
            # car1_l_predictor_cov = [ax_xy.plot([], [], '-r')[0] for _ in range(len(car1_solictor[0][-1]))]

            car2_l_v = ax_v.plot([], [], '-g')[0]
            car2_l_a = ax_a.plot([], [], 'g')[0]
            car2_l_s = ax_s.plot([], [], 'g')[0]

            car2_sol_rects = []
            for k in range(len(car2_sol[0][1])):
                _r = patches.Rectangle((0, 0), VL, VW, 
                                        linestyle='solid', 
                                        edgecolor=car2_c, 
                                        facecolor='none', 
                                        alpha=0.5, 
                                        rotation_point='center')
                _r.set(xy=(car2_state[0][1]-VL/2, car2_state[0][2]-VW/2), angle=np.rad2deg(car2_state[0][3]))
                ax_xy.add_patch(_r)
                car2_sol_rects.append(_r)

            car1_rect = patches.Rectangle((-0.5*VL, -0.5*VW), VL, VW, linestyle='solid', color=car1_c, alpha=0.5)
            car2_rect = patches.Rectangle((-0.5*VL, -0.5*VW), VL, VW, linestyle='solid', color=car2_c, alpha=0.5)
            ax_xy.add_patch(car1_rect)
            ax_xy.add_patch(car2_rect)

            b_left = car1_state[0][1] - VL/2
            b_bot  = car1_state[0][2] - VW/2
            r = Affine2D().rotate_around(car1_state[0][1], car1_state[0][2], car1_state[0][3]) + ax_xy.transData
            car1_rect.set_xy((b_left,b_bot))
            car1_rect.set_transform(r)

            b_left = car2_state[0][1] - VL/2
            b_bot  = car2_state[0][2] - VW/2
            r = Affine2D().rotate_around(car2_state[0][1], car2_state[0][2], car2_state[0][3]) + ax_xy.transData
            car2_rect.set_xy((b_left,b_bot))
            car2_rect.set_transform(r)

            if show_pred:
                ego_sol_circles = []
                for k in range(ego_params['N']):
                    _c = patches.Circle((0,0), radius=0.01, edgecolor=ego_c, facecolor='none')
                    ax_xy.add_patch(_c)
                    ego_sol_circles.append(_c)

                ego_pred_circles = []
                for k in range(ego_params['N']):
                    _c = patches.Circle((0,0), radius=0.01, edgecolor=tar_c, facecolor='none')
                    ax_xy.add_patch(_c)
                    ego_pred_circles.append(_c)

            fps = 20

            for i in range(car1_state.shape[0]):
                if car1_state[i,4] > 0.1:
                    car1_start = car1_state[i,0]
                    break
            for i in range(car2_state.shape[0]):
                if car2_state[i,4] > 0.1:
                    car2_start = car2_state[i,0]
                    break
            t_start = np.amin([car1_start, car2_start])
            t_end = car1_state_t_range[1]
            # t_end = t_start + 1
            t_span = np.linspace(t_start, t_end, int((t_end-t_start)*fps))
            
            # Plot actuation bounds
            ax_a.plot([-1, t_end-t_start+1], [1.0, 1.0], 'k')
            ax_a.plot([-1, t_end-t_start+1], [-1.0, -1.0], 'k')
            ax_s.plot([-1, t_end-t_start+1], [0.436, 0.436], 'k')
            ax_s.plot([-1, t_end-t_start+1], [-0.436, -0.436], 'k')

            buff_len = 5*fps
            t_buff = deque([], maxlen=buff_len)
            car1_v_buff = deque([], maxlen=buff_len)
            car1_a_buff = deque([], maxlen=buff_len)
            car1_s_buff = deque([], maxlen=buff_len)
            car2_a_buff = deque([], maxlen=buff_len)
            car2_v_buff = deque([], maxlen=buff_len)
            car2_s_buff = deque([], maxlen=buff_len)

            car1_x_prev = car1_state_interp(t_start)
            car2_x_prev = car2_state_interp(t_start)
            
            if show_value:
                XYV = []
                V_min, V_max = np.inf, -np.inf
                n_grid = 30
                for t in t_span:
                    if ego_pred_t_range[0] <= t <= ego_pred_t_range[1]:
                        if ego_car == 'car1':
                            ego_sol_interp = car1_sol_interp
                            tar_sol_interp = car2_sol_interp
                            ego_sol_t_range = car1_sol_t_range
                        elif ego_car == 'car2':
                            ego_sol_interp = car2_sol_interp
                            tar_sol_interp = car1_sol_interp
                            ego_sol_t_range = car2_sol_t_range

                        _ego_pred = ego_pred_interp(t)
                        _ego_sol = ego_sol_interp(t)
                        _tar_sol = tar_sol_interp(t)

                        # Evaluate value function about terminal state
                        tar_s, tar_ey, tar_ep = track.global_to_local(_ego_pred[-1])
                        tar_p = _ego_pred[-1,2]
                        tar_v = _tar_sol[-1,3]

                        s_eval = np.linspace(tar_s-3*VL, tar_s+3*VL, n_grid)
                        ey_eval = np.linspace(-track.right_width(tar_s), track.left_width(tar_s), n_grid)

                        X, Y = np.zeros((n_grid, n_grid)), np.zeros((n_grid, n_grid))
                        V = np.zeros((n_grid, n_grid))
                        S, EY = np.meshgrid(s_eval, ey_eval)
                        for i in range(S.shape[0]):
                            for j in range(EY.shape[1]):
                                X[i,j], Y[i,j], _ = track.local_to_global((S[i,j], EY[i,j], 0))
                                if EY[i,j] > track.left_width(S[i,j]) or EY[i,j] < -track.right_width(S[i,j]):
                                    V[i,j] = np.nan
                                else:
                                    ego_p = _ego_sol[-1,2]
                                    ego_v = _ego_sol[-1,3]
                                    car1_state_N = VehicleState(p=ParametricPose(s=S[i,j], x_tran=EY[i,j]), e=OrientationEuler(psi=ego_p), v=BodyLinearVelocity(v_long=ego_v))
                                    car2_state_N = VehicleState(p=ParametricPose(s=tar_s, x_tran=tar_ey), e=OrientationEuler(psi=tar_p), v=BodyLinearVelocity(v_long=tar_v))
                                    V[i,j] = value_function(car1_state_N, car2_state_N)
                        _min = np.nanmin(V)
                        _max = np.nanmax(V)
                        if _min < V_min:
                            V_min = _min
                        if _max > V_max:
                            V_max = _max
                        XYV.append([X, Y, V])
                    else:
                        XYV.append(None)

                sm = ScalarMappable(None)
                sm.set_clim(V_min, V_max)
                plt.colorbar(mappable=sm, ax=ax_xy)

                V_cm, V_co = None, None

            writer = FFMpegWriter(fps=fps)
            video_path = pathlib.Path(exp, f'{_d}_{ego_car}.mp4')
            with writer.saving(fig, video_path, 100):
                for k, t in enumerate(t_span):
                    writer.grab_frame()
                    
                    car1_x = car1_state_interp(t)

                    car1_x[2] = np.unwrap([car1_x_prev[2], car1_x[2]])[1]

                    # Filter out artifacts in heading
                    if np.abs(car1_x[2] - car1_x_prev[2]) > np.pi/4:
                        print(f'Anomalous car 1 heading found')
                        print(f'Time: {(t-t_start):.2f} | last heading: {car1_x_prev[2]:.2f} | current heading: {car1_x[2]:.2f}')
                        car1_x[2] = car1_x_prev[2]
                    car1_x_prev = copy.copy(car1_x)

                    ex, ey, ep, evx, evy, ew, _, _, _ = car1_x

                    ea, es = None, None
                    if car1_control_t_range[0] <= t <= car1_control_t_range[1]:
                        ea, es = car1_control_interp(t)
                    b_left = float(ex) - VL/2
                    b_bot  = float(ey) - VW/2
                    r = Affine2D().rotate_around(float(ex), float(ey), float(ep)) + ax_xy.transData
                    car1_rect.set_xy((b_left,b_bot))
                    car1_rect.set_transform(r)

                    tvx = None
                    if car2_state_t_range[0] <= t <= car2_state_t_range[1]:
                        car2_x = car2_state_interp(t)
                        
                        car2_x[2] = np.unwrap([car2_x_prev[2], car2_x[2]])[1]

                        # Filter out artifacts in heading
                        if np.abs(car2_x[2] - car2_x_prev[2]) > np.pi/4:
                            print(f'Anomalous car 2 heading found')
                            print(f'Time: {(t-t_start):.2f} | last heading: {car2_x_prev[2]:.2f} | current heading: {car2_x[2]:.2f}')
                            car2_x[2] = car2_x_prev[2]
                        car2_x_prev = copy.copy(car2_x)

                        tx, ty, tp, tvx, tvy, tw, _, _, _ = car2_x

                        b_left = float(tx) - VL/2
                        b_bot  = float(ty) - VW/2
                        r = Affine2D().rotate_around(float(tx), float(ty), float(tp)) + ax_xy.transData
                        car2_rect.set_xy((b_left,b_bot))
                        car2_rect.set_transform(r)

                    ta, ts = None, None
                    if car2_control_t_range[0] <= t <= car2_control_t_range[1]:
                        ta, ts = car2_control_interp(t)
                    
                    t_buff.append(t - t_span[0])
                    car1_v_buff.append(evx)
                    car1_a_buff.append(ea)
                    car1_s_buff.append(es)
                    car2_v_buff.append(tvx)
                    car2_a_buff.append(ta)
                    car2_s_buff.append(ts)

                    car1_l_v.set_data(t_buff, car1_v_buff)
                    car1_l_a.set_data(t_buff, car1_a_buff)
                    car1_l_s.set_data(t_buff, car1_s_buff)

                    car2_l_v.set_data(t_buff, car2_v_buff)
                    car2_l_a.set_data(t_buff, car2_a_buff)
                    car2_l_s.set_data(t_buff, car2_s_buff)

                    # Draw value function
                    if show_value:
                        if XYV[k] is None:
                            if V_cm is not None:
                                V_cm.remove()
                                V_co.remove()
                                V_cm, V_co = None, None
                        else:
                            if V_cm is not None:
                                V_cm.remove()
                                V_co.remove()
                            V_cm = ax_xy.pcolormesh(*XYV[k], vmin=V_min, vmax=V_max)
                            V_co = ax_xy.contour(*XYV[k], [0.0, 1.0, 2.0], cmap='Greys')

                    # Draw collision avoidance constraint boundaries
                    if show_pred:
                        if ego_pred_t_range[0] <= t <= ego_pred_t_range[1]:
                            if ego_car == 'car1':
                                ego_sol_interp = car1_sol_interp
                                tar_sol_interp = car2_sol_interp
                                ego_sol_t_range = car1_sol_t_range
                            elif ego_car == 'car2':
                                ego_sol_interp = car2_sol_interp
                                tar_sol_interp = car1_sol_interp
                                ego_sol_t_range = car2_sol_t_range

                            _ego_pred = ego_pred_interp(t)
                            _ego_sol = ego_sol_interp(t)

                            for k in range(ego_params['N']):
                                ego_sol_circles[k].remove()
                                _c = patches.Circle((_ego_sol[k+1,0], _ego_sol[k+1,1]), radius=ego_sol_r[k], edgecolor=ego_c, facecolor='none')
                                ax_xy.add_patch(_c)
                                ego_sol_circles[k] = _c

                            for k in range(ego_params['N']):
                                ego_pred_circles[k].remove()
                                _c = patches.Circle((_ego_pred[k,0], _ego_pred[k,1]), radius=ego_pred_r[k], edgecolor=tar_c, facecolor='none')
                                ax_xy.add_patch(_c)
                                ego_pred_circles[k] = _c
                        elif t > ego_pred_t_range[1]:
                            for k in range(ego_params['N']):
                                try:
                                    ego_sol_circles[k].remove()
                                    ego_pred_circles[k].remove()
                                except:
                                    pass

                    # Draw MPC solutions for the vehicles
                    if car1_sol_t_range[0] <= t <= car1_sol_t_range[1]:
                        _car1_sol = car1_sol_interp(t)
                        for k in range(_car1_sol.shape[0]):
                            car1_sol_rects[k].remove()
                            _r = patches.Rectangle((0, 0), VL, VW, 
                                                    linestyle='solid', 
                                                    edgecolor=car1_c, 
                                                    facecolor='none', 
                                                    alpha=0.5, 
                                                    rotation_point='center')
                            _r.set(xy=(_car1_sol[k,0]-VL/2, _car1_sol[k,1]-VW/2), angle=np.rad2deg(_car1_sol[k,2]))
                            ax_xy.add_patch(_r)
                            car1_sol_rects[k] = _r

                    if car2_sol_t_range[0] <= t <= car2_sol_t_range[1]:
                        _car2_sol = car2_sol_interp(t)
                        for k in range(_car2_sol.shape[0]):
                            car2_sol_rects[k].remove()
                            _r = patches.Rectangle((0, 0), VL, VW, 
                                                    linestyle='solid', 
                                                    edgecolor=car2_c, 
                                                    facecolor='none', 
                                                    alpha=0.5, 
                                                    rotation_point='center')
                            _r.set(xy=(_car2_sol[k,0]-VL/2, _car2_sol[k,1]-VW/2), angle=np.rad2deg(_car2_sol[k,2]))
                            ax_xy.add_patch(_r)
                            car2_sol_rects[k] = _r

                    ax_xy.set_title(f't = {(t-t_span[0]):.2f} s', fontsize=15)
                    ax_xy.set_ylim(xy_lims[1])
                    ax_xy.set_xlim(xy_lims[0])
                    
                    ax_v.set_ylim([0, 3.5])
                    ax_v.set_xlim([t_buff[0], t_buff[-1]])
                    ax_v.autoscale_view(scalex=True, scaley=False)

                    ax_a.set_ylim([-1.2, 1.2])
                    ax_a.set_xlim([t_buff[0], t_buff[-1]])
                    ax_a.autoscale_view(scalex=True, scaley=False)

                    ax_s.set_ylim([-0.46, 0.46])
                    ax_s.set_xlim([t_buff[0], t_buff[-1]])
                    ax_s.autoscale_view(scalex=True, scaley=False)

                writer.grab_frame()

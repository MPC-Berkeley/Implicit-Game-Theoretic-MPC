#!/usr/bin/env python3

import numpy as np
import scipy as sp
import casadi as ca

from scipy.interpolate import interp1d

import pathlib
import pickle
import pdb
from datetime import datetime
import copy

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import Normalize
from matplotlib.transforms import Affine2D
from matplotlib.animation import FFMpegWriter
import matplotlib
matplotlib.use('TkAgg')
# plt.rcParams['text.usetex'] = True

import torch
import os
from model import mlp

from mpclab_controllers.CA_LTV_MPC import CA_LTV_MPC
from mpclab_controllers.utils.controllerTypes import CALTVMPCParams

from mpclab_simulation.dynamics_simulator import DynamicsSimulator

from mpclab_common.pytypes import VehicleState, VehicleActuation, VehiclePrediction, Position, ParametricPose, BodyLinearVelocity, OrientationEuler, BodyAngularVelocity
from mpclab_common.models.dynamics_models import CasadiDynamicCLBicycle
from mpclab_common.models.model_types import DynamicBicycleConfig
from mpclab_common.track import get_track, load_mpclab_raceline

model_dir = os.getcwd()

data_file = 'processed_data_progress_diff_racing.pkl'
model_file = 'V_GT_racing.pt'

data_path = model_dir.joinpath(data_file)
with open(data_path, 'rb') as f:
    D = pickle.load(f)

n_in = len(D['test'][0][0])
model_path = model_dir.joinpath(model_file)
model = mlp(input_layer_size=len(D['test'][0][0]),
            output_layer_size=len(D['test'][0][1]),
            hidden_layer_sizes=[48, 48], 
            activation='tanh', 
            batch_norm=False)
model.load_state_dict(torch.load(model_path))
model.to(torch.device('cuda'))
model.eval()

# Output directory for evaluation plots
time_str = datetime.now().strftime('%m-%d-%Y_%H-%M-%S')
out_dir = model_dir.joinpath(f'eval_{time_str}')
out_dir.mkdir(parents=True)

VL = 0.37
VW = 0.195

track_name = 'L_track_barc_reverse'
track = get_track(track_name)
H = track.half_width
L = track.track_length

path = f'{track_name}_raceline.npz'
raceline, s2t, raceline_mat = load_mpclab_raceline(path, track_name, time_scale=1.7)

N = 20
dt = 0.1
discretization_method = 'rk4'

t = 0.0

dynamics_config = DynamicBicycleConfig(dt=dt,
                                        model_name='dynamic_bicycle',
                                        noise=False,
                                        discretization_method=discretization_method,
                                        simple_slip=False,
                                        tire_model='pacejka',
                                        mass=2.2187,
                                        yaw_inertia=0.02723,
                                        wheel_friction=0.9,
                                        pacejka_b_front=5.0,
                                        pacejka_b_rear=5.0,
                                        pacejka_c_front=2.28,
                                        pacejka_c_rear=2.28,
                                        use_mx=True,
                                        M=10)
dyn_model = CasadiDynamicCLBicycle(t, dynamics_config, track=track)

sim_dynamics_config = DynamicBicycleConfig(dt=0.01,
                                        model_name='dynamic_bicycle',
                                        noise=False,
                                        discretization_method=discretization_method,
                                        simple_slip=False,
                                        tire_model='pacejka',
                                        mass=2.2187,
                                        yaw_inertia=0.02723,
                                        wheel_friction=0.9,
                                        pacejka_b_front=5.0,
                                        pacejka_b_rear=5.0,
                                        pacejka_c_front=2.28,
                                        pacejka_c_rear=2.28,
                                        use_mx=True,
                                        M=10)
dynamics_simulator = DynamicsSimulator(t, sim_dynamics_config, track=track)

state_input_ub = VehicleState(x=Position(x=1e9, y=1e9),
                            e=OrientationEuler(psi=1e9),
                            p=ParametricPose(s=2*L, x_tran=H, e_psi=1e9),
                            v=BodyLinearVelocity(v_long=5, v_tran=2),
                            w=BodyAngularVelocity(w_psi=10),
                            u=VehicleActuation(u_a=2.0, u_steer=0.436))
state_input_lb = VehicleState(x=Position(x=-1e9, y=-1e9),
                            e=OrientationEuler(psi=-1e9),
                            p=ParametricPose(s=-2*L, x_tran=-H, e_psi=-1e9),
                            v=BodyLinearVelocity(v_long=0.1, v_tran=-2),
                            w=BodyAngularVelocity(w_psi=-10),
                            u=VehicleActuation(u_a=-2.0, u_steer=-0.436))
input_rate_ub = VehicleState(u=VehicleActuation(u_a=20.0, u_steer=4.5))
input_rate_lb = VehicleState(u=VehicleActuation(u_a=-20.0, u_steer=-4.5))

# Symbolic placeholder variables
sym_q = ca.SX.sym('q', dyn_model.n_q)
sym_u = ca.SX.sym('u', dyn_model.n_u)
sym_du = ca.SX.sym('du', dyn_model.n_u)

s_idx = 4
ey_idx = 5

ua_idx = 0
us_idx = 1

sym_q_ref = ca.SX.sym('q_ref', dyn_model.n_q)

Q = np.diag([1, 0, 0, 1, 1, 1])
sym_state_stage = 0.5*ca.bilin(Q, sym_q-sym_q_ref, sym_q-sym_q_ref)
sym_state_term = 0.5*ca.bilin(Q, sym_q-sym_q_ref, sym_q-sym_q_ref)

sym_input_stage = 0.5*(1e-2*(sym_u[ua_idx])**2 + 1e-2*(sym_u[us_idx])**2)
sym_input_term = 0.5*(1e-2*(sym_u[ua_idx])**2 + 1e-2*(sym_u[us_idx])**2)

sym_rate_stage = 0.5*(0.1*(sym_du[ua_idx])**2 + 0.1*(sym_du[us_idx])**2)

sym_costs = {'state': [None for _ in range(N+1)], 'input': [None for _ in range(N+1)], 'rate': [None for _ in range(N)]}
for k in range(N):
    sym_costs['state'][k] = ca.Function(f'state_stage_{k}', [sym_q, sym_q_ref], [sym_state_stage])
    sym_costs['input'][k] = ca.Function(f'input_stage_{k}', [sym_u], [sym_input_stage])
    sym_costs['rate'][k] = ca.Function(f'rate_stage_{k}', [sym_du], [sym_rate_stage])
sym_costs['state'][N] = ca.Function('state_term', [sym_q, sym_q_ref], [sym_state_term])
sym_costs['input'][N] = ca.Function('input_term', [sym_u], [sym_input_term])

# left_boundary = sym_q[ey_idx] - track.left_width(sym_q[s_idx])
# right_boundary = -track.right_width(sym_q[s_idx]) - sym_q[ey_idx]
# track_boundary_constraint = ca.Function('track_boundary', [sym_q, sym_u], [ca.vertcat(right_boundary, left_boundary)])
# sym_constrs = {'state_input': [None] + [track_boundary_constraint for _ in range(N)], 
#                 'rate': [None for _ in range(N)]}

sym_constrs = {'state_input': [None for _ in range(N+1)], 
                'rate': [None for _ in range(N)]}


mpc_params = CALTVMPCParams(N=N,
                            state_scaling=[10, 10, 7, np.pi/2, L, 1.0],
                            input_scaling=[2, 0.45],
                            soft_state_bound_idxs=[ey_idx],
                            soft_state_bound_quad=[1e6],
                            soft_state_bound_lin=[0],
                            damping=0.75,
                            qp_iters=2,
                            delay=None,
                            verbose=False,
                            qp_interface='hpipm')

mpc_controller = CA_LTV_MPC(dyn_model, 
                            sym_costs, 
                            sym_constrs, 
                            {'qu_ub': state_input_ub, 'qu_lb': state_input_lb, 'du_ub': input_rate_ub, 'du_lb': input_rate_lb},
                            mpc_params)

# dey_vals = [-0.2, 0, 0.2]
# dv_vals = [-0.75, -0.3, 0, 0.3, 0.75]
dey_vals = [0]
dv_vals = [0]

dp_vals = [-20*np.pi/180, -10*np.pi/180, 0, 10*np.pi/180, 20*np.pi/180]
dp_c = ['Greys', 'Purples', 'Blues', 'Greens', 'Oranges']

for dey in dey_vals:
    for dv in dv_vals:
        print(f'Evaluating dey: {dey}, dv: {dv}')
        scaling = 1.0

        sim_state = VehicleState(t=0.0)
        s0 = 1
        t0 = float(s2t(s0))
        _, _, _, _v, _, _, _ep, _, _ey = np.array(raceline(s2t(s0))).squeeze()
        sim_state.p.s = s0
        sim_state.p.x_tran = _ey + dey
        sim_state.p.e_psi = _ep
        sim_state.v.v_long = _v
        track.local_to_global_typed(sim_state)

        t_ref = t0 + dt*np.arange(N+1)*scaling
        q_ref = np.array(raceline(t_ref)).squeeze().T[:,3:]
        q_ref[:,:3] *= scaling
        q_ref[:,ey_idx] += dey
        P = q_ref.ravel()

        u_ws = 0.01*np.ones((N+1, dyn_model.n_u))
        du_ws = 0.01*np.ones((N, dyn_model.n_u))

        mpc_controller.set_warm_start(u_ws, du_ws, state=sim_state, parameters=P)

        sim_state.u.u_a, sim_state.u.u_steer = 0.0, 0.0
        pred = VehiclePrediction()
        control = VehicleActuation(t=t, u_a=0, u_steer=0)

        t_log, x_log, y_log, p_log, XYPV_log = [], [], [], [], []
        if n_in == 6:
            n_grid = 30
        elif n_in == 8:
            n_grid = 10
        V_min, V_max = np.inf, -np.inf
        while sim_state.p.s <= L - 1:
            print(L - sim_state.p.s)

            state = copy.deepcopy(sim_state)

            # Evaluate value function about current state
            s_eval = np.linspace(state.p.s-3*VL, state.p.s+3*VL, n_grid)
            ey_eval = np.linspace(state.p.x_tran-H, state.p.x_tran+H, n_grid)
            v_eval = state.v.v_long + dv

            X, Y, P = np.zeros((n_grid, n_grid)), np.zeros((n_grid, n_grid)), np.zeros((n_grid, n_grid))
            if n_in == 6:
                V = np.zeros((n_grid, n_grid))
            elif n_in == 8:
                V = np.zeros((n_grid, n_grid, len(dp_vals)))
            S, EY = np.meshgrid(s_eval, ey_eval)
            for i in range(S.shape[0]):
                for j in range(EY.shape[1]):
                    X[i,j], Y[i,j], _ = track.local_to_global((S[i,j], EY[i,j], 0))
                    P[i,j] = state.e.psi
                    if EY[i,j] > track.left_width(S[i,j]) or EY[i,j] < -track.right_width(S[i,j]):
                        if n_in == 6:
                            V[i,j] = np.nan
                        elif n_in == 8:
                            V[i,j,:] = np.nan
                    else:
                        if n_in == 6:
                            car2_f = np.array([state.v.v_long, state.p.s, state.p.x_tran])
                            car1_df = np.array([dv, S[i,j]-state.p.s, EY[i,j]-state.p.x_tran])
                            f = np.concatenate((car2_f, car1_df))
                            f_norm = sp.linalg.solve(sp.linalg.sqrtm(D['train'].feature_cov), f - D['train'].feature_mean, assume_a='pos')
                            V[i,j] = float(model(torch.tensor(f_norm.reshape((1,-1))).cuda()).cpu())*D['train'].target_cov + D['train'].target_mean
                        elif n_in == 8:
                            for k, dp in enumerate(dp_vals):
                                car2_f = np.array([state.v.v_long, state.e.psi, state.p.s, state.p.x_tran])
                                car1_df = np.array([dv, dp, S[i,j]-state.p.s, EY[i,j]-state.p.x_tran])
                                f = np.concatenate((car2_f, car1_df))
                                f_norm = sp.linalg.solve(sp.linalg.sqrtm(D['train'].feature_cov), f - D['train'].feature_mean, assume_a='pos')
                                V[i,j,k] = float(model(torch.tensor(f_norm.reshape((1,-1))).cuda()).cpu())*D['train'].target_cov + D['train'].target_mean

            _V_min = np.nanmin(V)
            _V_max = np.nanmax(V)
            if _V_min < V_min:
                V_min = _V_min
            if _V_max > V_max:
                V_max = _V_max

            t_log.append(t)
            x_log.append(state.x.x)
            y_log.append(state.x.y)
            p_log.append(state.e.psi)
            XYPV_log.append((X, Y, P, V))

            # Solve reference tracking problem
            t_ref = float(s2t(state.p.s)) + dt*np.arange(N+1)*scaling
            q_ref = np.array(raceline(t_ref)).squeeze().T[:,3:]
            q_ref[:,:3] *= scaling
            q_ref[:,ey_idx] += dey

            mpc_controller.step(state, parameters=q_ref.ravel())
            state.copy_control(control)
            pred = mpc_controller.get_prediction()
            pred.t = t

            # Simulate
            t += dt
            sim_state = copy.deepcopy(state)
            dynamics_simulator.step(sim_state, T=dt)

            pdb.set_trace()

        # Interpolators for saved data
        _x = interp1d(t_log, x_log, kind='linear', assume_sorted=True)
        _y = interp1d(t_log, y_log, kind='linear', assume_sorted=True)
        _p = interp1d(t_log, p_log, kind='linear', assume_sorted=True)

        _X = interp1d(t_log, [_XYPV[0] for _XYPV in XYPV_log], kind='linear', assume_sorted=True, axis=0)
        _Y = interp1d(t_log, [_XYPV[1] for _XYPV in XYPV_log], kind='linear', assume_sorted=True, axis=0)

        fps = 20
        T = t_log[-1] - t_log[0]
        t_span = np.linspace(t_log[0], t_log[-1], int(T*fps+1))

        # Initialize plot
        fig = plt.figure()
        ax_xy = fig.gca()
        track.plot_map(ax_xy)
        ax_xy.plot(raceline_mat[:,0], raceline_mat[:,1], 'r')
        rect = patches.Rectangle((x_log[0]-0.5*VL, y_log[0]-0.5*VW), VL, VW, linestyle='solid', color='b', alpha=0.5)
        r = Affine2D().rotate_around(x_log[0], y_log[0], p_log[0]) + ax_xy.transData
        rect.set_transform(r)
        ax_xy.add_patch(rect)
        # ax_xy.set_xlim([x_log[0] - 5, x_log[0] + 5])
        # ax_xy.set_ylim([y_log[0] - 5, y_log[0] + 5])
        ax_xy.set_aspect('equal')
        ax_xy.set_title(f'dey: {dey} | dv: {dv}')

        # Initialize video writer
        writer = FFMpegWriter(fps=fps)
        video_path = out_dir.joinpath(f'dey_{dey}_dv_{dv}.mp4')
        with writer.saving(fig, video_path, 300):
            for i, t in enumerate(t_span):
                print(f'Grabbing frame {i+1}/{len(t_span)}')
                writer.grab_frame()

                X = _X(t)
                Y = _Y(t)
                _k = np.argmin(np.abs(np.array(t_log)-t))
                XYPV = XYPV_log[_k]

                if n_in == 6:
                    if i > 0:
                        cm.remove()
                        zl.remove()
                    cm = ax_xy.pcolormesh(X, Y, XYPV[-1], vmin=V_min, vmax=V_max)
                    zl = ax_xy.contour(X, Y, XYPV[-1], [0.0], colors='m')
                    if i == 0:
                        cb = plt.colorbar(mappable=cm, ax=ax_xy)
                elif n_in == 8:
                    d = 0.3
                    if i > 0:
                        for _q in q:
                            _q.remove()
                    q = []
                    for j, dp in enumerate(dp_vals):
                        U, V = d*np.cos(XYPV[2] + dp), d*np.sin(XYPV[2] + dp)
                        q.append(ax_xy.quiver(X, Y, U, V, XYPV[-1][:,:,j], 
                                              angles='xy', 
                                              scale_units='xy', 
                                              scale=1, 
                                              width=0.002, 
                                              headwidth=2, 
                                              norm=Normalize(vmin=V_min, vmax=V_max), 
                                              cmap='plasma'))
                
                x = _x(t)
                y = _y(t)
                psi = _p(t)

                rect.remove()
                rect = patches.Rectangle((x-VL/2, y-VW/2), VL, VW, linestyle='solid', color='b', alpha=0.5)
                r = Affine2D().rotate_around(x, y, psi) + ax_xy.transData
                rect.set_transform(r)
                ax_xy.add_patch(rect)
                # ax_xy.set_xlim([x - 5, x + 5])
                # ax_xy.set_ylim([y - 5, y + 5])
                ax_xy.set_aspect('equal')
                

#!/usr/bin python3

import numpy as np
import casadi as ca
import matplotlib.pyplot as plt

import copy
from datetime import datetime
import os
import shutil
import pathlib
import pdb

from mpclab_common.models.dynamics_models import CasadiDynamicsModel
from mpclab_common.pytypes import VehicleState, VehiclePrediction

from mpclab_controllers.abstract_controller import AbstractController
from mpclab_controllers.utils.controllerTypes import iLQRParams

class iLQR_RHC(AbstractController):
    def __init__(self, dynamics: CasadiDynamicsModel, costs, track, params=iLQRParams()):
        self.dynamics = dynamics
        self.track = track

        # Symbolic cost functions formulated using casadi
        self.costs_sym = costs

        self.trajectory_cost_prev = np.inf

        self.N = params.N
        self.max_iters = params.max_iters
        self.verbose = params.verbose
        self.code_gen = params.code_gen
        self.jit = params.jit
        self.opt_flag = params.opt_flag
        self.solver_name = params.solver_name
        self.ddp = params.ddp

        if not params.enable_jacobians:
            jac_opts = dict(enable_fd=False, enable_jacobian=False, enable_forward=False, enable_reverse=False)
        else:
            jac_opts = dict()

        if self.code_gen:
            if self.jit:
                self.options = dict(jit=True, jit_name=self.solver_name, compiler='shell', jit_options=dict(compiler='gcc', flags=['-%s' % self.opt_flag], verbose=self.verbose), **jac_opts)
            else:
                self.options = dict(jit=False, **jac_opts)
                self.c_file_name = self.solver_name + '.c'
                self.so_file_name = self.solver_name + '.so'
                if params.solver_dir is not None:
                    self.solver_dir = pathlib.Path(params.solver_dir).expanduser().joinpath(self.solver_name)
        else:
            self.options = dict(jit=False, **jac_opts)

        # self.debug = params.debug
        # if self.debug: self.verbose = True

        self.state_input_prediction = VehiclePrediction()

        self.n_u = self.dynamics.n_u
        self.n_q = self.dynamics.n_q

        # if len(self.costs_sym) != self.N+1:
        #     raise ValueError('Number of stage cost functions expected: %i, number provided: %i' % (self.N+1, len(self.costs_sym)))

        self.tol = params.tol
        self.rel_tol = params.rel_tol

        self.control_reg_init = params.control_reg
        self.state_reg_init = params.state_reg

        self.debug_plot = params.debug_plot
        self.pause_on_plot = params.pause_on_plot
        self.local_pos = params.local_pos
        if self.debug_plot:
            plt.ion()
            self.fig = plt.figure(figsize=(10,5))
            self.ax_xy = self.fig.add_subplot(1,2,1)
            # self.dynamics.track.remove_phase_out()
            self.dynamics.track.plot_map(self.ax_xy, close_loop=False)
            self.l1_xy = self.ax_xy.plot([], [], 'bo')[0]
            self.ax_a = self.fig.add_subplot(2,2,2)
            self.l1_a = self.ax_a.plot([], [], '-bo')[0]
            self.ax_s = self.fig.add_subplot(2,2,4)
            self.l1_s = self.ax_s.plot([], [], '-bo')[0]
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()

        self.alpha = np.power(10, np.linspace(0, -3, 5)) # backtracking line search coefficients

        self.q_nom = np.zeros((self.N+1, self.n_q))
        self.u_nom = np.zeros((self.N, self.n_u))

        self.u_prev = np.zeros(self.n_u)

        # Symbolic quadratic approximations
        self.ilqr_sym = None
        # if self.debug:
        #     self.debug_fns = [None for _ in range(self.N)]

        if params.solver_dir:
            self._load_solver()
        else:
            self._build_solver()

        self.initialized = True

        # self.pool = mp.Pool(processes=len(self.alpha)) # Set up pool of workers to do line search in parallel

    def initialize(self, u_init=None):
        if u_init is not None:
            self.u_nom = u_init

        self.initialized = True

    def step(self, vehicle_state: VehicleState, env_state=None):
        info = self.solve(vehicle_state)

        self.dynamics.qu2state(vehicle_state, None, self.u_nom[0])
        self.dynamics.qu2prediction(self.state_input_prediction, self.q_nom, self.u_nom)
        self.state_input_prediction.t = vehicle_state.t

        self.u_prev = self.u_nom[0]

        return

    def solve(self, vehicle_state: VehicleState):
        init_start = datetime.now()
        converged = False
        # Rollout nominal belief trajectory using policies from last step
        self.q_nom[0] = self.dynamics.state2q(vehicle_state)
        self.u_nom = np.vstack((self.u_nom[1:], self.u_nom[-1].reshape((1,-1)), self.u_prev.reshape((1,-1))))

        cost_nom = 0
        for k in range(self.N):
            args_cost = [self.q_nom[k], self.u_nom[k], self.u_nom[k-1]]
            cost_nom += float(self.costs_sym['stage'](*args_cost))

            args = [self.q_nom[k], self.u_nom[k]]
            self.q_nom[k+1] = self.dynamics.fd(*args).toarray().squeeze()

        cost_nom += float(self.costs_sym['term'](self.q_nom[-1]))
        cost_prev = copy.copy(cost_nom)

        q_nom = ca.DM(self.q_nom.T)
        u_nom = ca.DM(self.u_nom.T)

        self.control_reg = copy.copy(self.control_reg_init)
        self.state_reg = copy.copy(self.state_reg_init)

        if self.debug_plot:
            self._update_debug_plot(q_nom, u_nom)
            if self.pause_on_plot:
                ilqr_in = ca.horzsplit(q_nom, 1) + ca.horzsplit(u_nom, 1) + [self.state_reg, self.control_reg]
                pdb.set_trace()
        
        # Do iLQG iterations
        converged = False
        init_time = (datetime.now()-init_start).total_seconds()
        for i in range(self.max_iters):
            # Do backward and forward pass
            iter_start_time = datetime.now()
            ilqg_in = ca.horzsplit(q_nom, 1) + ca.horzsplit(u_nom, 1) + [self.state_reg, self.control_reg]
            ilqg_out = self.ilqr_sym(*ilqg_in)

            # Do line search
            cost = np.inf
            for j in range(len(self.alpha)):
                b_j, u_j, cost_j = ilqg_out[j*3:(j+1)*3]
                if float(cost_j) < cost:
                    cost = float(cost_j)
                    q_new, u_new = b_j, u_j

            back_forw_time = (datetime.now()-iter_start_time).total_seconds()

            if np.abs(cost - cost_nom) <= self.tol:
                converged = True
                if self.verbose:
                    print('it: %i | %g s | nom cost: %g | curr cost: %g'  % (i, back_forw_time, cost_nom, cost))
                break

            if cost <= cost_nom:
                s = 'nom cost: %g | curr cost: %g | prev cost: %g | improvement, update accepted, lowering reg' % (cost_nom, cost, cost_prev)
                self.control_reg = self.control_reg/3
                self.state_reg = self.state_reg/3

                q_nom = copy.copy(q_new)
                u_nom = ca.horzcat(u_new, ca.DM(self.u_prev))
                cost_nom = copy.copy(cost)
            else:
                if np.abs(cost - cost_prev) <= self.rel_tol:
                    s = 'nom cost: %g | curr cost: %g | prev cost: %g | no improvement but cost sequence converged, update accepted, not changing reg' % (cost_nom, cost, cost_prev)
                    q_nom = copy.copy(q_new)
                    u_nom = ca.horzcat(u_new, ca.DM(self.u_prev))
                    cost_nom = copy.copy(cost)
                else:
                    s = 'nom cost: %g | curr cost: %g | prev cost: %g | no improvement, update rejected, increasing reg' % (cost_nom, cost, cost_prev)
                    self.control_reg = self.control_reg*2
                    self.state_reg = self.state_reg*2

            cost_prev = copy.copy(cost)
            iter_time = (datetime.now()-iter_start_time).total_seconds()
            if self.verbose:
                print('it: %i | %g s | total: %g s | %s' % (i, back_forw_time, iter_time, s))

            if self.debug_plot:
                self._update_debug_plot(q_nom, u_nom)
                if self.pause_on_plot:
                    pdb.set_trace()

        self.q_nom = copy.copy(q_new.toarray().T)
        self.u_nom = copy.copy(u_new.toarray().T)
        self.trajectory_cost_prev = copy.copy(cost)

        ilqg_time = (datetime.now()-init_start).total_seconds()

        if converged:
            return_status = 'Converged in %i iterations | final cost: %g | init time: %g | total time: %g' % (i+1, cost_nom, init_time, ilqg_time)
        else:
            return_status = 'Max its reached, solution did not converge | final cost: %g | init time: %g | total time: %g' % (cost_nom, init_time, ilqg_time)
        if self.verbose:
            print(return_status)
            print('==========================')

        if self.debug_plot:
            plt.ioff()

        info = {'success': converged, 'return_status': return_status, 'solve_time': ilqg_time, 'info': None, 'output': None}

        return info

    def _build_solver(self):
        # =================================
        # Create cost computation functions
        # =================================
        # Placehold symbolic variables for nominal sequence
        q_ph = ca.MX.sym('q_ph', self.n_q) # State
        u_ph = ca.MX.sym('u_ph', self.n_u) # Input
        um_ph = ca.MX.sym('um_ph', self.n_u) # Previous input

        C = self.costs_sym['stage'](q_ph, u_ph, um_ph)
        Dqq_C, _ = ca.hessian(C, q_ph)
        Dq_C = ca.jacobian(C, q_ph)
        Duu_C, _ = ca.hessian(C, u_ph)
        Du_C = ca.jacobian(C, u_ph)
        Duq_C = ca.jacobian(Du_C, q_ph)

        args_in = [q_ph, u_ph, um_ph]
        args_out = [C, Dq_C, Du_C, Dqq_C, Duu_C, Duq_C]
        self.sym_stage_cost = ca.Function('stage_cost', args_in, args_out)

        V = self.costs_sym['term'](q_ph)
        Dqq_V, _ = ca.hessian(V, q_ph)
        Dq_V = ca.jacobian(V, q_ph)

        args_in = [q_ph]
        args_out = [V, Dq_V, Dqq_V]
        self.sym_term_cost = ca.Function('term_cost', args_in, args_out)

        # ==============================================
        # Create state-action value computation function
        # ==============================================
        # Stage cost placeholders
        C_ph = ca.MX.sym('C_ph', 1)
        Dq_C_ph = ca.MX.sym('Dq_C_ph', self.n_q)
        Du_C_ph = ca.MX.sym('Du_C_ph', self.n_u)
        Dqq_C_ph = ca.MX.sym('Dqq_C_ph', self.n_q, self.n_q)
        Duu_C_ph = ca.MX.sym('Duu_C_ph', self.n_u, self.n_u)
        Duq_C_ph = ca.MX.sym('Duq_C_ph', self.n_u, self.n_q)
        cost_args = [C_ph, Dq_C_ph, Du_C_ph, Dqq_C_ph, Duu_C_ph, Duq_C_ph]

        # Dynamics placeholders
        Dq_f_ph = ca.MX.sym('Dq_f_ph', self.n_q, self.n_q)
        Du_f_ph = ca.MX.sym('Du_f_ph', self.n_q, self.n_u)
        dyn_args = [Dq_f_ph, Du_f_ph]
        if self.ddp:
            Dqq_f_ph = [ca.MX.sym(f'Dqq_f_ph_{i}', self.n_q, self.n_q) for i in range(self.n_q)]
            Duu_f_ph = [ca.MX.sym(f'Duu_f_ph_{i}', self.n_u, self.n_u) for i in range(self.n_q)]
            Duq_f_ph = [ca.MX.sym(f'Duq_f_ph_{i}', self.n_u, self.n_q) for i in range(self.n_q)]
            dyn_args += Dqq_f_ph + Duu_f_ph + Duq_f_ph

        # Value function at next time step placeholders
        Vp_ph = ca.MX.sym('Vp_ph', 1)
        Dq_Vp_ph = ca.MX.sym('Dq_Vp_ph', self.n_q)
        Dqq_Vp_ph = ca.MX.sym('Dqq_Vp_ph', self.n_q, self.n_q)
        value_args = [Vp_ph, Dq_Vp_ph, Dqq_Vp_ph]

        # State regularization placeholder
        q_reg_ph = ca.MX.sym('q_reg_ph', 1)

        # Constant term of quadratic approx of Q
        Q = C_ph + Vp_ph

        # Linear term of quadratic approx of Q
        Dq_Q = Dq_C_ph + Dq_f_ph.T @ Dq_Vp_ph
        Du_Q = Du_C_ph + Du_f_ph.T @ Dq_Vp_ph

        # Quadratic term of quadratic approx of Q
        Dqq_Vp_reg = Dqq_Vp_ph + q_reg_ph*ca.DM.eye(self.n_q)
        Dqq_Q = Dqq_C_ph + Dq_f_ph.T @ Dqq_Vp_reg @ Dq_f_ph
        Duu_Q = Duu_C_ph + Du_f_ph.T @ Dqq_Vp_reg @ Du_f_ph
        Duq_Q = Duq_C_ph + Du_f_ph.T @ Dqq_Vp_reg @ Dq_f_ph
        if self.ddp:
            for i in range(self.n_q):
                Dqq_Q += Dq_Vp_ph[i]*Dqq_f_ph[i]
                Duu_Q += Dq_Vp_ph[i]*Duu_f_ph[i]
                Duq_Q += Dq_Vp_ph[i]*Duq_f_ph[i]
        
        args_in = cost_args \
                + value_args \
                + dyn_args \
                + [q_reg_ph]
        args_out = [Q, Dq_Q, Du_Q, Dqq_Q, Duu_Q, Duq_Q]
        self.sym_state_action_value = ca.Function('state_action_value', args_in, args_out)

        # ==============================================
        # Create feedback policy computation function
        # ==============================================
        # Stage-action value function placeholders
        Q_ph = ca.SX.sym('Q_ph', 1)
        Dq_Q_ph = ca.SX.sym('Dq_Q_ph', self.n_q)
        Du_Q_ph = ca.SX.sym('Du_Q_ph', self.n_u)
        Dqq_Q_ph = ca.SX.sym('Dqq_Q_ph', self.n_q, self.n_q)
        Duu_Q_ph = ca.SX.sym('Duu_Q_ph', self.n_u, self.n_u)
        Duq_Q_ph = ca.SX.sym('Duq_Q_ph', self.n_u, self.n_q)

        Du_Q_hat_ph = ca.SX.sym('Du_Q_hat_ph', self.n_u)
        Duu_Q_hat_ph = ca.SX.sym('Duu_Q_hat_ph', self.n_u, self.n_u)
        Duq_Q_hat_ph = ca.SX.sym('Duq_Q_hat_ph', self.n_u, self.n_q)

        # Input regularization placeholder
        u_reg_ph = ca.SX.sym('u_reg_ph', 1)

        L = Duu_Q_hat_ph + u_reg_ph*ca.DM.eye(self.n_u)
        K_fb = -ca.solve(L, Duq_Q_hat_ph)
        k_ff = -ca.solve(L, Du_Q_hat_ph)

        args_in = [Du_Q_hat_ph, Duu_Q_hat_ph, Duq_Q_hat_ph, u_reg_ph]
        args_out = [K_fb, k_ff]
        self.sym_feedback_policy = ca.Function('feedback_policy', args_in, args_out)

        # ==============================================
        # Create value function computation function
        # ==============================================
        # Feedback policy placeholders
        K_fb_ph = ca.SX.sym('K_fb_ph', self.n_u, self.n_q)
        k_ff_ph = ca.SX.sym('k_ff_ph', self.n_u)

        V = Q_ph + ca.dot(Du_Q_ph, k_ff_ph) + ca.bilin(Duu_Q_ph, k_ff_ph, k_ff_ph)/2
        Dq_V = ca.sparsify(Dq_Q_ph + K_fb_ph.T @ Duu_Q_ph @ k_ff_ph + K_fb_ph.T @ Du_Q_ph + Duq_Q_ph.T @ k_ff_ph)
        Dqq_V = Dqq_Q_ph + K_fb_ph.T @ Duu_Q_ph @ K_fb_ph + K_fb_ph.T @ Duq_Q_ph + Duq_Q_ph.T @ K_fb_ph

        args_in = [Q_ph, Dq_Q_ph, Du_Q_ph, Dqq_Q_ph, Duu_Q_ph, Duq_Q_ph, K_fb_ph, k_ff_ph]
        args_out = [V, Dq_V, Dqq_V]
        self.sym_value = ca.Function('value', args_in, args_out)

        # ==============================================
        # Do iLQG
        # ==============================================
        # Nominal state and input
        q_nom = [ca.MX.sym('q_nom_%i' % k, self.n_q) for k in range(self.N+1)] # [b_k, ..., b_k+N]
        u_nom = [ca.MX.sym('u_nom_%i' % k, self.n_u) for k in range(self.N+1)] # [u_k, ..., u_k+N-1, u_k-1]

        q_reg = ca.MX.sym('q_reg', 1)
        u_reg = ca.MX.sym('u_reg', 1)

        ilqr_in = q_nom + u_nom + [q_reg, u_reg]

        K_fb = [None for _ in range(self.N)]
        k_ff = [None for _ in range(self.N)]

        # Terminal value function is just the terminal cost
        V_out = list(self.sym_term_cost(q_nom[-1]))

        # Backward pass
        for k in range(self.N-1, -1, -1):
            # Compute linear approx of belief dynamics
            Dq_f_k = self.dynamics.fAd(q_nom[k], u_nom[k])
            Du_f_k = self.dynamics.fBd(q_nom[k], u_nom[k])
            dyn_args = [Dq_f_k, Du_f_k]
            if self.ddp:
                Dqq_f_k = list(self.dynamics.fEd(q_nom[k], u_nom[k]))
                Duu_f_k = list(self.dynamics.fFd(q_nom[k], u_nom[k]))
                Duq_f_k = list(self.dynamics.fGd(q_nom[k], u_nom[k]))
                dyn_args += Dqq_f_k + Duu_f_k + Duq_f_k

            # Compute Jacobian and Hessian of stage cost function at current time step
            args_C = [q_nom[k], u_nom[k], u_nom[k-1]]
            C_out = list(self.sym_stage_cost(*args_C))

            # Compute quadratic approx of state-action value function Q
            args_Q = C_out + V_out + dyn_args + [q_reg]
            Q_k, Dq_Q_k, Du_Q_k, Dqq_Q_k, Duu_Q_k, Duq_Q_k = self.sym_state_action_value(*args_Q)

            # Compute feedback policy
            args_policy = [Du_Q_k, Duu_Q_k, Duq_Q_k, u_reg]
            K_fb[k], k_ff[k] = self.sym_feedback_policy(*args_policy)

            # Compute quadratic approx of value function V
            args_V = [Q_k, Dq_Q_k, Du_Q_k, Dqq_Q_k, Duu_Q_k, Duq_Q_k, K_fb[k], k_ff[k]]
            V_out = list(self.sym_value(*args_V))

        # Forward pass for each line search coefficient
        ilqg_out = []
        for a in self.alpha:
            q_new = [ca.MX.sym('q_new_ph_%i' % k, self.n_q) for k in range(self.N+1)]
            u_new = [ca.MX.sym('u_new_ph_%i' % k, self.n_u) for k in range(self.N+1)]
            trajectory_cost = ca.DM.zeros(1)

            q_new[0] = q_nom[0]
            u_new[-1] = u_nom[-1]
            for k in range(self.N):
                # Compute control action
                u_new[k] = u_nom[k] + a*k_ff[k] + K_fb[k] @ (q_new[k]-q_nom[k])

                # Update belief
                q_new[k+1] = self.dynamics.fd(q_new[k], u_new[k])

                # Compute stage cost
                args_cost = [q_new[k], u_new[k], u_new[k-1]]
                trajectory_cost += self.costs_sym['stage'](*args_cost)

            # Compute terminal cost
            trajectory_cost += self.costs_sym['term'](q_new[-1])

            ilqg_out += [ca.horzcat(*q_new), ca.horzcat(*u_new[:-1]), trajectory_cost]

        # Store ilqg iteration
        self.ilqr_sym = ca.Function('ilqg_iter', ilqr_in, ilqg_out, self.options)

        if self.code_gen and not self.jit:
            generator = ca.CodeGenerator(self.c_file_name)
            generator.add(self.ilqr_sym)

            # Set up paths
            cur_dir = pathlib.Path.cwd()
            gen_path = cur_dir.joinpath(self.solver_name)
            c_path = gen_path.joinpath(self.c_file_name)
            if gen_path.exists():
                shutil.rmtree(gen_path)
            gen_path.mkdir(parents=True)

            os.chdir(gen_path)
            if self.verbose:
                print('- Generating C code for solver %s at %s' % (self.solver_name, str(gen_path)))
            generator.generate()
            # pdb.set_trace()
            # Compile into shared object
            so_path = gen_path.joinpath(self.so_file_name)
            if self.verbose:
                print('- Compiling shared object %s from %s with optimization flag -%s' % (so_path, c_path, self.opt_flag))
            os.system('gcc -fPIC -shared -%s %s -o %s' % (self.opt_flag, c_path, so_path))

            # Swtich back to working directory
            os.chdir(cur_dir)

            install_dir = self.install()
            # Load solver
            self._load_solver(install_dir.joinpath(self.so_file_name))

    def _load_solver(self, solver_path=None):
        if solver_path is None:
            solver_path = pathlib.Path(self.solver_dir, self.so_file_name).expanduser()
        if self.verbose:
            print('- Loading solver from %s' % str(solver_path))
        self.ilqr_sym = ca.external('ilqg_iter', str(solver_path))

    def get_prediction(self):
        return self.state_input_prediction

    def _update_debug_plot(self, q_nom, u_nom):
        if not self.local_pos:
            self.l1_xy.set_data(q_nom.toarray()[0,:], q_nom.toarray()[1,:])
        else:
            epsi, s, ey = q_nom.toarray()[1,:], q_nom.toarray()[2,:], q_nom.toarray()[3,:]
            global_pos = np.zeros((3, q_nom.toarray().shape[1]))
            for i in range(global_pos.shape[1]):
                global_pos[:,i] = self.track.local_to_global((s[i], ey[i], epsi[i]))
            self.l1_xy.set_data(global_pos[0,:], global_pos[1,:])
        self.ax_xy.set_aspect('equal')
        self.l1_a.set_data(np.arange(self.N), u_nom.toarray()[0,:-1])
        self.l1_s.set_data(np.arange(self.N), u_nom.toarray()[1,:-1])
        self.ax_a.relim()
        self.ax_a.autoscale_view()
        self.ax_s.relim()
        self.ax_s.autoscale_view()
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

if __name__ == '__main__':
    from mpclab_common.models.dynamics_models import CasadiDynamicCLBicycle
    from mpclab_common.models.observation_models import DynamicBicycleFullStateObserver
    from mpclab_common.models.belief_models import CasadiBeliefModel
    from mpclab_common.pytypes import VehicleConfig, ObserverConfig
    from mpclab_common.track import get_track

    t = 0
    dt = 0.1
    N = 3

    track_name = 'LTrack_barc'
    track = get_track(track_name)

    dynamics_config = VehicleConfig(dt=dt, track=track_name, model='dynamic_bicycle_cl', noise=True, noise_cov=np.diag([1e-3, 1e-3, 1e-3, 1e-3, 1e-3, 1e-3]))
    dyn_model = CasadiDynamicCLBicycle(t, dynamics_config)

    observer_config = ObserverConfig(noise=True, noise_cov=np.diag([1e-3, 1e-3, 1e-3, 1e-2, 1e-2, 1e-2]))
    obs_model = DynamicBicycleFullStateObserver(observer_config)

    dynamics = CasadiBeliefModel(t, dyn_model, obs_model)

    sym_b = ca.SX.sym('b', dynamics.n_q)
    sym_u = ca.SX.sym('u', dynamics.n_u)

    c_stage = ca.bilin(np.eye(dynamics.n_q), sym_b, sym_b) + ca.bilin(np.eye(dynamics.n_u), sym_u, sym_u)
    stage_costs = [ca.Function('f_c_stage_%i' % i, [sym_b, sym_u], [c_stage]) for i in range(N)]

    c_term = ca.bilin(np.eye(dynamics.n_q), sym_b, sym_b)
    terminal_cost = ca.Function('f_c_term', [sym_b], [c_term])

    costs = {'stage': stage_costs, 'terminal': terminal_cost}
    params = iLQRParams(dt=dt, N=N, tol=1e-3, control_reg=1e-3, state_reg=1e-3)
    controller = iLQR_RHC(dynamics, costs, track, params)

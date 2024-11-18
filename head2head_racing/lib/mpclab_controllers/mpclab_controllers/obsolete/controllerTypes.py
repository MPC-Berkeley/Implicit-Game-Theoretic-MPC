#!/usr/bin python3

from ast import boolop
from dataclasses import dataclass, field
import string
import numpy as np

from mpclab_common.pytypes import PythonMsg, VehicleState

@dataclass
class ControllerConfig(PythonMsg):
    dt: float = field(default=0.1)

@dataclass
class PIDParams(ControllerConfig):
    Kp: float = field(default=2.0)
    Ki: float = field(default=0.0)
    Kd: float = field(default=0.0)

    int_e_max: float = field(default=100)
    int_e_min: float = field(default=-100)
    u_max: float = field(default=None)
    u_min: float = field(default=None)
    du_max: float = field(default=None)
    du_min: float = field(default=None)

    u_ref: float = field(default=0.0)
    x_ref: float = field(default=0.0)

    noise: bool = field(default=False)
    noise_max: float = field(default=0.1)
    noise_min: float = field(default=-0.1)

    periodic_disturbance: bool = field(default=False)
    disturbance_amplitude: float = field(default=0.1)
    disturbance_period: float = field(default=1.0)

    def default_speed_params(self):
        self.Kp = 1
        self.Ki = 0
        self.Kd = 0
        self.u_min = -2
        self.u_max = 2
        self.du_min = -10 * self.dt
        self.du_max =  10 * self.dt
        self.noise = False
        return

    def default_steer_params(self):
        self.Kp = 1
        self.Ki = 0.0005 / self.dt
        self.Kd = 0
        self.u_min = -0.35
        self.u_max = 0.35
        self.du_min = -4 * self.dt
        self.du_max = 4 * self.dt
        self.noise = False
        return

@dataclass
class JoystickParams(ControllerConfig):
    dt: float                           = field(default = 0.1)

    u_steer_max: float                  = field(default = 0.436)
    u_steer_min: float                  = field(default = -0.436)
    u_steer_neutral: float              = field(default = 0.0)
    u_steer_rate_max: float             = field(default=None)
    u_steer_rate_min: float             = field(default=None)

    u_a_max: float                      = field(default = 2.0)
    u_a_min: float                      = field(default = -2.0)
    u_a_neutral: float                  = field(default = 0.0)
    u_a_rate_max: float                 = field(default=None)
    u_a_rate_min: float                 = field(default=None)
    
    throttle_pid: bool                  = field(default=False)
    steering_pid: bool                  = field(default=False)

    throttle_pid_params: PIDParams      = field(default=None)
    steering_pid_params: PIDParams      = field(default=None)

@dataclass
class CALMPCParams(ControllerConfig):
    N: int                                  = field(default=10)

    n_ss_pts: int                           = field(default=48)
    n_ss_its: int                           = field(default=4)

    convex_hull_slack_quad: list            = field(default=None)
    convex_hull_slack_lin: list             = field(default=None)

    state_scaling: list                     = field(default=None)
    input_scaling: list                     = field(default=None)

    terminal_cost_scaling: float            = field(default=1.0)

    soft_state_bound_idxs: list             = field(default=None)
    soft_state_bound_quad: list             = field(default=None)
    soft_state_bound_lin: list              = field(default=None)

    regression_regularization: float        = field(default=1e-3)
    regression_state_in_idxs: list          = field(default=None)
    regression_input_in_idxs: list          = field(default=None)
    regression_state_out_idxs: list         = field(default=None)

    nearest_neighbor_weights: list          = field(default=None)
    nearest_neighbor_bw: float              = field(default=5.0)
    nearest_neighbor_max_points: int        = field(default=25)

    wrapped_state_idxs: list                = field(default=None)
    wrapped_state_periods: list             = field(default=None)

    damping: float                          = field(default=0.0)
    qp_iters: int                           = field(default=1)
    
    delay: list                             = field(default=None)

    debug_plot: bool                        = field(default=False)
    verbose: bool                           = field(default=False)

    safe_set_init_data_file: str            = field(default='')
    safe_set_topic: str                     = field(default='')
    keep_init_safe_set: bool                = field(default=False)

@dataclass
class LTVLMPCParams(ControllerConfig):
    n: int = field(default=6) # dimension state space
    d: int = field(default=2) # dimension input space
    N: int = field(default=10) # horizon length

    n_ss_pts: int       = field(default=40)
    n_ss_its: int       = field(default=4)
    id_delay: int       = field(default=0)

    Q: np.array         = field(default=None) # quadratic state cost
    R: np.array         = field(default=None) # quadratic input cost
    Q_f: np.array       = field(default=None) # quadratic state cost final
    R_d: np.array       = field(default=None) # Quadratic rate cost
    Q_slack: np.array   = field(default=None) # Quadratic slack cost Q_s*eye
    Q_lane: np.array    = field(default=None) # Linear slack cost l_s*ones

    regression_regularization: float = field(default=0.0)

    time_varying: bool              = field(default=True)
    safe_set_init_data_file: str    = field(default = '')
    safe_set_topic: str             = field(default = '/experiment/barc_1/closed_loop_traj')

    debug_save: bool = field(default=False)

@dataclass
class NLMPCParams(ControllerConfig):
    optlevel: int = field(default=1)
    slack: bool = field(default=True)
    solver_dir: str = field(default='')

    n: int = field(default=6) # dimension state space
    d: int = field(default=2) # dimension input space

    N: int = field(default=10) # horizon length

    Q: np.array = field(default=None) # quadratic state cost
    R: np.array = field(default=None) # quadratic input cost
    Q_f: np.array = field(default=None) # quadratic state cost final
    R_d: np.array = field(default=None) # Quadratic rate cost
    Q_s: float = field(default=1.0) # Quadratic slack cost Q_s*eye
    l_s: float = field(default=1.0) # Linear slack cost l_s*ones

    # named constraints
    v_long_max: float       = field(default = np.inf)
    v_long_min: float       = field(default = -np.inf)
    v_tran_max: float       = field(default = np.inf)
    v_tran_min: float       = field(default = -np.inf)
    psidot_max: float       = field(default = np.inf)
    psidot_min: float       = field(default = -np.inf)
    e_psi_max: float        = field(default = np.inf)
    e_psi_min: float        = field(default = -np.inf)
    s_max: float            = field(default = np.inf)
    s_min: float            = field(default = -np.inf)
    x_tran_max: float       = field(default = np.inf)
    x_tran_min: float       = field(default = -np.inf)
    u_steer_max: float      = field(default = np.inf)
    u_steer_min: float      = field(default = -np.inf)
    u_a_max: float          = field(default = np.inf)
    u_a_min: float          = field(default = -np.inf)
    u_steer_rate_max: float = field(default = np.inf)
    u_steer_rate_min: float = field(default = -np.inf)
    u_a_rate_max: float     = field(default = np.inf)
    u_a_rate_min: float     = field(default = -np.inf)

    # vector constraints
    state_ub: np.array = field(default=None)
    state_lb: np.array = field(default=None)
    input_ub: np.array = field(default=None)
    input_lb: np.array = field(default=None)
    input_rate_ub: np.array = field(default=None)
    input_rate_lb: np.array = field(default=None)


    def __post_init__(self):
        if self.Q is None:
            self.Q = np.ones(self.n)
        if self.R is None:
            self.R = np.ones(self.d)
        if self.Q_f is None:
            self.Q_f = np.zeros(self.n)
        if self.R_d is None:
            self.R_d = np.zeros(self.d)
        if self.state_ub is None:
            self.state_ub = np.inf*np.ones(self.n)
        if self.state_lb is None:
            self.state_lb = -np.inf*np.ones(self.n)
        if self.input_ub is None:
            self.input_ub = np.inf*np.ones(self.d)
        if self.input_lb is None:
            self.input_lb = -np.inf*np.ones(self.d)
        if self.input_rate_ub is None:
            self.input_rate_ub = np.inf*np.ones(self.d)
        if self.input_rate_lb is None:
            self.input_rate_lb = -np.inf*np.ones(self.d)
        self.vectorize_constraints()

    def vectorize_constraints(self, kinematic=False):
        if kinematic:
            #['v_long', 'e_psi', 's', 'x_tran']  ['u_a', 'u_steer']
            self.state_ub = np.array([self.v_long_max,
                                      self.e_psi_max,
                                      self.s_max,
                                      self.x_tran_max])
            self.state_lb = np.array([self.v_long_min,
                                      self.e_psi_min,
                                      self.s_min,
                                      self.x_tran_min])
        else:
            #['v_long', 'v_tran', 'psidot', 'e_psi', 's', 'x_tran']  ['u_a', 'u_steer']
            self.state_ub = np.array([self.v_long_max,
                                      self.v_tran_max,
                                      self.psidot_max,
                                      self.e_psi_max,
                                      self.s_max,
                                      self.x_tran_max])
            self.state_lb = np.array([self.v_long_min,
                                      self.v_tran_min,
                                      self.psidot_min,
                                      self.e_psi_min,
                                      self.s_min,
                                      self.x_tran_min])

        self.input_ub = np.array([self.u_a_max, self.u_steer_max])
        self.input_lb = np.array([self.u_a_min, self.u_steer_min])
        self.input_rate_ub = np.array([self.u_a_rate_max, self.u_steer_rate_max])
        self.input_rate_lb = np.array([self.u_a_rate_min, self.u_steer_rate_min])

        return

@dataclass
class CANLMPCParams(ControllerConfig):
    N: int                              = field(default=10) # horizon length

    # Code gen options
    verbose: bool                       = field(default=False)
    code_gen: bool                      = field(default=False)
    jit: bool                           = field(default=False)
    opt_flag: str                       = field(default='O0')
    enable_jacobians: bool              = field(default=True)
    solver_name: str                    = field(default='CA_NL_MPC')
    solver_dir: str                     = field(default=None)

    soft_state_bound_idxs: list         = field(default=None)
    soft_state_bound_quad: list         = field(default=None)
    soft_state_bound_lin: list          = field(default=None)

    wrapped_state_idxs: list            = field(default=None)
    wrapped_state_periods: list         = field(default=None)

    delay: list                         = field(default=None)

@dataclass
class CALTVMPCParams(ControllerConfig):
    N: int                              = field(default=10) # horizon length
    
    # Code gen options
    verbose: bool                       = field(default=False)
    code_gen: bool                      = field(default=False)
    jit: bool                           = field(default=False)
    opt_flag: str                       = field(default='O0')
    solver_name: str                    = field(default='LTV_MPC')
    debug_plot: bool                    = field(default=False)

    soft_state_bound_idxs: list         = field(default=None)
    soft_state_bound_quad: list         = field(default=None)
    soft_state_bound_lin: list          = field(default=None)

    soft_constraint_idxs: list         = field(default=None)
    soft_constraint_quad: list         = field(default=None)
    soft_constraint_lin: list          = field(default=None)

    wrapped_state_idxs: list            = field(default=None)
    wrapped_state_periods: list         = field(default=None)

    state_scaling: list                 = field(default=None)
    input_scaling: list                 = field(default=None)
    damping: float                      = field(default=0.75)
    qp_iters: int                       = field(default=2)
    qp_interface: str                   = field(default='casadi')

    delay: list                         = field(default=None)

@dataclass
class CADLTVMPCParams(CALTVMPCParams):
    regression_its: int                     = field(default=4)
    regression_regularization: float        = field(default=1e-3)
    regression_state_in_idxs: list          = field(default=None)
    regression_input_in_idxs: list          = field(default=None)
    regression_state_out_idxs: list         = field(default=None)

    nearest_neighbor_weights: list          = field(default=None)
    nearest_neighbor_bw: float              = field(default=5.0)
    nearest_neighbor_max_points: int        = field(default=25)

    regression_init_data_file: str          = field(default='')
    regression_topic: str                   = field(default='')
    keep_init_data: bool                    = field(default=False)

@dataclass
class CAMPCCParams(ControllerConfig):
    N: int                              = field(default=10) # horizon length

    qp_interface: str                   = field(default='casadi')
    
    # Code gen options
    verbose: bool                       = field(default=False)
    code_gen: bool                      = field(default=False)
    jit: bool                           = field(default=False)
    opt_flag: str                       = field(default='O0')
    enable_jacobians: bool              = field(default=True)
    solver_name: str                    = field(default='CA_MPCC')
    solver_dir: str                     = field(default=None)
    debug_plot: bool                    = field(default=False)

    conv_approx: bool                   = field(default=False)
    soft_track: bool                    = field(default=False)
    track_tightening: float             = field(default=0)

    soft_constraint_idxs: list          = field(default=None)
    soft_constraint_quad: list          = field(default=None)
    soft_constraint_lin: list           = field(default=None)

    pos_idx: list                       = field(default_factory=lambda : [3, 4])
    state_scaling: list                 = field(default=None)
    input_scaling: list                 = field(default=None)
    damping: float                      = field(default=0.75)
    qp_iters: int                       = field(default=2)

    parametric_contouring_cost: bool    = field(default=False)
    contouring_cost: float              = field(default=0.1)
    contouring_cost_N: float            = field(default=1.0)
    lag_cost: float                     = field(default=1000.0)
    lag_cost_N: float                   = field(default=1000.0)
    performance_cost: float             = field(default=0.02)
    vs_cost: float                      = field(default=1e-4)
    vs_rate_cost: float                 = field(default=1e-3)
    track_slack_quad: float             = field(default=100.0)
    track_slack_lin: float              = field(default=0.0)

    vs_max: float                       = field(default=5.0)
    vs_min: float                       = field(default=0.0)
    vs_rate_max: float                  = field(default=5.0)
    vs_rate_min: float                  = field(default=-5.0)

    delay: list                         = field(default=None)
    
@dataclass
class NLLMPCParams(ControllerConfig):
    n: int              = field(default=6) # dimension state space
    d: int              = field(default=2) # dimension input space

    N: int              = field(default=10) # horizon length

    Q: np.array         = field(default=None) # quadratic state cost
    R: np.array         = field(default=None) # quadratic input cost
    Q_f: np.array       = field(default=None) # quadratic state cost final
    R_d: np.array       = field(default=None) # Quadratic rate cost
    Q_s: float          = field(default=1.0) # Quadratic slack cost
    l_s: float          = field(default=1.0) # Linear slack cost
    Q_ch: np.array      = field(default=None) # Quadratic cost on convex hull slack

    # named constraints
    v_long_max: float       = field(default = np.inf)
    v_long_min: float       = field(default = -np.inf)
    v_tran_max: float       = field(default = np.inf)
    v_tran_min: float       = field(default = -np.inf)
    psidot_max: float       = field(default = np.inf)
    psidot_min: float       = field(default = -np.inf)
    e_psi_max: float        = field(default = np.inf)
    e_psi_min: float        = field(default = -np.inf)
    s_max: float            = field(default = np.inf)
    s_min: float            = field(default = -np.inf)
    x_tran_max: float       = field(default = 1.0)
    x_tran_min: float       = field(default = -1.0)
    u_steer_max: float      = field(default = 0.5)
    u_steer_min: float      = field(default = -0.5)
    u_a_max: float          = field(default = 2.0)
    u_a_min: float          = field(default = -2.0)
    u_steer_rate_max: float = field(default = 0.5)
    u_steer_rate_min: float = field(default = -0.5)
    u_a_rate_max: float     = field(default = 2.0)
    u_a_rate_min: float     = field(default = -2.0)

    # vector constraints
    state_ub: np.array = field(default=None)
    state_lb: np.array = field(default=None)
    input_ub: np.array = field(default=None)
    input_lb: np.array = field(default=None)
    input_rate_ub: np.array = field(default=None)
    input_rate_lb: np.array = field(default=None)

    optlevel: int = field(default=1)
    slack: bool = field(default=True)
    solver_dir: str = field(default='')

    n_ss_pts: int = field(default=10) # Number of safe set points per lap
    n_ss_its: int = field(default=3) # Number of previous laps
    ss_selection_weights: np.array = field(default=None)

    safe_set_init_data_file: str = field(default = '')
    safe_set_topic: str          = field(default = 'closed_loop_traj')

    def __post_init__(self):
        if self.Q is None:
            self.Q = np.ones(self.n)
        if self.R is None:
            self.R = np.ones(self.d)
        if self.Q_f is None:
            self.Q_f = np.zeros(self.n)
        if self.R_d is None:
            self.R_d = np.zeros(self.d)
        if self.Q_ch is None:
            self.Q_ch = np.ones(self.n)
        if self.state_ub is None:
            self.state_ub = np.inf*np.ones(self.n)
        if self.state_lb is None:
            self.state_lb = -np.inf*np.ones(self.n)
        if self.input_ub is None:
            self.input_ub = np.inf*np.ones(self.d)
        if self.input_lb is None:
            self.input_lb = -np.inf*np.ones(self.d)
        if self.input_rate_ub is None:
            self.input_rate_ub = np.inf*np.ones(self.d)
        if self.input_rate_lb is None:
            self.input_rate_lb = -np.inf*np.ones(self.d)
        if self.ss_selection_weights is None:
            self.ss_selection_weights = np.ones(self.n)

    def vectorize_constraints(self, kinematic=False):
        if kinematic:
            #['v_long', 'e_psi', 's', 'x_tran']  ['u_a', 'u_steer']
            self.state_ub = np.array([self.v_long_max,
                                      self.e_psi_max,
                                      self.s_max,
                                      self.x_tran_max])
            self.state_lb = np.array([self.v_long_min,
                                      self.e_psi_min,
                                      self.s_min,
                                      self.x_tran_min])
        else:
            #['v_long', 'v_tran', 'psidot', 'e_psi', 's', 'x_tran']  ['u_a', 'u_steer']
            self.state_ub = np.array([self.v_long_max,
                                      self.v_tran_max,
                                      self.psidot_max,
                                      self.e_psi_max,
                                      self.s_max,
                                      self.x_tran_max])
            self.state_lb = np.array([self.v_long_min,
                                      self.v_tran_min,
                                      self.psidot_min,
                                      self.e_psi_min,
                                      self.s_min,
                                      self.x_tran_min])

        self.input_ub = np.array([self.u_a_max, self.u_steer_max])
        self.input_lb = np.array([self.u_a_min, self.u_steer_min])
        self.input_rate_ub = np.array([self.u_a_rate_max, self.u_steer_rate_max])
        self.input_rate_lb = np.array([self.u_a_rate_min, self.u_steer_rate_min])
        return

'''
Similar solver parameters specified from
Optimization-Based Autonomous Racing of 1:43 Scale RC Cars
Alexander Liniger, Alexander Domahidi and Manfred Morari
'''
@dataclass
class MPCCParams(ControllerConfig):
    n: int          = field(default=6) # dimension state space
    d: int          = field(default=3) # dimension input space

    N: int          = field(default=10)
    Qc: float       = field(default=200.0)
    Ql: float       = field(default=1000.0)
    Q_theta: float  = field(default=100.0)
    R_d: float      = field(default=0.01)
    R_delta: float  = field(default=0.01)

    # Track parameters, TODO: move to separate track params
    density: float  = field(default=100)

    # Slack for collisions
    slack: bool     = field(default=False)
    Q_s: float      = field(default=1.0) # Quadratic slack cost Q_s*eye
    l_s: float      = field(default=1.0) # Linear slack cost l_s*ones

    # named constraints
    posx_max: float = field(default=np.inf)
    posx_min: float = field(default=-np.inf)
    posy_max: float = field(default=np.inf)
    posy_min: float = field(default=-np.inf)
    psi_max: float  = field(default=np.inf)
    psi_min: float  = field(default=-np.inf)
    vx_max: float   = field(default=np.inf)
    vx_min: float   = field(default=-np.inf)
    vy_max: float   = field(default=np.inf)
    vy_min: float   = field(default=-np.inf)
    psidot_max: float = field(default=np.inf)
    psidot_min: float = field(default=-np.inf)

    u_a_max: float          = field(default = 2.0)
    u_a_min: float          = field(default = -2.0)
    u_theta_max: float      = field(default=100.0)
    u_theta_min: float      = field(default=-100.0)
    u_steer_max: float      = field(default = 0.5)
    u_steer_min: float      = field(default = -0.5)
    u_a_rate_max: float     = field(default = 2.0)
    u_a_rate_min: float     = field(default = -2.0)
    u_theta_rate_max: float = field(default = 5.0)
    u_theta_rate_min: float = field(default=-5.0)
    u_steer_rate_max: float = field(default = 0.5)
    u_steer_rate_min: float = field(default = -0.5)

    # vector constraints
    state_ub: np.array = field(default=None)
    state_lb: np.array = field(default=None)
    input_ub: np.array = field(default=None)
    input_lb: np.array = field(default=None)
    input_rate_ub: np.array = field(default=None)
    input_rate_lb: np.array = field(default=None)

    optlevel: int = field(default=1)
    solver_dir: str = field(default='')

    def __post_init__(self):
        if self.state_ub is None:
            self.state_ub = np.inf*np.ones(self.n)
        if self.state_lb is None:
            self.state_lb = -np.inf*np.ones(self.n)
        if self.input_ub is None:
            self.input_ub = np.inf*np.ones(self.d)
        if self.input_lb is None:
            self.input_lb = -np.inf*np.ones(self.d)
        if self.input_rate_ub is None:
            self.input_rate_ub = np.inf*np.ones(self.d)
        if self.input_rate_lb is None:
            self.input_rate_lb = -np.inf*np.ones(self.d)

    def vectorize_constraints(self):
        self.state_ub = np.array([self.vx_max,
                                  self.vy_max,
                                  self.psidot_max,
                                  self.posx_max,
                                  self.posy_max,
                                  self.psi_max])
        self.state_lb = np.array([self.vx_min,
                                  self.vy_min,
                                  self.psidot_min,
                                  self.posx_min,
                                  self.posy_min,
                                  self.psi_min])

        self.input_ub = np.array([self.u_a_max, self.u_steer_max, self.u_theta_max])
        self.input_lb = np.array([self.u_a_min, self.u_steer_min, self.u_theta_min])
        self.input_rate_ub = np.array([self.u_a_rate_max, self.u_steer_rate_max, self.u_theta_rate_max])
        self.input_rate_lb = np.array([self.u_a_rate_min, self.u_steer_rate_min, self.u_theta_rate_min])
        return

@dataclass
class MPCCApproxFullModelParams(ControllerConfig):
    name: str = field(default=None)
    all_tracks: bool = field(default=True)
    n: int = field(default=11) # dimension state space, 1 extra dim for theta
    d: int = field(default=3) # dimension input space, 1 extra dim for v_proj

    N: int = field(default=10)
    Qc: float = field(default=20.0)
    Ql: float = field(default=80.0)
    Q_theta: float = field(default=100.0)
    Q_xref: float = field(default=25.0)
    R_d: float = field(default=0.01)
    R_delta: float = field(default=0.01)

    slack: bool = field(default=False)

    # Slack for collisions
    Q_cs: float = field(default=1.0)  # Quadratic slack cost Q_s*eye
    l_cs: float = field(default=1.0)  # Linear slack cost l_s*ones
    Q_cs_e: float = field(default=1.0) # Quadratic slack cost for safety bound
    l_cs_e: float = field(default=1.0) # Linear slack cost for safety bound
    Q_ts: float = field(default=1.0)  # Quadratic cost for soft track constraint violation
    Q_vmax: float = field(default=1.0)  # Quadratic cost for v_long max soft constraint
    vlong_max_soft: float = field(default=1.0) # Max v_long (only for soft constraint)

    num_std_deviations: float = field(default=1.0)  # Number of std deviations from mean to consider for obstacle slack

    # named constraints
    posx_max: float = field(default=np.inf)
    posx_min: float = field(default=-np.inf)
    posy_max: float = field(default=np.inf)
    posy_min: float = field(default=-np.inf)
    psi_max: float = field(default=np.inf)
    psi_min: float = field(default=-np.inf)
    vx_max: float = field(default=np.inf)
    vx_min: float = field(default=-np.inf)
    vy_max: float = field(default=np.inf)
    vy_min: float = field(default=-np.inf)
    psidot_max: float = field(default=np.inf)
    psidot_min: float = field(default=-np.inf)
    e_psi_max: float = field(default=np.inf)
    e_psi_min: float = field(default=-np.inf)
    s_max: float = field(default=np.inf)
    s_min: float = field(default=-np.inf)
    x_tran_max: float = field(default=np.inf)
    x_tran_min: float = field(default=-np.inf)
    theta_max: float = field(default=np.inf)
    theta_min: float = field(default=0)

    u_a_max: float          = field(default = 2.0)
    u_a_min: float          = field(default = -2.0)
    u_steer_max: float      = field(default = 0.5)
    u_steer_min: float      = field(default = -0.5)
    u_a_rate_max: float     = field(default = 2.0)
    u_a_rate_min: float     = field(default = -2.0)
    u_theta_rate_max: float = field(default = 5.0)
    u_theta_rate_min: float = field(default=-5.0)
    u_steer_rate_max: float     = field(default = 0.5)
    u_steer_rate_min: float     = field(default = -0.5)
    v_proj_max: float = field(default=None)
    v_proj_min: float = field(default=None)
    v_proj_rate_max: float = field(default=np.inf)
    v_proj_rate_min: float = field(default=-np.inf)

    # vector constraints
    state_ub: np.array = field(default=None)
    state_lb: np.array = field(default=None)
    input_ub: np.array = field(default=None)
    input_lb: np.array = field(default=None)
    input_rate_ub: np.array = field(default=None)
    input_rate_lb: np.array = field(default=None)

    optlevel: int = field(default=1)
    solver_dir: str = field(default='')

    def __post_init__(self):
        # TODO Temporary fix
        if self.v_proj_max is None or True:
            self.v_proj_max = self.vx_max*3
        if self.v_proj_min is None or True:
            self.v_proj_min = 0
        if self.state_ub is None:
            self.state_ub = np.inf*np.ones(self.n)
        if self.state_lb is None:
            self.state_lb = -np.inf*np.ones(self.n)
        if self.input_ub is None:
            self.input_ub = np.inf*np.ones(self.d)
        if self.input_lb is None:
            self.input_lb = -np.inf*np.ones(self.d)
        if self.input_rate_ub is None:
            self.input_rate_ub = np.inf*np.ones(self.d)
        if self.input_rate_lb is None:
            self.input_rate_lb = -np.inf*np.ones(self.d)
        self.vectorize_constraints()

    def vectorize_constraints(self):
        self.state_ub = np.array([self.vx_max,
                                  self.vy_max,
                                  self.psidot_max,
                                  self.posx_max,
                                  self.posy_max,
                                  self.psi_max,
                                  self.e_psi_max,
                                  self.s_max,
                                  self.x_tran_max,
                                  self.theta_max])
        self.state_lb = np.array([self.vx_min,
                                  self.vy_min,
                                  self.psidot_min,
                                  self.posx_min,
                                  self.posy_min,
                                  self.psi_min,
                                  self.e_psi_min,
                                  self.s_min,
                                  self.x_tran_min,
                                  self.theta_min])

        self.input_ub = np.array([self.u_a_max, self.u_steer_max, self.v_proj_max])
        self.input_lb = np.array([self.u_a_min, self.u_steer_min, self.v_proj_min])
        self.input_rate_ub = np.array([self.u_a_rate_max, self.u_steer_rate_max, self.v_proj_rate_max])
        self.input_rate_lb = np.array([self.u_a_rate_min, self.u_steer_rate_min, self.v_proj_rate_min])
        return


@dataclass
class MPCCFullModelParams(ControllerConfig):
    all_tracks: bool = field(default=True)
    n: int = field(default=9) # dimension state space
    d: int = field(default=2) # dimension input space

    N: int = field(default=10)
    Qc: float = field(default=200.0)
    Ql: float = field(default=1000.0)
    Q_theta: float = field(default=100.0)
    Q_xref: float = field(default=25.0)
    R_d: float = field(default=0.01)
    R_delta: float = field(default=0.01)

    # Track parameters, TODO: move to separate track params
    density: float = field(default=100)

    slack: bool = field(default=False)

    # Slack for collisions
    Q_cs: float = field(default=1.0)  # Quadratic slack cost Q_s*eye
    l_cs: float = field(default=1.0)  # Linear slack cost l_s*ones
    Q_cs_e: float = field(default=1.0) # Quadratic slack cost for safety bound
    l_cs_e: float = field(default=1.0) # Linear slack cost for safety bound
    # Slack for track constraints
    Q_ts: float = field(default=1.0)  # Quadratic cost for track constraint violation
    Q_vmax: float = field(default=1.0)  # Quadratic cost for v_long max

    num_std_deviations: float = field(default=1.0)  # Number of std deviations from mean to consider for obstacle slack

    # named constraints
    posx_max: float = field(default=np.inf)
    posx_min: float = field(default=-np.inf)
    posy_max: float = field(default=np.inf)
    posy_min: float = field(default=-np.inf)
    psi_max: float = field(default=np.inf)
    psi_min: float = field(default=-np.inf)
    vx_max: float = field(default=np.inf)
    vx_min: float = field(default=-np.inf)
    vy_max: float = field(default=np.inf)
    vy_min: float = field(default=-np.inf)
    psidot_max: float = field(default=np.inf)
    psidot_min: float = field(default=-np.inf)
    e_psi_max: float = field(default=np.inf)
    e_psi_min: float = field(default=-np.inf)
    s_max: float = field(default=np.inf)
    s_min: float = field(default=-np.inf)
    x_tran_max: float = field(default=np.inf)
    x_tran_min: float = field(default=-np.inf)

    u_a_max: float          = field(default = 2.0)
    u_a_min: float          = field(default = -2.0)
    u_theta_max: float = field(default=100.0)
    u_theta_min: float = field(default=-100.0)
    u_steer_max: float      = field(default = 0.5)
    u_steer_min: float      = field(default = -0.5)
    u_a_rate_max: float     = field(default = 2.0)
    u_a_rate_min: float     = field(default = -2.0)
    u_theta_rate_max: float = field(default = 5.0)
    u_theta_rate_min: float = field(default=-5.0)
    u_steer_rate_max: float     = field(default = 0.5)
    u_steer_rate_min: float     = field(default = -0.5)

    # vector constraints
    state_ub: np.array = field(default=None)
    state_lb: np.array = field(default=None)
    input_ub: np.array = field(default=None)
    input_lb: np.array = field(default=None)
    input_rate_ub: np.array = field(default=None)
    input_rate_lb: np.array = field(default=None)

    optlevel: int = field(default=1)
    solver_dir: str = field(default='')

    def __post_init__(self):
        if self.state_ub is None:
            self.state_ub = np.inf*np.ones(self.n)
        if self.state_lb is None:
            self.state_lb = -np.inf*np.ones(self.n)
        if self.input_ub is None:
            self.input_ub = np.inf*np.ones(self.d)
        if self.input_lb is None:
            self.input_lb = -np.inf*np.ones(self.d)
        if self.input_rate_ub is None:
            self.input_rate_ub = np.inf*np.ones(self.d)
        if self.input_rate_lb is None:
            self.input_rate_lb = -np.inf*np.ones(self.d)
        self.vectorize_constraints()

    def vectorize_constraints(self):
        self.state_ub = np.array([self.vx_max,
                                  self.vy_max,
                                  self.psidot_max,
                                  self.posx_max,
                                  self.posy_max,
                                  self.psi_max,
                                  self.e_psi_max,
                                  self.s_max,
                                  self.x_tran_max])
        self.state_lb = np.array([self.vx_min,
                                  self.vy_min,
                                  self.psidot_min,
                                  self.posx_min,
                                  self.posy_min,
                                  self.psi_min,
                                  self.e_psi_min,
                                  self.s_min,
                                  self.x_tran_min])

        self.input_ub = np.array([self.u_a_max, self.u_steer_max])
        self.input_lb = np.array([self.u_a_min, self.u_steer_min])
        self.input_rate_ub = np.array([self.u_a_rate_max, self.u_steer_rate_max])
        self.input_rate_lb = np.array([self.u_a_rate_min, self.u_steer_rate_min])
        return


@dataclass
class iLQRParams(ControllerConfig):
    N: int                  = field(default=10) # Horizon length
    max_iters: int          = field(default=100)
    ddp: bool               = field(default=False) # Do differential dynamic programming backwards pass

    tol: float              = field(default=1e-3)
    rel_tol: float          = field(default=1e-3)
    control_reg: float      = field(default=1e-3)
    state_reg: float        = field(default=1e-3)
    line_search: bool       = field(default=False)
    alpha_init: float       = field(default=0.05)
    gamma: float            = field(default=0.9)

    verbose: bool           = field(default=False)
    code_gen: bool          = field(default=False)
    jit: bool               = field(default=False)
    opt_flag: str           = field(default='O0')
    enable_jacobians: bool  = field(default=True)
    solver_name: str        = field(default='iLQR')
    solver_dir: str         = field(default=None)

    debug_plot: bool        = field(default=False)
    pause_on_plot: bool     = field(default=False)
    local_pos: bool         = field(default=False)

    ego_state_input_max: VehicleState = field(default=None)
    ego_state_input_min: VehicleState = field(default=None)
    tar_state_input_max: VehicleState = field(default=None)
    tar_state_input_min: VehicleState = field(default=None)

    ego_state_input_rate_max: VehicleState = field(default=None)
    ego_state_input_rate_min: VehicleState = field(default=None)
    tar_state_input_rate_max: VehicleState = field(default=None)
    tar_state_input_rate_min: VehicleState = field(default=None)

@dataclass
class CiLQRParams(iLQRParams):
    constr_reg: float       = field(default=1e-3)

@dataclass
class ALGAMESMPCParams(ControllerConfig):
    N: int                  = field(default=10) # Horizon length

    rho: float              = field(default=1.0) # Lagrangian regularization
    gamma: float            = field(default=10.0) # rho update schedule
    rho_max: float          = field(default=1e7)
    lam_max: float          = field(default=1e7)

    beta: float             = field(default=0.25) # Line search param
    tau: float              = field(default=0.5) # Line search param

    q_reg: float            = field(default=1e-2) # Jacobian regularization
    u_reg: float            = field(default=1e-2) # Jacobian regularization
    line_search_tol: float  = field(default=1e-6)
    newton_step_tol: float  = field(default=1e-6) # Newton step size
    ineq_tol: float         = field(default=1e-3) # Inequality constraint violation
    eq_tol: float           = field(default=1e-3) # Equality constraint violation
    opt_tol: float          = field(default=1e-3) # Optimality violation

    dynamics_hessians: bool = field(default=False)

    outer_iters: int        = field(default=50)
    line_search_iters: int  = field(default=50)
    newton_iters: int       = field(default=50)

    verbose: bool           = field(default=False)
    code_gen: bool          = field(default=False)
    jit: bool               = field(default=False)
    opt_flag: str           = field(default='O0')
    solver_name: str        = field(default='ALGAMESMPC')
    solver_dir: str         = field(default=None)

    debug: bool             = field(default=False)
    debug_plot: bool        = field(default=False)
    pause_on_plot: bool     = field(default=False)
    local_pos: bool         = field(default=False)

@dataclass
class DGSQPParams(ControllerConfig):
    N: int                  = field(default=10) # Horizon length

    beta: float             = field(default=0.25) # Line search param
    tau: float              = field(default=0.5) # Line search param

    p_tol: float            = field(default=1e-3)
    d_tol: float            = field(default=1e-3)

    reg: float              = field(default=1e-3)
    line_search_iters: int  = field(default=50)
    nonmono_ls: bool        = field(default=False)
    sqp_iters: int          = field(default=50)
    conv_approx: bool       = field(default=True)
    merit_function: str     = field(default='stat_l1')

    verbose: bool           = field(default=False)
    save_iter_data: bool    = field(default=False)
    time_limit: float       = field(default=None)

    code_gen: bool          = field(default=False)
    jit: bool               = field(default=False)
    opt_flag: str           = field(default='O0')
    enable_jacobians: bool  = field(default=True)
    solver_name: str        = field(default='DGSQP')
    solver_dir: str         = field(default=None)
    so_name: str            = field(default=None)
    qp_interface: str       = field(default='casadi')
    qp_solver: str          = field(default='osqp')
    hessian_approximation: str = field(default='none')

    debug: bool             = field(default=False)
    debug_plot: bool        = field(default=False)
    pause_on_plot: bool     = field(default=False)
    local_pos: bool         = field(default=False)

@dataclass
class DGSQPRaceParams(ControllerConfig):
    N: int                  = field(default=10) # Horizon length

    beta: float             = field(default=0.25) # Line search param
    tau: float              = field(default=0.5) # Line search param

    p_tol: float            = field(default=1e-3)
    d_tol: float            = field(default=1e-3)

    reg: float              = field(default=1e-3)
    line_search_iters: int  = field(default=50)
    nonmono_ls: bool        = field(default=False)
    sqp_iters: int          = field(default=50)
    conv_approx: bool       = field(default=True)
    merit_function: str     = field(default='stat_l1')

    verbose: bool           = field(default=False)
    save_iter_data: bool    = field(default=False)
    time_limit: float       = field(default=None)

    code_gen: bool          = field(default=False)
    jit: bool               = field(default=False)
    opt_flag: str           = field(default='O0')
    enable_jacobians: bool  = field(default=True)
    solver_name: str        = field(default='DGSQP')
    solver_dir: str         = field(default=None)
    so_name: str            = field(default=None)
    qp_interface: str       = field(default='casadi')
    qp_solver: str          = field(default='osqp')
    hessian_approximation: str = field(default='none')

    debug: bool             = field(default=False)
    debug_plot: bool        = field(default=False)
    pause_on_plot: bool     = field(default=False)
    local_pos: bool         = field(default=False)

@dataclass
class DGIPMParams(ControllerConfig):
    N: int                  = field(default=10) # Horizon length

    mu_init: float          = field(default=0.1)
    beta: float             = field(default=0.25) # Line search param
    tau: float              = field(default=0.5) # Line search param
    tau_min: float          = field(default=0.99)
    k_eps: float            = field(default=10)
    k_mu: float             = field(default=0.2)
    theta_mu: float         = field(default=1.5)

    max_vio: float          = field(default=1e4)

    p_tol: float            = field(default=1e-3)
    d_tol: float            = field(default=1e-3)

    reg: float              = field(default=1e-3)
    line_search_iters: int  = field(default=50)
    nonmono_ls: bool        = field(default=False)
    max_iters: int          = field(default=50)
    conv_approx: bool       = field(default=True)
    merit_function: str     = field(default='stat_l1')
    max_scaling: float      = field(default=100.0)

    verbose: bool           = field(default=False)
    save_iter_data: bool    = field(default=True)

    code_gen: bool          = field(default=False)
    jit: bool               = field(default=False)
    opt_flag: str           = field(default='O0')
    enable_jacobians: bool  = field(default=True)
    solver_name: str        = field(default='DGIPM')
    solver_dir: str         = field(default=None)
    so_name: str            = field(default=None)

    debug: bool             = field(default=False)
    debug_plot: bool        = field(default=False)
    pause_on_plot: bool     = field(default=False)
    local_pos: bool         = field(default=False)

@dataclass
class PSIBRMPCParams(ControllerConfig):
    N: int                  = field(default=10) # Horizon length

    use_ps: bool            = field(default=True)
    p_tol: float            = field(default=1e-3)
    d_tol: float            = field(default=1e-3)

    line_search_iters: int  = field(default=50)
    ibr_iters: int          = field(default=50)

    verbose: bool           = field(default=False)
    code_gen: bool          = field(default=False)
    jit: bool               = field(default=False)
    opt_flag: str           = field(default='O0')
    enable_jacobians: bool  = field(default=True)
    solver_name: str        = field(default='PSIBRMPC')
    solver_dir: str         = field(default=None)

    debug: bool             = field(default=False)
    debug_plot: bool        = field(default=False)
    pause_on_plot: bool     = field(default=False)
    local_pos: bool         = field(default=False)

@dataclass
class PATHMCPParams(ControllerConfig):
    N: int                  = field(default=10) # Horizon length

    beta: float             = field(default=0.25) # Line search param
    tau: float              = field(default=0.5) # Line search param

    p_tol: float            = field(default=1e-3)
    d_tol: float            = field(default=1e-3)

    reg: float              = field(default=1e-3)
    
    nonmono_ls: bool        = field(default=False)
    outer_iters: int        = field(default=50)
    inner_iters: int        = field(default=50)
    merit_function: str     = field(default='stat_l1')
    
    solver_name: str        = field(default='PATHMCP')
    verbose: bool           = field(default=False)
    save_iter_data: bool    = field(default=False)
    time_limit: float       = field(default=None)

    debug: bool             = field(default=False)
    debug_plot: bool        = field(default=False)
    pause_on_plot: bool     = field(default=False)
    
if __name__ == "__main__":
    pass

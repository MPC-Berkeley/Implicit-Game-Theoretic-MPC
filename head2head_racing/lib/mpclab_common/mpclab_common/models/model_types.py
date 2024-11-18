#!/usr/bin python3

from dataclasses import dataclass, field
import numpy as np

from mpclab_common.pytypes import PythonMsg
from barc3d.pytypes import VehicleConfig

@dataclass
class ModelConfig(PythonMsg):
    model_name: str                 = field(default = 'model')
    use_mx: bool                    = field(default = False)
    enable_jacobians: bool          = field(default = True)
    compute_hessians: bool          = field(default = False)
    verbose: bool                   = field(default = False)
    code_gen: bool                  = field(default = False)
    jit: bool                       = field(default = True)
    opt_flag: str                   = field(default = 'O0')
    install: bool                   = field(default = True)
    install_dir: str                = field(default = '~/.mpclab_common/models')

@dataclass
class DynamicsConfig(ModelConfig):
    track_name: str                 = field(default = None)

    dt: float                       = field(default = 0.01)   # interval of an entire simulation step
    discretization_method: str      = field(default = 'euler')
    M: int                          = field(default = 10) # RK4 integration steps

    # Flag indicating whether dynamics are affected by exogenous noise
    noise: bool                     = field(default = False)
    noise_cov: np.ndarray           = field(default = None)

@dataclass
class ObserverConfig(ModelConfig):
    track_name: str                 = field(default = None)

    # Flag indicating whether observations are affected by exogenous noise
    noise: bool                     = field(default = False)
    noise_cov: np.ndarray           = field(default = None)

@dataclass
class BeliefConfig(ModelConfig):
    use_mx: bool                    = field(default = False)

    inter_agent_covariance: bool    = field(default = True)

@dataclass
class DynamicBicycleConfig(DynamicsConfig):  # configurations for simulated vehicle model, can grow to be used elsewhere.
    wheel_dist_front: float         = field(default = 0.13)
    wheel_dist_rear: float          = field(default = 0.13)
    wheel_dist_center_front: float  = field(default = 0.1)
    wheel_dist_center_rear:  float  = field(default = 0.1)
    bump_dist_front: float          = field(default = 0.15)
    bump_dist_rear: float           = field(default = 0.15)
    bump_dist_center: float         = field(default = 0.1)
    bump_dist_top: float            = field(default = 0.1)
    com_height: float               = field(default = 0.05)

    mass: float                     = field(default = 2.2187)
    gravity: float                  = field(default = 9.81)

    yaw_inertia: float              = field(default = 0.02723)
    pitch_inertia: float            = field(default = 0.03)  # Not being used in dynamics
    roll_inertia: float             = field(default = 0.03)  # Not being used in dynamics
    
    drag_coefficient: float         = field(default = 0.0)  # .05
    damping_coefficient: float      = field(default = 0.0)
    rolling_resistance: float       = field(default = 0.0)
    rolling_resistance_exponent: float = field(default = 0.0)

    tire_model: str                 = field(default = 'pacejka')
    drive_wheels: str               = field(default = 'all')

    wheel_friction: float           = field(default = 0.9)
    pacejka_b_front: float          = field(default = 5.0)
    pacejka_b_rear: float           = field(default = 5.0)
    pacejka_c_front: float          = field(default = 2.28)
    pacejka_c_rear: float           = field(default = 2.28)
    pacejka_d_front: float          = field(default = None)
    pacejka_d_rear: float           = field(default = None)

    linear_bf: float                = field(default = 1.0)
    linear_br: float                = field(default = 1.0)

    simple_slip: bool               = field(default=False)

    def __post_init__(self):
        if self.pacejka_d_front is None:
            self.pacejka_d_front = self.wheel_friction*self.mass*self.gravity * self.wheel_dist_rear / (self.wheel_dist_rear + self.wheel_dist_front)
        if self.pacejka_d_rear is None:
            self.pacejka_d_rear  = self.wheel_friction*self.mass*self.gravity * self.wheel_dist_front / (self.wheel_dist_rear + self.wheel_dist_front)

    def plot_lateral_tire_model(self):
        import matplotlib.pyplot as plt
        alpha = np.linspace(-15, 15, 200)*np.pi/180
        fy = self.pacejka_d_front * np.sin(self.pacejka_c_front * np.arctan(self.pacejka_b_front * alpha))

        plt.plot(alpha, fy)
        plt.show()
        
@dataclass
class KinematicBicycleConfig(DynamicsConfig):  # configurations for simulated vehicle model, can grow to be used elsewhere.
    wheel_dist_front: float         = field(default = 0.13)
    wheel_dist_rear: float          = field(default = 0.13)
    wheel_dist_center_front: float  = field(default = 0.1)
    wheel_dist_center_rear:  float  = field(default = 0.1)
    bump_dist_front: float          = field(default = 0.15)
    bump_dist_rear: float           = field(default = 0.15)
    bump_dist_center: float         = field(default = 0.1)
    bump_dist_top: float            = field(default = 0.1)
    com_height: float               = field(default = 0.05)

    mass: float                     = field(default = 2.366)

    drag_coefficient: float         = field(default = 0.0)
    damping_coefficient: float      = field(default = 0.0)
    slip_coefficient: float         = field(default = 0.0)
    rolling_resistance: float       = field(default = 0.0)
    rolling_resistance_exponent: float = field(default = 0.5)

@dataclass
class PointMassConfig(DynamicsConfig):  # configurations for simulated vehicle model, can grow to be used elsewhere.
    mass: float                         = field(default = 2.366)
    damping_coefficient: float          = field(default = 0.0)  # .05
    drag_coefficient: float             = field(default = 0.0)  # .05
    rolling_resistance: float           = field(default = 0.0)
    rolling_resistance_exponent: float  = field(default = 0.5)

@dataclass
class UnicycleConfig(DynamicsConfig):  # configurations for simulated vehicle model, can grow to be used elsewhere.
    mass: float                         = field(default = 2.366)
    damping_coefficient: float          = field(default = 0.0)  # .05
    drag_coefficient: float             = field(default = 0.0)  # .05
    rolling_resistance: float           = field(default = 0.0)
    rolling_resistance_exponent: float  = field(default = 0.5)

@dataclass
class MultiAgentModelConfig(DynamicsConfig):
    use_mx: bool                    = field(default = False)

@dataclass
class VehicleConfig3D(DynamicsConfig):
    model_config: VehicleConfig     = field(default = None)
    tire_model: str                 = field(default = 'pacejka')
    tire_layout: str                = field(default = 'bicycle')
    input_model: str                = field(default = 'accel')
    weight_distribution: str        = field(default = 'quasiplanar')
    planar: bool                    = field(default = True)

    def __post_init__(self):
        if self.model_config is None:
            self.model_config = VehicleConfig().apply_barc_defaults()

@dataclass
class VehicleConstraints(PythonMsg):
    u_steer_max: float      = field(default = 0.5)
    u_steer_min: float      = field(default = -0.5)
    u_a_max: float          = field(default = 2.0)
    u_a_min: float          = field(default = -2.0)
    u_steer_rate_max: float = field(default = 0.5)
    u_steer_rate_min: float = field(default = -0.5)
    u_a_rate_max: float     = field(default = 2.0)
    u_a_rate_min: float     = field(default = -2.0)

@dataclass
class Measurement(PythonMsg):
    t: float = field(default=None)

@dataclass
class PoseVelMeasurement(Measurement):
    x: float            = field(default=None)
    y: float            = field(default=None)
    z: float            = field(default=None)
    roll: float         = field(default=None)
    pitch: float        = field(default=None)
    yaw: float          = field(default=None)
    v_long: float       = field(default=None)
    v_tran: float       = field(default=None)
    v_vert: float       = field(default=None)
    roll_dot: float     = field(default=None)
    pitch_dot: float    = field(default=None)
    yaw_dot: float      = field(default=None)

    def __str__(self):
        return 't:{self.t}, x:{self.x}, y:{self.y}, z:{self.z}, roll:{self.roll}, pitch:{self.pitch}, yaw:{self.yaw}, v_long:{self.v_long}, v_tran:{self.v_tran}, v_vert:{self.v_vert}, roll_dot:{self.roll_dot}, pitch_dot:{self.pitch_dot}, yaw_dot:{self.yaw_dot}'.format(self=self)

@dataclass
class AccelMeasurement(Measurement):
    x: float = field(default=0)
    y: float = field(default=0)
    z: float = field(default=0)

    def __str__(self):
        if self.t is not None:
            s = 't:{self.t}, '
        else:
            s = ''
        return (s + 'ax:{self.x}, ay:{self.y}, az:{self.z},').format(self=self)

@dataclass
class GyroMeasurement(Measurement):
    x: float = field(default=0)
    y: float = field(default=0)
    z: float = field(default=0)

    def __str__(self):
        if self.t is not None:
            s = 't:{self.t}, '
        else:
            s = ''
        return (s + 'wx:{self.x}, wy:{self.y}, wz:{self.z}').format(self=self)

@dataclass
class MagMeasurement(Measurement):
    x: float = field(default = 1)
    y: float = field(default = 0)
    z: float = field(default = 0)
    w: float = field(default = 0)

    def __str__(self):
        if self.t is not None:
            s = 't:{self.t}, '
        else:
            s = ''
        return (s + 'qx:{self.x}, qy:{self.y}, qz:{self.z}, qw:{self.w}').format(self=self)

@dataclass
class IMUMeasurement(Measurement):
    linear_acceleration: AccelMeasurement = field(default=None)
    angular_velocity: GyroMeasurement = field(default=None)
    orientation: MagMeasurement = field(default=None)

    def __post_init__(self):
        if self.linear_acceleration is None:
            self.linear_acceleration = AccelMeasurement()
        if self.angular_velocity is None:
            self.angular_velocity = GyroMeasurement()
        if self.orientation is None:
            self.orientation = MagMeasurement()

    def __str__(self):
        s = 't:%g, %s, %s, %s' % (self.t, str(self.linear_acceleration), str(self.angular_velocity), str(self.orientation))
        return s

@dataclass
class EncoderMeasurement(Measurement):
    ds: int = field(default=None)
    fl: int = field(default=None)
    fr: int = field(default=None)
    bl: int = field(default=None)
    br: int = field(default=None)

    def __str__(self):
        return 't:{self.t}, ds:{self.ds}, fl:{self.fl}, fr:{self.fr}, bl:{self.bl}, br:{self.br}'.format(self=self)

if __name__ == '__main__':
    import pdb
    config = KinematicBicycleConfig(dt=0.1, noise=True, noise_cov=4*np.eye(6))
    pdb.set_trace()

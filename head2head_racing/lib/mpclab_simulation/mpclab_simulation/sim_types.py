#!/usr/bin python3

from dataclasses import dataclass, field
from mpclab_common.pytypes import PythonMsg

@dataclass
class GPSConfig(PythonMsg):
    rate_div: float = field(default = 3)     # measurement is updated every so many intervals

    n_bound: float  = field(default = 2.5)

    #TODO: These parameters are contrived and may not resemble the physical vehicle
    x_std: float    = field(default = 0.03)
    y_std: float    = field(default = 0.03)
    z_std: float    = field(default = 0.15)

    offset_x: float = field(default = 0.03)     # constant x offset from the vehicle center of mass to the GPS sensor center
    offset_y: float = field(default = 0.0)
    offset_z: float = field(default = 0.15)

    ref_offset_x: float = field(default = 1) # constant x offset from the vehicle's starting position to the GPS reference position
    ref_offset_y: float = field(default = 2)
    ref_offset_z: float = field(default = 3)


@dataclass
class T265SimConfig(PythonMsg):
    n_bound: float  = field(default = 0.5)

    x_std: float    = field(default = None)
    y_std: float    = field(default = None)
    z_std: float    = field(default = None)
    yaw_std: float  = field(default = None)
    pitch_std: float  = field(default = None)
    roll_std: float  = field(default = None)

    v_long_std: float = field(default = None)
    v_tran_std: float = field(default = None)
    v_vert_std: float = field(default = None)
    yaw_dot_std:float = field(default = None)
    roll_dot_std:float = field(default = None)
    pitch_dot_std:float = field(default = None)

    a_long_std: float = field(default = None)
    a_tran_std: float = field(default = None)
    a_vert_std: float = field(default = None)

    offset_long: float = field(default = 0)
    offset_tran: float = field(default = 0)
    offset_vert: float = field(default = 0)
    offset_yaw: float = field(default = 0)

    dt: float = field(default = None)
    heading_drift_rate: float = field(default=None)

@dataclass
class ViveSimConfig(PythonMsg):
    n_bound: float  = field(default = 0.5)

    x_std: float    = field(default = None)
    y_std: float    = field(default = None)
    z_std: float    = field(default = None)
    yaw_std: float  = field(default = None)
    pitch_std: float  = field(default = None)
    roll_std: float  = field(default = None)

    v_long_std: float = field(default = None)
    v_tran_std: float = field(default = None)
    v_vert_std: float = field(default = None)
    yaw_dot_std:float = field(default = None)
    roll_dot_std:float = field(default = None)
    pitch_dot_std:float = field(default = None)

    origin_x: float   = field(default = 0)
    origin_y: float   = field(default = 0)
    origin_z: float   = field(default = 0)
    origin_yaw: float = field(default = 0)

    offset_long: float = field(default = 0)
    offset_tran: float = field(default = 0)
    offset_vert: float = field(default = 0)
    offset_yaw: float = field(default = 0)

@dataclass
class OptiTrackSimConfig(PythonMsg):
    n_bound: float          = field(default = 0.5)

    x_std: float            = field(default = None)
    y_std: float            = field(default = None)
    z_std: float            = field(default = None)

    yaw_std: float          = field(default = None)
    pitch_std: float        = field(default = None)
    roll_std: float         = field(default = None)

    v_long_std: float       = field(default = None)
    v_tran_std: float       = field(default = None)
    v_vert_std: float       = field(default = None)

    yaw_dot_std:float       = field(default = None)
    roll_dot_std:float      = field(default = None)
    pitch_dot_std:float     = field(default = None)

    origin_x: float         = field(default = 0)
    origin_y: float         = field(default = 0)
    origin_z: float         = field(default = 0)

    origin_roll: float      = field(default = 0)
    origin_pitch: float     = field(default = 0)
    origin_yaw: float       = field(default = 0)

    offset_long: float      = field(default = 0)
    offset_tran: float      = field(default = 0)
    offset_vert: float      = field(default = 0)

    offset_roll: float      = field(default = 0)
    offset_pitch: float     = field(default = 0)
    offset_yaw: float       = field(default = 0)

@dataclass
class IMUSimConfig(PythonMsg):
    n_bound: float  = field(default = 0.5)

    yaw_std: float          = field(default = None)
    pitch_std: float        = field(default = None)
    roll_std: float         = field(default = None)

    yaw_dot_std: float      = field(default = None)
    roll_dot_std: float     = field(default = None)
    pitch_dot_std: float    = field(default = None)

    a_long_std: float       = field(default = None)
    a_tran_std: float       = field(default = None)
    a_vert_std: float       = field(default = None)

    offset_long: float      = field(default = 0)
    offset_tran: float      = field(default = 0)
    offset_vert: float      = field(default = 0)
    offset_yaw: float       = field(default = 0)
    offset_pitch: float     = field(default = 0)
    offset_roll: float      = field(default = 0)

@dataclass
class D435iSimConfig(PythonMsg):
    n_bound: float  = field(default = 0.5)

    yaw_dot_std: float      = field(default = None)
    roll_dot_std: float     = field(default = None)
    pitch_dot_std: float    = field(default = None)

    a_long_std: float       = field(default = None)
    a_tran_std: float       = field(default = None)
    a_vert_std: float       = field(default = None)

    offset_long: float      = field(default = 0)
    offset_tran: float      = field(default = 0)
    offset_vert: float      = field(default = 0)
    offset_yaw: float       = field(default = 0)
    offset_pitch: float     = field(default = 0)
    offset_roll: float      = field(default = 0)

@dataclass
class EncConfig(PythonMsg):
    rate_div: float = field(default = 1)
    n_bound: float  = field(default = 0.5)
    v_std: float    = field(default = 0.01)

@dataclass
class EncMsg(PythonMsg):
    t: float        = field(default = None)
    fl: float       = field(default = None)
    fr: float       = field(default = None)
    bl: float       = field(default = None)
    br: float       = field(default = None)
    ds: float       = field(default = None)

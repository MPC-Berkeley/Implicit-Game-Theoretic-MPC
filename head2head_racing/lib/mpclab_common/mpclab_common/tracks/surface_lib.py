#!/usr/bin/env python3

import numpy as np
from numpy import pi

from mpclab_common.pytypes import VehicleState

from barc3d.surfaces.frenet_offset_surface import FrenetOffsetSurface, \
    FrenetOffsetSurfaceConfig, INTERP_PARTIAL_PCHIP, \
    FrenetExpOffsetSurface, FrenetExpOffsetSurfaceConfig
from barc3d.pytypes import VehicleState as VehicleState3D

class FrenetExpOffsetSurfaceMod(FrenetExpOffsetSurface):

    def global_to_frenet(self, state):
        if isinstance(state, VehicleState):
            state_3d = VehicleState3D()
            self.state_to_state3d(state, state_3d)
            super().global_to_frenet(state_3d)
            self.state3d_to_state(state_3d, state)
        else:
            super().global_to_frenet(state)
    
    def state_to_state3d(self, state: VehicleState, state_3d: VehicleState3D):
        state_3d.t      = state.t
        state_3d.x.xi   = state.x.x
        state_3d.x.xj   = state.x.y
        state_3d.x.xk   = state.x.z
        state_3d.q.qi   = state.q.qi
        state_3d.q.qj   = state.q.qj
        state_3d.q.qk   = state.q.qk
        state_3d.q.qr   = state.q.qr
        state_3d.v.v1   = state.v.v_long
        state_3d.v.v2   = state.v.v_tran
        state_3d.v.v3   = state.v.v_n
        state_3d.w.w1   = state.w.w_phi
        state_3d.w.w2   = state.w.w_theta
        state_3d.w.w3   = state.w.w_psi
        state_3d.p.s    = state.p.s
        state_3d.p.y    = state.p.x_tran
        state_3d.p.n    = state.p.n
        state_3d.p.ths  = state.p.e_psi
        state_3d.pt.ds  = state.pt.ds
        state_3d.pt.dy  = state.pt.dx_tran
        state_3d.pt.dths = state.pt.de_psi
        state_3d.pt.dn  = state.pt.dn
        state_3d.u.a    = state.u.u_a
        state_3d.u.y    = state.u.u_steer
    
    def state3d_to_state(self, state_3d: VehicleState3D, state: VehicleState):
        state.t             = state_3d.t
        state.x.x           = state_3d.x.xi
        state.x.y           = state_3d.x.xj
        state.x.z           = state_3d.x.xk
        state.q.qi          = state_3d.q.qi
        state.q.qj          = state_3d.q.qj
        state.q.qk          = state_3d.q.qk
        state.q.qr          = state_3d.q.qr
        state.v.v_long      = state_3d.v.v1
        state.v.v_tran      = state_3d.v.v2
        state.v.v_n         = state_3d.v.v3
        state.w.w_phi       = state_3d.w.w1
        state.w.w_theta     = state_3d.w.w2
        state.w.w_psi       = state_3d.w.w3
        state.p.s           = state_3d.p.s
        state.p.x_tran      = state_3d.p.y
        state.p.n           = state_3d.p.n
        state.p.e_psi       = state_3d.p.ths
        state.pt.ds         = state_3d.pt.ds
        state.pt.dx_tran    = state_3d.pt.dy
        state.pt.de_psi     = state_3d.pt.dths
        state.pt.dn         = state_3d.pt.dn
        state.u.u_a         = state_3d.u.a
        state.u.u_steer     = state_3d.u.y

def _get_surface() -> FrenetOffsetSurface:
    # pylint: disable=line-too-long

    # currently, w > h, and some portions of later code assume this for computing the radius of the last 90 turn.

    # full width, left centerline to far right tangent of rightmost 180
    w = 2.251 + 1.1515 + 1.2206275 + 0.905905
    # full height - centerline to topmost part of topmost 180 centerline
    h = 1.2281825 + 2.25454 + 1.2206275

    # first 180 radius (at center)
    r1 = 1.125
    # first 90 radius (at center)
    r2 = 1.125
    # second 180 radius (at center)
    r3 = 1.125
    # second 90 radius (at center)
    # currently made as big as possible

    # initial transition distance (to help cleanly close)
    d0 = 0.1
    # curvature transition distance
    ks = 1
    # curvature
    K = 0.75 / 1.1

    # slope transition distance
    bs = 1
    # slope
    B = 0.3

    config = FrenetOffsetSurfaceConfig(y_min = -0.55, y_max = 0.55, use_pchip=INTERP_PARTIAL_PCHIP, closed=True)

    # interpolation for heading
    #             start       180 turn  straight        90 turn     straight        180 turn   straight   90 turn            end
    ds = np.array([0, d0, ks, r1*np.pi, w-(r1+r2+2*r2), r2*np.pi/2, h-(r1*2+r2+r3), r3*np.pi,  ks,        (h-r3-ks)*np.pi/2, w-h+r3-r1-d0])
    a =  np.array([0, 0,  0,  np.pi,    np.pi,          np.pi/2,    np.pi/2,        3*np.pi/2, 3*np.pi/2, np.pi*2,            np.pi*2]) -np.pi/2
    config.s = np.cumsum(ds)
    config.a = a
    # config.s = np.cumsum(np.concatenate((ds, ds[1:])))
    # config.a = np.concatenate((a, a[1:]))

    l1 = config.s[3]                # arc length to end of first 180
    l2 = config.s[6] - config.s[3]  # arc length from end of first 180 to start of second 180
    l3 = config.s[7] - config.s[6]  # arc length of second 180
    l4 = config.s[-1] - config.s[7] # arc length from end of second 180 to finish line

    # interpolation for curvature "t" is short for transition segment
    #              start   t   180       t            t   180       t
    ds = np.array([0, d0,  ks, r1*np.pi, ks, l2-2*ks, ks, r3*np.pi, ks, l4])
    k  = np.array([0,  0,  K,  K,        0,  0,       K,  K,        0,  0])
    config.k = k
    config.s_k = np.cumsum(ds)
    # config.k = np.concatenate((k, k[1:]))
    # config.s_k = np.cumsum(np.concatenate((ds, ds[1:])))

    # interpolation for cross-sectional slope.
    ds = np.array([0, l1, bs, l2-2*bs, bs, l3, bs, l4-2*bs, bs, d0])
    b  = np.array([0, 0,  B,  B,       0,  0, -B, -B,       0, 0])
    config.b = b
    config.s_b = np.cumsum(ds)
    # config.b = np.concatenate((b, b[1:]))
    # config.s_b = np.cumsum(np.concatenate((ds, ds[1:])))

    # interpolation for cross-sectional offset
    config.c = a * 0
    # config.c = np.concatenate((a, a[1:])) * 0

    # remove any zero curvature terms
    config.k[config.k == 0] = 0.001

    # add origin offset
    config.x0 = np.array([-3.4669, 1.9382 - 2.25 + ks, 0])

    surf = FrenetOffsetSurface()
    surf.initialize(config)
    return surf

def _get_identified_surface() -> FrenetExpOffsetSurfaceMod:

    config = FrenetExpOffsetSurfaceConfig(y_min = -0.4, y_max = 0.55, use_pchip=INTERP_PARTIAL_PCHIP, closed=True)

    ds = np.array(
        [0,2.251,pi*1.1515,0.901,1.0201675*pi/2,0.15,
        1.2281825*pi,2.25454,1.2206275*pi/2,0.905905])
    a = np.array([0,0,pi,pi,pi/2,pi/2,pi*1.5,pi*1.5,pi*2,pi*2]) - pi/2
    config.s = np.cumsum(ds)
    config.a = a

    config.b = np.array([1.,1.]) * 6.1637
    config.s_b = np.array([0, config.s.max()])

    config.c = np.array([1., 1.]) * -0.3767
    config.s_c = np.array([0, config.s.max()])

    config.d = np.array([0., 0., 1., 1., 0., 0.]) * 0.092119
    config.s_d = np.array([0., 8.5, 9.3, 11.6, 12.5, config.s.max()])

    config.e = np.array([1., 1.]) * 0.076
    config.s_e = np.array([0, config.s.max()])

    surf = FrenetExpOffsetSurfaceMod()
    surf.initialize(config)
    return surf

def _get_visualization_surface() -> FrenetExpOffsetSurfaceMod:

    config = FrenetExpOffsetSurfaceConfig(y_min = -0.4, y_max = 0.55, use_pchip=INTERP_PARTIAL_PCHIP, closed=True)

    ds = np.array(
        [0,2.251,pi*1.1515,0.901,1.0201675*pi/2,0.15,
        1.2281825*pi,2.25454,1.2206275*pi/2,0.905905])
    a = np.array([0,0,pi,pi,pi/2,pi/2,pi*1.5,pi*1.5,pi*2,pi*2])
    config.s = np.cumsum(ds)
    config.a = a

    config.b = np.array([1.,1.]) * 6.1637
    config.s_b = np.array([0, config.s.max()])

    config.c = np.array([1., 1.]) * -0.3767
    config.s_c = np.array([0, config.s.max()])

    config.d = np.array([0., 0., 1., 1., 0., 0.]) * 0.092119
    config.s_d = np.array([0., 8.5, 9.3, 11.6, 12.5, config.s.max()])

    config.e = np.array([1., 1.]) * 0.076
    config.s_e = np.array([0, config.s.max()])

    surf = FrenetExpOffsetSurfaceMod()
    surf.initialize(config)
    return surf

def get_lookahead(s: float, surf, l: float = 1.5, dl: float = 0.5):
    n = int(l/dl)
    lookahead_s = np.linspace(0, l, n)
    lookahead_p = []
    for ls in lookahead_s:
        _s = np.mod(s + ls, surf.s_max())
        if isinstance(surf, FrenetOffsetSurface):
            lookahead_p.append([float(surf.k(_s)), float(surf.b(_s))])
        elif isinstance(surf, FrenetExpOffsetSurface):
            lookahead_p.append([float(surf.tha(_s)), float(surf.d(_s))])
    return lookahead_p
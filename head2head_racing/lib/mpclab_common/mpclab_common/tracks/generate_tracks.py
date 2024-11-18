import numpy as np
import pickle
import pdb
import csv
import os
import urllib.request
import json
from matplotlib import pyplot as plt

from scipy import optimize
from scipy.interpolate import CubicSpline

from mpclab_common.tracks.radius_arclength_track import RadiusArclengthTrack
from mpclab_common.tracks.casadi_bspline_track import CasadiBSplineTrack

def get_save_folder():
    return os.path.join(os.path.dirname(__file__), 'track_data')

def generate_curvature_and_path_length_track(filename, track_width, cl_segs, slack):

    save_folder = get_save_folder()
    os.makedirs(save_folder, exist_ok=True)

    save_path = os.path.join(save_folder, filename)
    np.savez(save_path, save_mode = 'radius_and_arc_length', track_width = track_width, cl_segs = cl_segs, slack = slack)
    return

def generate_casadi_bspline_track(filename, xy, left_w, right_w, s):
    save_folder = get_save_folder()
    os.makedirs(save_folder, exist_ok=True)

    save_path = os.path.join(save_folder, filename)
    np.savez(save_path, save_mode = 'casadi_bspline', xy_waypoints=xy, left_width=left_w, right_width=right_w, s_waypoints=s)

def generate_straight_track():
    track_width = 1.0
    slack = 0.45

    length = 10.0
    cl_segs = np.array([length, 0]).reshape((1,-1))

    generate_curvature_and_path_length_track('Straight_Track', track_width, cl_segs, slack)
    print('Generated Straight_Track')
    return

def generate_LTrack_barc():
    track_width = 1.1
    slack     = 0.3

    ninety_radius_1     = ((1.5912+0.44723)/2 + (1.5772+0.46504)/2)/2 
    ninety_radius_2     = ((0.65556 + 1.12113/2) + (0.6597 + 1.13086/2))/2

    oneeighty_radius_1  = (1.171 + 1.1473/2 + 1.1207/2)/2
    oneeighty_radius_2  = (1.3165 + 1.15471/2 + 1.12502/2)/2

    straight_1          = 2.401 - 0.15
    straight_2          = 1.051 - 0.15
    straight_3          = 0.450 - 0.3
    straight_4          = 2*oneeighty_radius_1 + ninety_radius_1 + straight_3 - ninety_radius_2 #2.5515
    straight_5          = np.abs(straight_1 - straight_2 - ninety_radius_1 - 2*oneeighty_radius_2 + ninety_radius_2)

    cl_segs = np.array([[straight_1,                    0],
                        [np.pi*oneeighty_radius_1,      oneeighty_radius_1],
                        [straight_2,                    0],
                        [np.pi/2*ninety_radius_1,       -ninety_radius_1],
                        [straight_3,                    0],
                        [np.pi*oneeighty_radius_2,      oneeighty_radius_2],
                        [straight_4,                    0],
                        [np.pi/2*ninety_radius_2,       ninety_radius_2],
                        [straight_5,                    0]])

    generate_curvature_and_path_length_track('L_track_barc', track_width, cl_segs, slack)
    print('Generated L_track_barc')
    return

def generate_LTrack_barc_reverse():
    track_width = 1.1
    slack     = 0.3

    ninety_radius_1     = ((1.5912+0.44723)/2 + (1.5772+0.46504)/2)/2 
    ninety_radius_2     = ((0.65556 + 1.12113/2) + (0.6597 + 1.13086/2))/2

    oneeighty_radius_1  = (1.171 + 1.1473/2 + 1.1207/2)/2
    oneeighty_radius_2  = (1.3165 + 1.15471/2 + 1.12502/2)/2

    straight_1          = 2.401 - 0.15
    straight_2          = 1.051 - 0.15
    straight_3          = 0.450 - 0.3
    straight_4          = 2*oneeighty_radius_1 + ninety_radius_1 + straight_3 - ninety_radius_2 #2.5515
    straight_5          = np.abs(straight_1 - straight_2 - ninety_radius_1 - 2*oneeighty_radius_2 + ninety_radius_2)

    cl_segs = np.array([[straight_5,                    0],
                        [np.pi/2*ninety_radius_2,       -ninety_radius_2],
                        [straight_4,                    0],
                        [np.pi*oneeighty_radius_2,      -oneeighty_radius_2],
                        [straight_3,                    0],
                        [np.pi/2*ninety_radius_1,       ninety_radius_1],
                        [straight_2,                    0],
                        [np.pi*oneeighty_radius_1,      -oneeighty_radius_1],
                        [straight_1,                    0]])

    generate_curvature_and_path_length_track('L_track_barc_reverse', track_width, cl_segs, slack)
    print('Generated L_track_barc_reverse')
    return

# def generate_Lab_track():
#     track_width = 0.75
#     slack     = 0.45
#
#     straight_1 = 2.3364
#     straight_2 = 1.9619
#     straight_3 = 0.5650
#     straight_4 = 0.5625
#     straight_5 = 1.3802
#     straight_6 = 0.8269
#     total_width = 3.4779
#
#     ninety_radius = 0.758 # (3.49 - 2.11)/2 # Radius of 90 deg turns
#
#     thirty_secant = 0.5*((straight_6 + straight_1)-(straight_3 + straight_5 + straight_4*np.cos(np.pi/6)))/np.cos(15*np.pi/180)
#     thirty_radius = 0.5*(thirty_secant)/np.cos(75*np.pi/180) # Radius of 30 deg turns
#
#     oneeighty_radius = 0.5*(total_width-straight_4*np.sin(np.pi/6)-2*thirty_secant*np.cos(75*np.pi/180))
#
#     cl_segs = np.array([[straight_1,                0], #[2.375, 0],
#                         [np.pi/2 * ninety_radius,   ninety_radius],
#                         [straight_2,                0],  # [2.11, 0],
#                         [np.pi/2 * ninety_radius,   ninety_radius],
#                         [straight_3,                0], # [0.62, 0],
#                         [np.pi/6 * thirty_radius,   thirty_radius],
#                         [straight_4,                0], # [0.555, 0],
#                         [np.pi/6 * thirty_radius,   -thirty_radius],
#                         [straight_5,                0], #[1.08, 0],
#                         [np.pi * oneeighty_radius,  oneeighty_radius],
#                         [straight_6,                0]]) #[0.78, 0]])
#
#     generate_curvature_and_path_length_track('Lab_Track_barc', track_width, cl_segs, slack)
#     print('Generated Lab_Track_barc')
#     return

def generate_Lab_track():
    track_width = 1.19
    slack     = 0.45
    origin = [3.377, 0.9001]
    pt_before_circle_inside = [2.736, 2.078]
    pt_after_half_circle_inside = [1.211,  2.078]
    end_2nd_straight_inside = [1.300, -0.396]
    last_pt = [2.8166, -0.396]
    straight_1 = pt_before_circle_inside[1] - origin[1] - .225
    straight_2 = -end_2nd_straight_inside[1] + pt_after_half_circle_inside[1] - .45
    straight_final = straight_2 - straight_1
    radius_first = (-pt_after_half_circle_inside[0] + pt_before_circle_inside[0] + track_width)/2

    cl_segs = np.array([[straight_1,                0],
                        [np.pi * radius_first,  radius_first],
                        [straight_2,                0],
                        [np.pi * radius_first,  radius_first],
                        [straight_final, 0]])

    generate_curvature_and_path_length_track('Lab_Track_barc', track_width, cl_segs, slack)
    print('Generated Lab_Track_barc')
    return

def generate_Monza_track():
    track_width = 1.5
    slack     = 0.45 # Don't change?

    straight_01 = np.array([[3.0, 0]])

    enter_straight_length = 0.5
    curve_length = 3
    curve_swept_angle = np.pi - np.pi/40
    s = 1
    exit_straight_length = 2.2
    curve_11 = np.array([
                        [curve_length, s * curve_length / curve_swept_angle],
                        [exit_straight_length, 0]])


    # WACK CURVE

    enter_straight_length = 0.2
    curve_length = 0.8
    curve_swept_angle = np.pi / 4
    s = -1
    exit_straight_length = 0.4
    curve_10 = np.array([
                         [curve_length, s * curve_length / curve_swept_angle],
                         [exit_straight_length, 0]])

    enter_straight_length = 0.05
    curve_length = 1.2
    curve_swept_angle = np.pi / 4
    s = 1
    exit_straight_length = 0.4
    curve_09 = np.array([
                         [curve_length, s * curve_length / curve_swept_angle],
                         [exit_straight_length, 0]])

    enter_straight_length = 0.05
    curve_length = 0.8
    curve_swept_angle = np.pi / 4
    s = -1
    exit_straight_length = 2.5
    curve_08 = np.array([
                         [curve_length, s * curve_length / curve_swept_angle],
                         [exit_straight_length, 0]])


    # Curve mini before 07
    enter_straight_length = 0
    curve_length = 1.0
    curve_swept_angle = np.pi / 12
    s = -1
    exit_straight_length = 1.5
    curve_before_07 = np.array([
                         [curve_length, s * curve_length / curve_swept_angle],
                         [exit_straight_length, 0]])

    # Curve 07
    enter_straight_length = 0
    curve_length = 1.2
    curve_swept_angle = np.pi / 3
    s = 1
    exit_straight_length = 1.5
    curve_07 = np.array([ [curve_length, s * curve_length / curve_swept_angle],
                                [exit_straight_length, 0]])

    # Curve 06
    enter_straight_length = 0
    curve_length = 1.5
    curve_swept_angle = np.pi / 2
    s = 1
    exit_straight_length = 2.0
    curve_06 = np.array([
                         [curve_length, s * curve_length / curve_swept_angle],
                         [exit_straight_length, 0]])

    # Chicane 05
    enter_straight_length = 0
    curve1_length = 0.4
    s1, s2 = 1, -1
    curve1_swept_angle = np.pi/8
    mid_straight_length = 1.0
    curve2_length = 0.4
    curve2_swept_angle = np.pi/8
    exit_straight_length = 1.0 + np.abs(np.sin(-1.6493361431346418)*(0.4299848245548139+0.0026469133545887783))


    chicane_05 = np.array([
                      [curve1_length, s1 * curve1_length / curve1_swept_angle],
                      [mid_straight_length, 0],
                      [curve2_length, s2 * curve2_length / curve2_swept_angle],
                      [exit_straight_length, 0]])

    # Curve 03
    enter_straight_length = 0.0
    curve_length = 4.0
    curve_swept_angle = np.pi / 2 + np.pi/16
    s = 1
    exit_straight_length = 2.0
    curve_03 = np.array([
                         [curve_length, s * curve_length / curve_swept_angle],
                         [exit_straight_length, 0]])

    # Curve 02
    enter_straight_length = 0.0
    curve_length = 1.0
    curve_swept_angle = np.pi / 10
    s = -1
    exit_straight_length = 1.0
    curve_02 = np.array([
                         [curve_length, s * curve_length / curve_swept_angle],
                         [exit_straight_length, 0]])

    # Final curve
    curve_length = 1.0
    curve_swept_angle = -np.pi / 10 + 0.11780972450961658
    exit_straight_length = 1.0 - 1.26433096e-01 + 0.0002070341780330276 + 0.00021382215942933325 + 1.6293947847880575e-05- 0.00023011610727452503
    s = -1
    curve_Final = np.array([
                         [curve_length, s * curve_length / curve_swept_angle],
                         [exit_straight_length, 0]])


    cl_segs = []
    cl_segs.extend(straight_01)
    cl_segs.extend(curve_11)
    cl_segs.extend(curve_10)
    cl_segs.extend(curve_09)
    cl_segs.extend(curve_08)
    cl_segs.extend(curve_before_07)
    cl_segs.extend(curve_07)
    cl_segs.extend(curve_06)
    cl_segs.extend(chicane_05)
    cl_segs.extend(curve_03)
    cl_segs.extend(curve_02)
    cl_segs.extend(curve_Final)


    generate_curvature_and_path_length_track('Monza_Track', track_width, cl_segs, slack)
    print('Generated Monza_Track')
    return

def generate_Circle_track():
    track_width = 1.0
    slack     = 0.45
    radius=4.5
    cl_segs = np.array([[np.pi*radius, radius],
                                [np.pi*radius, radius]])

    generate_curvature_and_path_length_track('Circle_Track_barc', track_width, cl_segs, slack)
    print('Generated Circle_Track_barc')
    return

def generate_Oval_track():
    straight_len=2.0
    curve_len=5.5
    track_width=1.0
    slack=0.45
    cl_segs = np.array([[straight_len/2, 0],
                                [curve_len, curve_len/np.pi],
                                [straight_len, 0],
                                [curve_len, curve_len/np.pi],
                                [straight_len/2, 0]])

    generate_curvature_and_path_length_track('Oval_Track_barc', track_width, cl_segs, slack)
    print('Generated Oval_Track_barc')
    return

def generate_f1_austin_track(tenth_scale=True):
    path = 'f1_source_data/Austin.csv'
    scaling = 0.1 if tenth_scale else 1.0
    file_name = 'f1_austin'
    if tenth_scale:
        file_name += '_tenth_scale'

    data = []
    with open(path, 'r') as f:
        _data = csv.reader(f)
        next(_data)
        for d in _data:
            data.append([float(_d) for _d in d])
    data = np.array(data)*scaling
    data = np.vstack((data, data[0].reshape((1,-1))))
    track = CasadiBSplineTrack(data[:,:2], data[:,2], data[:,3], 2.0)

    generate_casadi_bspline_track(file_name, track.xy_waypoints, track.left_width_points, track.right_width_points, track.s_waypoints)
    print(f'Generated {file_name}')
    return

def generate_f1_nuerburgring_track(tenth_scale=True):
    path = 'f1_source_data/Nuerburgring.csv'
    scaling = 0.1 if tenth_scale else 1.0
    file_name = 'f1_nuerburgring'
    if tenth_scale:
        file_name += '_tenth_scale'

    data = []
    with open(path, 'r') as f:
        _data = csv.reader(f)
        next(_data)
        for d in _data:
            data.append([float(_d) for _d in d])
    data = np.array(data)*scaling
    data = np.vstack((data, data[0].reshape((1,-1))))
    track = CasadiBSplineTrack(data[:,:2], data[:,2], data[:,3], 2.0)

    generate_casadi_bspline_track(file_name, track.xy_waypoints, track.left_width_points, track.right_width_points, track.s_waypoints)
    print(f'Generated {file_name}')
    return

def generate_Indy_track():
    straight_len=1006
    short_len = 201
    curve_len= 402
    track_width=15
    slack=0.45
    cl_segs = np.array([[straight_len/2, 0],
                                [curve_len, 2*curve_len/np.pi],
                                [short_len, 0],
                                [curve_len, 2*curve_len/np.pi],
                                [straight_len, 0],
                                [curve_len, 2*curve_len/np.pi],
                                [short_len, 0],
                                [curve_len, 2*curve_len/np.pi],
                                [straight_len/2, 0]])

    generate_curvature_and_path_length_track('Indy_Track', track_width, cl_segs, slack)
    print('Generated Indy_Track')
    return

def normsort(xycoords):
    sortedcoords = np.zeros(xycoords.shape)
    sorted_idx = []
    for i in range(xycoords.shape[0]):
        if i == 0:
            sortedcoords[i,:] = xycoords[i]
            sorted_idx.append(i)
        else:
            idx = np.argsort(np.linalg.norm(xycoords - sortedcoords[i-1], axis = 1)).tolist()
            while idx[0] in sorted_idx:
                idx.pop(0)
            sortedcoords[i,:] = xycoords[idx[0]]
            sorted_idx.append(idx[0])
    return sortedcoords

def main():
    # generate_straight_track()
    # generate_LTrack_barc()
    # generate_LTrack_barc_reverse()
    # generate_Lab_track()
    # generate_Circle_track()
    # generate_Oval_track()
    # generate_Indy_track()
    # generate_f1_austin_track()
    generate_f1_nuerburgring_track(tenth_scale=False)
    # generate_Monza_track()

if __name__ == '__main__':
    main()

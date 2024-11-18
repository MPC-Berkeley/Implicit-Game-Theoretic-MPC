import numpy as np
from numpy import linalg as la
import casadi as ca
from matplotlib import pyplot as plt

import pdb
import os
import pickle
import copy
import csv

from mpclab_common.tracks.radius_arclength_track import RadiusArclengthTrack
from mpclab_common.tracks.casadi_bspline_track import CasadiBSplineTrack

from mpclab_common.tracks.generate_tracks import get_save_folder
from mpclab_common.tracks.surface_lib import _get_visualization_surface, _get_identified_surface

def get_available_tracks():
    save_folder = get_save_folder()
    return os.listdir(save_folder)
    
def get_track(track_file):
    if track_file == 'visualization_surface':
        return _get_visualization_surface()
    elif track_file == 'identified_surface':
        return _get_identified_surface()
    
    tracks = get_available_tracks()
    track_found = False
    for t in tracks:
        if track_file == t.split('.')[0]:
            track_file = t
            track_found = True
            break
    
    if not track_found:
        raise ValueError('Chosen Track is unavailable: %s\nlooking in:%s\n Available Tracks: %s'%(track_file, 
                        os.path.join(os.path.dirname(__file__), 'tracks', 'track_data'),
                        str(get_available_tracks())))
    
    save_folder = get_save_folder()
    load_file = os.path.join(save_folder, track_file)

    if track_file.endswith('.npz'):
        data = np.load(load_file, allow_pickle = True)
        if data['save_mode'] == 'radius_and_arc_length':
            track = RadiusArclengthTrack()
            track.initialize(data['track_width'], data['slack'], data['cl_segs'])
        elif data['save_mode'] == 'casadi_bspline':
            track = CasadiBSplineTrack(data['xy_waypoints'], data['left_width'], data['right_width'], 2.0, s_waypoints=data['s_waypoints'])
        else:
            raise NotImplementedError('Unknown track save mode: %s' % data['save_mode'])
    elif track_file.endswith('.pkl'):
        with open(load_file, 'rb') as f:
            track = pickle.load(f)
    else:
        raise ValueError(f'Unable to load track file {load_file}')
        
    return track   

def load_mpclab_raceline(file_path, track_name, time_scale=1.0):
    track = get_track(track_name)
    f = np.load(file_path)
    raceline_mat = np.vstack((f['x'], f['y'], f['psi'], f['v_long']/time_scale, f['v_tran']/time_scale, f['psidot']/time_scale, f['e_psi'], f['s'], f['e_y'])).T
    T = f['t']*time_scale

    raceline_mat2 = copy.copy(raceline_mat)
    raceline_mat2[:,7] += track.track_length
    T2 = copy.copy(T)
    T2 += T[-1]
    raceline_two_laps = np.vstack((raceline_mat, raceline_mat2[1:]))
    T_two_laps = np.append(T, T2[1:])
    t_sym = ca.MX.sym('t', 1)
    raceline_interp = []
    for i in range(raceline_mat.shape[1]):
        raceline_interp.append(ca.interpolant(f'x{i}', 'linear', [T_two_laps], raceline_two_laps[:,i]))
    raceline = ca.Function('raceline', [t_sym], [ri(t_sym) for ri in raceline_interp])
    s2t = ca.interpolant('s2t', 'linear', [raceline_two_laps[:,7]], T_two_laps)

    return raceline, s2t, raceline_mat

def load_tum_raceline(file_path, track_name, tenth_scale=False, time_scale=1.0, segment=None, resample_resolution=None):
    track = get_track(track_name)
    size_scale = 0.1 if tenth_scale else 1.0

    raceline_mat = []
    raceline_s = []
    with open(file_path, 'r') as f:
        _data = csv.reader(f, delimiter=';')
        for d in _data:
            if '#' in d[0]:
                continue
            _s, _x, _y, _psi, k, _v, a = [float(_d) for _d in d]
            x = _x*size_scale
            y = _y*size_scale
            v = _v*size_scale/time_scale
            psi = _psi + np.pi/2
            s, ey, epsi = track.global_to_local((x, y, psi))
            # pdb.set_trace()
            if len(raceline_mat) > 0:
                if s < raceline_mat[-1][7]:
                    s += track.track_length
            raceline_mat.append([x, y, psi, v, 0, 0, epsi, s, ey])
            raceline_s.append(_s*size_scale)
    raceline_mat = np.array(raceline_mat)
    T = [0.0]
    for k in range(len(raceline_s)-1):
        ds = raceline_s[k+1] - raceline_s[k]
        v = raceline_mat[k, 3]
        dt = ds/v
        T.append(T[-1]+dt)
    T = np.array(T)

    if not resample_resolution:
        resample_resolution = int(len(raceline_s)/raceline_s[-1])

    if segment:
        t_sym = ca.MX.sym('t', 1)

        _raceline_interp = []
        for i in range(raceline_mat.shape[1]):
            _raceline_interp.append(ca.interpolant(f'x{i}', 'linear', [T], raceline_mat[:,i]))
        _raceline = ca.Function('raceline', [t_sym], [ri(t_sym) for ri in _raceline_interp])
        _s2t = ca.interpolant('s2t', 'linear', [raceline_mat[:,7]], T)

        n = int((segment[1]-segment[0])*resample_resolution)
        _T = np.array(_s2t(np.linspace(segment[0], segment[1], n))).squeeze()
        T = _T - _T[0]
        raceline_mat = np.array(_raceline(_T)).squeeze().T
        raceline_mat[:,7] -= segment[0]
        raceline_interp = []
        for i in range(raceline_mat.shape[1]):
            raceline_interp.append(ca.interpolant(f'x{i}', 'linear', [T], raceline_mat[:,i]))
        raceline = ca.Function('raceline', [t_sym], [ri(t_sym) for ri in raceline_interp])
        s2t = ca.interpolant('s2t', 'linear', [raceline_mat[:,7]], T)
    else:
        raceline_mat2 = copy.copy(raceline_mat)
        raceline_mat2[:,7] += track.track_length
        T2 = copy.copy(T)
        T2 += T[-1]
        raceline_two_laps = np.vstack((raceline_mat, raceline_mat2[1:]))
        T_two_laps = np.append(T, T2[1:])
        t_sym = ca.MX.sym('t', 1)
        raceline_interp = []
        for i in range(raceline_mat.shape[1]):
            raceline_interp.append(ca.interpolant(f'x{i}', 'linear', [T_two_laps], raceline_two_laps[:,i]))
        raceline = ca.Function('raceline', [t_sym], [ri(t_sym) for ri in raceline_interp])
        s2t = ca.interpolant('s2t', 'linear', [raceline_two_laps[:,7]], T_two_laps)

    return raceline, s2t, raceline_mat

def test_reconstruction_accuracy(track): # make sure the global <--> local conversions work well
    n_segs = 1000
    s_interp = np.linspace(0,track.track_length-1e-2,n_segs)
    errors = []
    for s in s_interp:
        e_y = np.random.uniform(-0.5*track.track_width, 0.5*track.track_width)
        e_y /= 10.
        e_psi = np.random.uniform(-1,1)
        x, y, psi = track.local_to_global((s, e_y, e_psi))
        s_n, e_y_n, e_psi_n = track.global_to_local((x, y, psi))
        
        if (s-s_n) / track.track_length > 0.9:
            s_n += track.track_length
        if (s-s_n) / track.track_length < -0.9:
            s_n -= track.track_length
        '''if (s-s_n) **2 > 3:
            print(s)
            pdb.set_trace()
        if (e_psi-e_psi_n)**2 > 5:
            pdb.set_trace()'''
        errors.append([(s-s_n)**2, (e_y - e_y_n)**2,(e_psi - e_psi_n)**2])
    for i in range(3):
        data = [err[i] for err in errors]
        plt.plot(data)
    plt.legend(('s','e_y','e_psi'))
    plt.title('Reconstruction errors')
    plt.show()


def main():
    track = get_track('F1_HU')
    test_reconstruction_accuracy(track)

    fig = plt.figure()
    ax = plt.subplot(111)
    track.plot_map(ax)
    
    x_list = []
    y_list = [] 
    for i in range(1000):
        #s = np.random.random() * track.track_length
        s = np.random.randint(0,10) * track.track_length / 10
        x_tran = -track.half_width + 2* np.random.random() * track.half_width
        
        x_coord = track.local_to_global((s,x_tran,0))
        
        x_list.append(x_coord[0])
        y_list.append(x_coord[1])
        
    plt.plot(x_list,y_list, 'o')
    
    plt.show()


def preview_texture():
    track = get_track('LTrack_barc')
    V,I = track.generate_texture(n_s = 100, n_w = 10)
    
    x = V[:,0]
    y = V[:,1]
    z = V[:,2]
    fig = plt.figure(figsize=plt.figaspect(0.5))
    ax = fig.add_subplot(1, 2, 1, projection='3d')
    ax.plot_trisurf(x, y, z, triangles=I, cmap=plt.cm.Spectral)
    ax.set_zlim(-1, 1)
    plt.show()

if __name__ == '__main__':
    #main()    
    preview_texture()


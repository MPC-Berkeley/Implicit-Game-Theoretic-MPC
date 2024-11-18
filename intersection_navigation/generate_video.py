import pickle
import matplotlib.pyplot as plt
import numpy as np
import torch as th
import os
from infrastructure.VehicleReference import VehicleReference
from utils import transform_Rectangle, frenet2global, route_encoding, route_one_hot_encoding, scenario_encoding, scenario_one_hot_encoding
import scipy as sp
from matplotlib.animation import FuncAnimation
import copy
import cv2
import os
import pdb
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import FormatStrFormatter 
from models.model_ed import mlp

def plot_scene(ax,cur_agent, road_length=50,road_width=11.4,fillet_radius=2.8,title=None):
    w = 2
    l = 4.47
    ax.clear()  # Clear the axis before drawing new artists
    ax.add_patch(plt.Rectangle((0,(road_width-road_length)/2), road_length, road_length, color='gray'))
    patch1_width = (road_length - road_width) /2
    patch1_length = patch1_width - fillet_radius

    ax.add_patch(plt.Rectangle((0,(road_width-road_length)/2), patch1_width, patch1_length, color='white',linewidth=0))
    ax.add_patch(plt.Rectangle((0,(road_width-road_length)/2), patch1_length, patch1_width, color='white',linewidth=0))
    ax.add_patch(plt.Circle(((road_length - road_width)/2-fillet_radius,0-fillet_radius),fillet_radius, color='white',linewidth=0))

    ax.add_patch(plt.Rectangle((0,(road_length+road_width)/2), patch1_width, -patch1_length, color='white',linewidth=0))
    ax.add_patch(plt.Rectangle((0,(road_length+road_width)/2), patch1_length, -patch1_width, color='white',linewidth=0))
    ax.add_patch(plt.Circle(((road_length - road_width)/2-fillet_radius,road_width+fillet_radius),fillet_radius, color='white',linewidth=0))
    
    ax.add_patch(plt.Rectangle((road_length,(road_length+road_width)/2), -patch1_width, -patch1_length, color='white',linewidth=0))
    ax.add_patch(plt.Rectangle((road_length,(road_length+road_width)/2), -patch1_length, -patch1_width, color='white',linewidth=0))
    ax.add_patch(plt.Circle(((road_length + road_width)/2+fillet_radius,road_width+fillet_radius),fillet_radius, color='white',linewidth=0))
    
    ax.add_patch(plt.Rectangle((road_length,(road_width-road_length)/2), -patch1_width, patch1_length, color='white',linewidth=0))
    ax.add_patch(plt.Rectangle((road_length,(road_width-road_length)/2), -patch1_length, patch1_width, color='white',linewidth=0))
    ax.add_patch(plt.Circle(((road_length + road_width)/2+fillet_radius,0-fillet_radius),fillet_radius, color='white',linewidth=0))

    # ax.add_patch(plt.Rectangle((0,0), road_length, road_width, color='gray'))
    # ax.add_patch(plt.Rectangle((road_length/2-road_width/2,-road_length/2+road_width/2), road_width, road_length, color='gray'))
    for i in range(len(cur_agent)):
        if i ==0:
            color = 'green'
        else:
            color = 'blue'
        ax.add_patch(plt.Rectangle(transform_Rectangle(cur_agent[i]['state'].x, cur_agent[i]['state'].y,w,l/2, 0),angle=cur_agent[i]['state'].heading*180/np.pi,rotation_point='center',width=l,height=w,color=color,fill=False))

    ax.set_xlim(0,road_length)
    ax.set_ylim(road_width/2-road_length/2,road_length/2+road_width/2)
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title,fontsize=10)

def v_mp(s_i, s_noti):
    return s_i + s_noti

def generate_video(cl_traj, mode, dataset_dir=None, model_dir=None, routes_str=None, N=10, save_dir='/home/mpc/'):
    n_points = 100
    routes = scenario_encoding(routes_str)
    N = N
    if mode == 'GT':
        with open(dataset_dir, 'rb') as f:
            D = pickle.load(f)
        device = th.device('cuda' if th.cuda.is_available() else 'cpu')

        mlp_model = mlp(input_layer_size=len(D['test'][0][0]),
                        output_layer_size=len(D['test'][0][1]),
                        hidden_layer_sizes=[128 for _ in range(3)],
                        activation='tanh',
                        batch_norm=False)
        mlp_model.load_state_dict(th.load(model_dir))
        vf_copy = copy.deepcopy(mlp_model).to(device)
   

    def animate(t):
        fig, ax = plt.subplots(1, 3, figsize=(18,5))
        colorbars = [None, None]  # Track colorbars for removal
        # Clear each axis to remove old plot elements and prevent accumulation
        # Define current agent states
        cur_agent = [{'state': VehicleReference({'x': cl_traj[7 * 0 + 0, t], 'y': cl_traj[7 * 0 + 1, t], 'v': cl_traj[7 * 0 + 6, t],
                                                's': cl_traj[7 * 0 + 2, t], 'ey': cl_traj[7 * 0 + 3, t], 'epsi': cl_traj[7 * 0 + 4, t],
                                                'K': 0, 'heading': cl_traj[7 * 0 + 6, t]})},
                    {'state': VehicleReference({'x': cl_traj[7 * 1 + 0, t], 'K': 0, 'y': cl_traj[7 * 1 + 1, t], 'v': cl_traj[7 * 1 + 5, t],
                                                's': cl_traj[7 * 1 + 2, t], 'ey': cl_traj[7 * 1 + 3, t], 'epsi': cl_traj[7 * 1 + 4, t],
                                                'heading': cl_traj[7 * 1 + 6, t]})}]

        # Generate title for the middle plot
        time_str = "{:.2f}".format(t * 0.1)
        plot_scene(ax[1],cur_agent=cur_agent, road_length=50, road_width=11.4, fillet_radius=11.4-2.8,title=fr'$t={time_str} s$')

      
        # End time for plotting trajectory
        tpN = min(t + N, cl_traj.shape[-1] - 1)
        for i, color, ind in zip(range(2), ['g', 'b'], [0, 2]):
            # Plot new trajectory
            ax[1].plot(cl_traj[7 * i + 0, t:tpN], cl_traj[7 * i + 1, t:tpN], color + '-s', markersize=1)

            # Initial state marker
            ax[ind].plot(cl_traj[7 * i + 2, t], cl_traj[7 * i + 5, t], color + 'o', markersize=7, markeredgewidth=2, markeredgecolor='black')

            # Terminal state marker
            ax[ind].plot(cl_traj[7 * i + 2, tpN], cl_traj[7 * i + 5, tpN], color + '*', markersize=7, markeredgewidth=2, markeredgecolor='black')

            # Generate contour plot
            s_N_ego = th.linspace(float(np.floor(min(agent['state'].s for agent in cur_agent) - 2)),
                                float(np.ceil(max(agent['state'].s for agent in cur_agent) + N * 5 * 0.1 + 1)), n_points)
            v_N_ego = th.linspace(-1, 5, n_points)

            x_N_ego = th.cat([th.cat([s_N_ego[:, None], v_N_ego[k] * th.ones(n_points, 1), routes[i] * th.ones(n_points, 1)], dim=1)
                            for k in range(n_points)])

            pred_ind = [m for m in range(2) if m != i]
            x_other_vehicles = th.cat([th.tensor([float(cl_traj[7 * p + 2, tpN]), float(cl_traj[7 * p + 5, tpN]), routes[p]])
                                    .repeat(n_points, 1) for p in pred_ind], dim=1)

            x_N_joint = th.cat([x_other_vehicles.repeat(n_points, 1), x_N_ego - x_other_vehicles.repeat(n_points, 1)], dim=1)
            
            grid_x, grid_y = th.meshgrid(s_N_ego, v_N_ego, indexing='xy')
            if mode == 'GT':
                x_norm = sp.linalg.solve(np.real(sp.linalg.sqrtm(D['train'].feature_cov)), (x_N_joint - D['train'].feature_mean).T, assume_a='pos').T
                value = vf_copy(th.tensor(x_norm).to(device)).reshape(n_points, n_points).cpu().detach().numpy() * np.sqrt(D['train'].target_cov) + D['train'].target_mean
            else:
                value = v_mp(x_N_ego[:, 0], x_other_vehicles.repeat(n_points,1)[:,0]).reshape(n_points,n_points)

            contour = ax[ind].contourf(grid_x.numpy(), grid_y.numpy(), value, levels=40, cmap='viridis')
            
            # Add color bar only once per axis to prevent accumulation
            if colorbars[i] is not None:
                colorbars[i].remove()  # Remove previous colorbar if it exists
            colorbars[i] = fig.colorbar(contour, ax=ax[ind], orientation='vertical', fraction=0.046, pad=0.04)
            colorbars[i].set_label("[m]", fontsize=10)
            colorbars[i].ax.tick_params(labelsize=6)
            colorbars[i].ax.yaxis.set_major_formatter(FormatStrFormatter('%d'))
            ax[ind].set_ylabel('v [m/s]', labelpad=-2)
            ax[ind].set_xlabel('s [m]', labelpad=-2)
            ax[ind].set_title(f'Vehicle {i+1}', color=color,fontsize=10)
        if not os.path.isdir(f'{save_dir}N_{N}_{mode}_int_navigation/'):
            os.mkdir(f'{save_dir}N_{N}_{mode}_int_navigation/')
        # Save each frame as a PNG image
        fig.savefig(f'{save_dir}N_{N}_{mode}_int_navigation/t{t}.png', dpi=100)
    # Instead of using FuncAnimation, loop over each frame and call animate manually
    for t in range(cl_traj.shape[-1]):
        animate(t)

    def sortNumber(val):
        return int(val.split('.')[0][1:])

    # Specify the directory containing your PNG images
    image_folder = f'{save_dir}N_{N}_{mode}_int_navigation/'

    # Get a list of image filenames in the folder, sorted by filename
    images = [img for img in os.listdir(image_folder) if img.endswith(".png")]
    images.sort(key=sortNumber)  # Sort files to maintain the correct order
    # Read the first image to get the size
    first_image = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, layers = first_image.shape

    # Define the codec and create a VideoWriter object
    video = cv2.VideoWriter(image_folder + 'cl_int_navigation.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 10, (width, height))

    # Loop through all images and write them to the video
    for image in images:
        img_path = os.path.join(image_folder, image)
        frame = cv2.imread(img_path)
        video.write(frame)  # Add the frame to the video

    # Release the VideoWriter object
    video.release()

if __name__ == '__main__':
    mode = 'MP'
    index = 3 #3 (sc6), 1 (sc1), 6(sc3)
    routes_str = ['42','34']
    N = 10
    if mode == 'GT':
        #Import cl_traj.pkl
        with open('/home/mpc/it-sim/data/evaluation/game_mpc_seed2026_20241105_000445/game_mpc/evaluation/cl_traj.pkl', 'rb') as f:
            cl_traj = pickle.load(f)
        model_dir = '/home/mpc/it-sim/data/models/2cars_test/game_theoretic_NN/models/Value_NN_2car_1route_sc6_encoding_model_N20_10-26-2024_14-41-49.pt'
        dataset_dir = '/home/mpc/it-sim/data/models/2cars_test/game_theoretic_NN/dataset/itsim_2car_data_processed_tpN20_1route_sc6_encoding_absolute_categorical_10-26-2024_14-41-19.pkl'
    else:
        with open('/home/mpc/it-sim/data/evaluation/game_mpc_seed2026_20241105_000445/mpc/cl_traj.pkl', 'rb') as f:
            cl_traj = pickle.load(f)
            dataset_dir = None
            model_dir = None
    save_dir = '/home/mpc/it-sim/data/evaluation/game_mpc_seed2026_20241105_000445/mpc/'
    generate_video(cl_traj[index],mode,dataset_dir=dataset_dir,model_dir=model_dir,routes_str=routes_str,N=N,save_dir=save_dir)

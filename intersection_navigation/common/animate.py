import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from utils import transform_Rectangle, frenet2global
from IPython.display import HTML

def animate_trajectory(initial_agent,radius,traj,target,filename='animation.mp4',roads=(50,10),fillet_radius=2.8,preds_cl=None,vis_preds=False,s_N_guides=None,refs=None,routes=None):
    #[x,y,s,ey,epsi,v,psi]
    #traj = [x1,y1,s1,ey1,epsi1,v1,psi1,x2,y2,s2,ey2,epsi2,v2,psi2] (7*2)x151 whgere nx=7, M=2
    #nx *i +0 -> x (if i =0 -> x1, if i = 1 -> x2)
    #nx *i +1 -> x (if i =0 -> y1, if i = 1 -> y2)
    #nx * i +
    # Create a figure and axis
    fig, ax = plt.subplots()
    ax.set_aspect('equal')
    road_length, road_width = roads

    ax.set_xlim(0,road_length)
    ax.set_ylim(road_width/2-road_length/2,road_length/2+road_width/2)
    if target.shape[1] == 3:
        ax.plot(target[0,1],target[1,1],'rx')
        ax.plot(target[0,[0,2]],target[1,[0,2]],'gx')
    else:
        ax.plot(target[0,[0,1]],target[1,[0,1]],'gx')
        
    l_f = 4.47/2
    l_r = 4.47/2
    l = l_f + l_r
    w = 2
    nx = int(len(traj) / len(initial_agent))

    #convert s to x,y
    if s_N_guides:
        assert refs is not None and routes is not None

    def init():
        ax.add_patch(plt.Rectangle((0,0), road_length, road_width, color='gray'))
        ax.add_patch(plt.Rectangle((road_length/2-road_width/2,-road_length/2+road_width/2), road_width, road_length, color='gray'))
        artists = []
        for agent in initial_agent:
            color = 'green' if agent['type'] == 'CAV' else 'red'
            artists.append(ax.add_patch(plt.Rectangle(transform_Rectangle(agent['state'].x, agent['state'].y,w,l/2,0),angle=agent['state'].heading*180/np.pi,width=l,height=w,rotation_point='center',color=color)))
            artists.append(ax.arrow(agent['state'].x,agent['state'].y, 2*np.cos(agent['state'].heading), 2*np.sin(agent['state'].heading), head_width=0.5, head_length=0.5, fc='k', ec='k'))
            artists.append(ax.add_patch(plt.Circle((agent['state'].x,agent['state'].y), radius, color='blue', fill=False)))
        plt.xlabel('X [m]')
        plt.ylabel('Y [m]')
        
        return artists

    def animate(t):
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

        artists = []
        for i in range(len(initial_agent)):
            try:
                if isinstance(s_N_guides[i][t],list):
                    for s_ref in s_N_guides[i][t]:
                        pos = frenet2global(s_ref,refs[i],routes[i],road_length,road_width,radius,verbose=True if t > 150 else False)
                else:
                    pos = frenet2global(s_N_guides[i][t],refs[i],routes[i],road_length,road_width,radius,verbose=True if t > 150 else False)
                artists.append(ax.add_patch(plt.Circle((pos[0,0],pos[1,0]), radius/3, color='orange', fill=False))) #s_N_guides visualization
            except:
                pass
            color = 'green' if initial_agent[i]['type'] == 'CAV' else 'red'
            artists.append(ax.add_patch(plt.Rectangle(transform_Rectangle(traj[nx*i+0,t], traj[nx*i+1,t],w,l/2, 0),angle=traj[nx*i+6,t]*180/np.pi,rotation_point='center',width=l,height=w,color=color)))
            artists.append(ax.arrow(traj[nx*i+0,t],traj[nx*i+1,t], 2*np.cos(traj[nx*i+6,t]), 2*np.sin(traj[nx*i+6,t]), head_width=0.5, head_length=0.5, fc='k', ec='k'))
            artists.append(ax.add_patch(plt.Circle((traj[nx*i+0,t],traj[nx*i+1,t]), radius, color='blue', fill=False)))
        if preds_cl and vis_preds:
            try:
                for preds in preds_cl[t]:
                    for pred in preds:
                        artists.append(ax.add_patch(plt.Circle((pred.x,pred.y), radius, color='red', fill=False)))
            except:
                pass
        ax.set_xlim(0,road_length)
        ax.set_ylim(road_width/2-road_length/2,road_length/2+road_width/2)

        if target.shape[1] == 3:
            ax.plot(target[0,1],target[1,1],'rx')
            ax.plot(target[0,[0,2]],target[1,[0,2]],'gx')
        else:
            ax.plot(target[0,[0,1]],target[1,[0,1]],'gx')

        return artists

    # Show the animation
    ani = animation.FuncAnimation(fig, animate, init_func=init, repeat=False, blit=True, interval=100, frames=range(traj.shape[1]))
    ani.save(filename, writer='ffmpeg')
    plt.close(fig)

import matplotlib.pyplot as plt
import numpy as np
import os

def plot_trajectory():
    traj_file = 'CameraTrajectory.txt'
    if not os.path.exists(traj_file):
        print(f"File {traj_file} not found.")
        return
        
    data = np.loadtxt(traj_file)
    if len(data) == 0:
        print("Trajectory file is empty.")
        return
        
    x = data[:, 1]
    y = data[:, 2]
    z = data[:, 3]
    
    plt.figure(figsize=(10, 8))
    plt.plot(x, z, marker='o', markersize=2, linestyle='-', linewidth=1, color='b', label='Camera Trajectory (X-Z plane)')
    plt.plot(x[0], z[0], 'go', markersize=8, label='Start')
    plt.plot(x[-1], z[-1], 'ro', markersize=8, label='End')
    
    plt.xlabel('X (m)')
    plt.ylabel('Z (m)')
    plt.title('SP_SLAM3 Camera Trajectory on EuRoC Test Sequence')
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    
    out_path = '/home/nalin/.gemini/antigravity-ide/brain/b9e5284b-9b2a-4983-958c-ed55daa6eb9b/slam_trajectory_plot.jpg'
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"Trajectory plotted and saved to {out_path}")

if __name__ == '__main__':
    plot_trajectory()

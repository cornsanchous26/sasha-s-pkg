#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy

import numpy as np
import pickle as pkl
import matplotlib.pyplot as plt

import roslaunch

package = 'rqt_gui'
executable = 'rqt_gui'
node = roslaunch.core.Node(package, executable)

launch = roslaunch.scriptapi.ROSLaunch()
launch.start()

process = launch.launch(node)
print(process.is_alive())
process.stop()

model_path = 'tphsmm_preproc.pkl'
labels = ['t4', 't7', 'rl', 'rh', 'll', 'lh']
data = []
curr_pose = []

def get_arm_frames(arm, data, labels):
    # Only for immobile parameters
    if arm == 'left':
        arm_idx = 0
        arm_pts = ['t4', 't7', 'rl', 'rh']
    elif arm == 'right':
        arm_idx = 1
        arm_pts = ['t7', 't4', 'll', 'lh']
    else:
        raise ValueError(f'Invalid arm name {arm}')
    arm_pts = [f"{a}.pose.x" for a in arm_pts]
    if not set(arm_pts).issubset(labels) :
        print(f'ERROR: arm points not in labels. Your data should contain {arm_pts}')
        return None
    # left arm
    As = []
    bs = []
    pts = []
    for l in arm_pts:
        ptr = labels.index(l)
        has_nan = np.any(np.isnan(data[0, ptr:ptr+3]))
        if has_nan:
            print(f"{l} - Number of nans: {np.count_nonzero(np.isnan(data[0, ptr:ptr+3]))}")

        pts.append(data[0, ptr:ptr+3])
    for a in range(3):
        if a < 2:
            A = get_rotation_matrix(pts[a], pts[a+1], pts[a+2])
        else:
            A = get_rotation_matrix(pts[a-1], pts[a], pts[a+1])
        A = np.kron(np.eye(4), A)
        b = np.kron(np.array([1,1,0,0]),
                    pts[a+1])
        # [A, 0 ,0, 0] [b] xl
        # [0, A ,0, 0] [b] xr
        # [0 ,0, A, 0] [0] dxl
        # [0, 0 ,0, A] [0] dxr
        As.append(A)
        bs.append(b)

    return As, bs

def plan(ot_data: np.ndarray, pose_data: np.ndarray=None,
         horizon=200):
    labels = ['t4', 't7', 'rl', 'rh', 'll', 'lh']
    ltp = get_arm_frames('left', ot_data, labels)
    rtp = get_arm_frames('right', ot_data, labels)
    task_params = (np.stack(ltp[0] + rtp[0]),  # N_obs, xdx_dim, xdx_dim
                    np.stack(ltp[1] + rtp[1]))  # N_obs, xdx_dim
    if pose_data is None:
        pose_data = model.demos_tp[0][0, :]
    traj = model.generate(0, task_params, pose_data, horizon)

    traj /= 1000

    axshift = np.kron(np.eye(2), np.array([[1,0,0],[0,0,-1],[0,1,0]]))
    traj = (axshift @ traj[:, :6].T).T
    if traj[:, ::3].max() > 0.75 :
        print("X value over 0.75 ! Abort")
        return
    return traj


def align_shoulders_z(s1, s2):
    rot = np.eye(3)
    rot[2, 0::2] = (s2 - s1)[0::2]
    rot[2, 0::2] /= np.linalg.norm(rot[2, 0::2])
    rot[0, :] = np.cross(rot[1, :], rot[2, :])

    return rot

if __name__ == '__main__':
    
    #initialize a node
    rospy.init_node("plain_py_node")
    # Display the namespace of the node handle
    rospy.loginfo("PLAIN PY NODE] namespace of node = " + rospy.get_namespace());
    
    with open(model_path, 'rb') as f:
        model = pkl.load(f)

    demo_id = 3
    A, b = model.task_params[demo_id]
    A, b = A[:, None, :, :], b[:, None, :]
    x0 = model.demos_x[demo_id][0, :][None, :]

    traj, prod = model.generate(demo_id, (A, b), x0, 80)
    traj /= 1000

    fig = plt.figure(num=f'Optitrack data visualization')
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.set_xlim(-1, 1)  # Because the canvas is cleared, the range of the coordinate axis needs to be reset
    ax.set_ylim(0, 2)
    ax.set_zlim(-1, 1)

    ax.plot(traj[:, 0], traj[:, 1], traj[:, 2], 'r')
    ax.plot(traj[:, 3], traj[:, 4], traj[:, 5], 'b')

    A, b = model.task_params[demo_id]
    b /= 1000
    orig = np.array([0,0,0])
    rot = align_shoulders_z(b[0, :3], b[3, :3])
    xax = np.array([orig, orig + rot[0]])
    ax.plot(xax[:, 0], xax[:, 1], xax[:, 2], c='r')
    yax = np.array([orig, orig + rot[1]])
    ax.plot(yax[:, 0], yax[:, 1], yax[:, 2], c='g')
    zax = np.array([orig, orig + rot[2]])
    ax.plot(zax[:, 0], zax[:, 1], zax[:, 2], c='b')
    for i in range(2,6,3):
        ax.scatter(*b[i, :3], c='g')
        ax.text(*b[i, :3], f'{i}')
    plt.show()

    traj = traj[200:, :]

    crot = np.kron(np.eye(2), rot)
    corx = -traj[0, 3] * np.kron(np.ones(2), np.array([1, 0, 0]))

    axshift = np.kron(np.eye(2), np.array([[1,0,0],[0,0,-1],[0,1,0]]))
    a = (axshift @ crot @ traj[:, :6].T).T + corx
    a = np.repeat(a, 10, axis=0)
    fig = plt.figure(num=f'Optitrack data visualization')
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_zlim(0, 2)

    ax.plot(a[:, 0], a[:, 1], a[:, 2], 'r')
    ax.plot(a[:, 3], a[:, 4], a[:, 5], 'b')
    plt.show()

    out_traj = np.zeros((2, a.shape[0], 3))
    out_traj[0] = a[:, :3]
    out_traj[1] = a[:, 3:]

    np.save('tphsmm_traj2_001.npy', out_traj)
    
    rate = rospy.Rate(80) # 80hz
    while not rospy.is_shutdown():
        out_traj = np.zeros((2, a.shape[0], 3))
        out_traj[0] = a[:, :3]
        out_traj[1] = a[:, 3:]
        pub.publish(out_traj)
        rate.sleep()

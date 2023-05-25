import os
import numpy as np
from tqdm import tqdm
from extensions.chamfer_dist import ChamferDistanceL1, ChamferDistanceL2

# want to create the numpy files of a joined trajectory
    # idx for main dataset 
    # other version saved just for that target shape

def join_trajectory(states, actions, rewards, discount=0.99):
    traj_length = states.shape[0]
    # I can vectorize this for all dataset as once,
    # but better to be safe and do it once and slow and right (and cache it)
    discounts = (discount ** np.arange(traj_length))

    values = np.zeros_like(rewards)
    for t in range(traj_length):
        # discounted return-to-go from state s_t:
        # r_{t+1} + y * r_{t+2} + y^2 * r_{t+3} + ...
        # .T as rewards of shape [len, 1], see https://github.com/Howuhh/faster-trajectory-transformer/issues/9
        values[t] = (rewards[t + 1:].T * discounts[:-t - 1]).sum()

    joined_transition = np.concatenate([states, actions, rewards, values], axis=-1)

    return joined_transition

def get_reward(target, pcl):
    """
    get reward from the diffrence between the point clouds
    """
    pass



def sliding_window(raw_trajectories, seq_len):
    """
    raw_trajectories is a dictionary with all the raw trajectories of varying length
        raw_trajectories[i]["states"]
        raw_trajectories[i]["actions"]
        raw_trajectories[i]["target_shape"]

    we want to iterate through these raw trajectories with seq_len, reorder states to have
    goal as first state in list, and calculate the reward w.r.t difference with target
        - sliding window until only one step trajectory? (will automatically pad the rest)
        
    Need to iterate through such that we only load a few pcls at a time for memory issues
        - save each as trajectory_int(idx) <-- such that when we load each datapoint, just load
          the saved numpy array and do the final processing
    """

    trajectories = {}

    return trajectories

def sliding_window_idxs(window_size, traj_len):
    """
    """
    idxs = []
    for i in range(traj_len - 2):
        print("\ni: ", i)
        print("i + window: ", i + window_size)
        if (i + window_size) <= traj_len:
            # idxs.append([i,(i+window_size)]) # TODO: turn this into a list of all the indices
            idxs.append(np.arange(i, i+window_size, 1))
        else:
            # idxs.append([i, traj_len])
            idxs.append(np.arrange(i, traj_len, 1))
    return idxs

def recon_center_loss(self, pcl, gt):
    # TODO: convert pcl and gt to cuda
    loss_func_cdl1 = ChamferDistanceL1().cuda()
    loss_func_cdl2 = ChamferDistanceL2().cuda()
    l1_loss = loss_func_cdl1(pcl, gt)
    l2_loss = loss_func_cdl2(pcl, gt)
    recon_loss = l1_loss + l2_loss
    # TODO: convert recon_loss to cpu
    return recon_loss

def cd_to_reward(cd):
    """
    convert chamfer distance to reward signal
    """
    # want reward to be between 1 and -1000
    if cd >= 0.25:
        reward = -1000
    else:
        reward = np.exp(-8 * cd) * (1 - (-1000)) + (-1000)
    return reward

def get_state_rewards(states):
    goal = states[-1]

    s = []
    r = []
    for i in range(len(states) - 1):
        state = states[i]
        cd = recon_center_loss(state, goal)
        print("\nCD: ", cd)
        reward = cd_to_reward(cd)
        assert False
        s.append(state)
        r.append(reward)
    return s, r

if __name__=="__main__":
    shapes = ['cylinder', 'Incomplete_X', 'line', 'pyramid', 'square', 'T', 'X']
    n_trajs = [5, 4, 5, 4, 2, 5, 5]
    lens_dict = {
        'cylinder': [9, 9, 5, 9, 10],
        'Incomplete_X': [5, 5, 8, 4],
        'line': [4, 7, 5, 4, 4],
        'pyramid': [17, 15, 12, 6],
        'square': [7],
        'T': [5, 5, 5, 5, 6],
        'X': [7, 7, 8, 4, 7]
    }

    window_size = 5
    global_idx = 0
    for s in range(len(shapes)):
        shape = shapes[s]
        shape_idx = 0

        for t in range(n_trajs[s]):
            traj_len = lens_dict[shape][t]
            idxs = sliding_window_idxs(window_size, traj_len)

            for i in range(60):
                traj = t*60 + i + 1

                for j in range(len(idxs)):
                    states = []
                    actions = []

                    for idx in idxs[j]:
                        path = '/home/alison/Clay_Data/Partially_Processed/Clay_Demo_Trajectories/' + shape + '/Trajectory' + str(traj)
                        s = np.load(path + '/State' + str(idx) + '.npy')
                        a = np.load(path + '/Action' + str(idx) + '.npy')

                        # normalize both arrays of actions
                        a_mins = np.array([0.55, -0.035, 0.19, -50, -50, -90, 0.005])
                        a_maxs = np.array([0.63, 0.035, 0.25, 50, 50, 90, 0.05])
                        norm_a = (a - a_mins) / (a_maxs - a_mins)

                        # a_mins5d = np.array([0.55, -0.035, 0.19, -90, 0.005])
                        # a_maxs5d = np.array([0.63, 0.035, 0.25, 90, 0.05])
                        # a_5d = np.hstack((a[0:3], a[5:7]))
                        # norm_a = (a_5d - a_mins5d) / (a_maxs5d - a_mins5d)

                        states.append(s)
                        actions.append(norm_a)

                    states, rewards = get_state_rewards(states)
                    joined = join_trajectory(states, actions, rewards)
                    global_save_path = '/home/alison/Clay_Data/Fully_Processed/Clay_Demo_Trajectories/' + shape + '/Trajectory' + str(shape_idx) + '.npy'
                    shape_save_path = '/home/alison/Clay_Data/Fully_Processed/Clay_Demo_Trajectories/Trajectory' + str(global_idx) + '.npy'
                    np.save(global_save_path, joined)
                    np.save(shape_save_path, joined)

                    global_idx += 1
                    shape_idx += 1
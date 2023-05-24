import os
import numpy as np
from tqdm import tqdm

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
            for i in range(60):
                traj = t*60 + i + 1

                # import the actions and normalize and get the 5D actions

                # go through the trajectory to generate the sliding window
                    # reorder the states such that the final state is at the front of the list
                    # calculate the reward for s0 -> st-1 w.r.t st
                    # get the join_trajectory given the window states, actions, rewards

                    # save the join_trajectory to global_idx and shape_idx trajectory respectively
                
                    # global_idx += 1
                    # shape_idx += 1
import numpy as np

class CountBasedExploration:
    def __init__(self, num_states):
        self.num_states = num_states
        self.nrows, self.ncols = np.sqrt(num_states), np.sqrt(num_states)
        self.cache = {i:0 for i in range(num_states)}
        self.total_visits = 0

    def discretize_obs(self, obs):
        row = int(self.nrows*obs[0])
        col = int(self.ncols*obs[1])
        cache_idx = row*self.ncols + col
        return cache_idx

    def update(self, ob_no):
        state_counts = []
        for obs in ob_no:
            cache_idx = self.discretize_obs(obs)
            if cache_idx in self.cache:
                self.cache[cache_idx] += 1
            else:
                self.cache[cache_idx] = 1
            state_counts.append(self.cache[cache_idx])
            self.total_visits += 1
        return 0.0 
        
    def forward_np(self, ob_no):
        state_counts = []
        for obs in ob_no:
            cache_idx = self.discretize_obs(obs)
            state_counts.append(self.cache[cache_idx])
        bonus = 1/np.sqrt(np.array(state_counts))
        return bonus

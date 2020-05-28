import numpy as np
from task import Task

class PD_Agent():
    def __init__(self, task):
        # Task (environment) information
        self.task = task
        self.state_size = task.state_size
        self.action_size = task.action_size
        self.action_low = task.action_low
        self.action_high = task.action_high
        self.action_range = self.action_high - self.action_low

        self.wp = np.random.normal(
            size=1,  # weights for simple linear policy: state_space x action_space
            scale=1) # start producing actions in a decent range
        self.wd = np.random.normal(
            size=1,  # weights for simple linear policy: state_space x action_space
            scale=1) # start producing actions in a decent range
#         self.bias = np.random.normal(
#             size=(self.action_size),  # weights for simple linear policy: state_space x action_space
#             scale=(self.action_range / 2)) # start producing actions in a decent range
        self.bias = np.array([self.action_range / 2]*4)


        # Score tracker and learning parameters
        self.best_wp = None
        self.best_wd = None
        self.best_bias = None
        self.best_score = -np.inf
#         self.noise_scale = 0.1
        self.noise_scale = 1
        self.noise_scale_trials = 0
        self.best_param_found = False

        # Episode variables
        self.reset_episode()

    def reset_episode(self):
        self.total_reward = 0.0
        self.count = 0
        state = self.task.reset()
        return state

    def step(self, reward, done):
        # Save experience / reward
        self.total_reward += reward
        self.count += 1

        # Learn, if at end of episode
        if done:
            self.learn()

    def act(self, state):
        # Choose action based on given state and policy
        dist = state[2] - self.task.target_pos[2]
        vel = self.task.sim.v[2]
#         action = np.dot(dist, self.w)  + self.bias # simple linear policy        
        action = self.bias + dist*self.wp + vel*self.wd # simple linear policy
#         print(action)
        return action

    def learn(self):
        # Learn by random policy search, using a reward-based score
#         self.score = self.total_reward / float(self.count) if self.count else 0.0
        self.score = self.total_reward
        self.noise_scale_trials += 1
        if self.score > self.best_score:
            self.best_score = self.score
            self.best_wp = self.wp
            self.best_wd = self.wd
            self.best_bias = self.bias
            self.best_param_found = True
            
        else:
            self.wp = self.best_wp
            self.wd = self.best_wd
            self.bias = self.best_bias
            
#         if self.noise_scale_trials >= 100:
# #             print('\n',self.bias)
#             if self.best_param_found:
#                 self.noise_scale = max(0.5 * self.noise_scale, 0.25)
# #                 self.noise_scale = 1
#             else:
#                 self.noise_scale = min(2.0 * self.noise_scale, 2)
# #                 self.noise_scale = 1
                
            self.noise_scale_trials = 0
            self.best_param_found = False
            
        self.wp = self.wp + self.noise_scale * np.random.normal(size=self.wp.shape)  # equal noise in all directions
        self.wd = self.wd + self.noise_scale * np.random.normal(size=self.wd.shape)  # equal noise in all directions
#         self.bias = self.bias + self.noise_scale * np.random.normal(size=self.bias.shape)  # equal noise in all directions
        self.bias = self.bias + np.array([self.noise_scale * np.random.normal()]*4)  # equal noise in all directions
#         print('\n',self.bias)
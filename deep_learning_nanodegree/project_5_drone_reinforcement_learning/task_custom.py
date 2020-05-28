import numpy as np
from physics_sim import PhysicsSim
from scipy.spatial.distance import euclidean as eucl_dist

class Task():
    """Task (environment) that defines the goal and provides feedback to the agent."""
    def __init__(self, init_pose=None, init_velocities=None, 
        init_angle_velocities=None, runtime=5., target_pos=None):
        """Initialize a Task object.
        Params
        ======
            init_pose: initial position of the quadcopter in (x,y,z) dimensions and the Euler angles
            init_velocities: initial velocity of the quadcopter in (x,y,z) dimensions
            init_angle_velocities: initial radians/second for each of the three Euler angles
            runtime: time limit for each episode
            target_pos: target/goal (x,y,z) position for the agent
        """
        # Simulation
        self.sim = PhysicsSim(init_pose, init_velocities, init_angle_velocities, runtime)
        self.action_repeat = 3

#         self.state_size = self.action_repeat * 6
        self.state_size = 2
        self.action_low = 350
        self.action_high = 450
#         self.action_size = 4
        self.action_size = 1

        # Maximum distance before ending the episode
        self.max_dist = 9
        self.max_vel = 6

        # Goal
#         self.target_pos = target_pos if target_pos is not None else np.array([0., 0., 10.]) 
        self.target_pos = target_pos if target_pos is not None else np.array([10.]) 
        
        self.in_colision = False
        self.out_of_bounds = False

    def get_reward(self, done):
        """Uses current pose of sim to return reward."""
        z = self.sim.pose[2]
        v = self.sim.v[2]
        
        fix_rwrd = 0.4
    
        dist_norm = eucl_dist([z],self.target_pos)/self.max_dist
        if dist_norm >=1:
            if not self.out_of_bounds:
                self.out_of_bounds = True
                dist_rwrd = -1.4
            else:
                dist_rwrd = 0
        else:
            self.out_of_bounds = False
            dist_rwrd = 1-(dist_norm)**0.4
        
        vel_norm = max(abs(v)/self.max_vel,0.01)
        vel_discount = max((1-vel_norm),0)**(1/max(dist_norm,0.01)**0.5)
    
#         if (self.sim.time < self.sim.runtime) and done:            
#             if not self.in_colision:
#                 self.in_colision = True
#                 colis_rwrd = -100
#             else:
#                 colis_rwrd = 0
#         else:
#             self.in_colision = False
#             colis_rwrd = 0

#         dist_floor = eucl_dist([self.sim.pose[2]],[0])
#         dist_floor_rwrd = 1*(np.tanh(dist_floor*0.1))
        
#         act_homog_rwrd = self.sim.
            
#         reward = fix_rwrd+dist_rwrd+dist_floor_rwrd
        reward = fix_rwrd + dist_rwrd*vel_discount
#         print('-------------------')
#         print('z:',z)
#         print('v:',v)
#         print('dist_norm:',dist_norm)
#         print('vel_norm:',vel_norm)
#         print('dist_rwrd:',dist_rwrd)
#         print('vel_discount:',vel_discount)
#         print('reward:',reward)
        
        return reward

    def step(self, rotor_speeds):
        """Uses action to obtain next state, reward, done."""
        reward = 0
        pose_all = []
        for _ in range(self.action_repeat):
#             done = self.sim.next_timestep(rotor_speeds) # update the sim pose and velocities
            done = self.sim.next_timestep(rotor_speeds*4) # update the sim pose and velocities
            dist = eucl_dist([self.sim.pose[2]],self.target_pos)
            done = done or (dist >= self.max_dist)
            reward += self.get_reward(done)/3
            pose_all.append(self.sim.pose)
#         next_state = np.concatenate(pose_all)
        next_state = [self.sim.pose[2], self.sim.v[2]]
        return next_state, reward, done

    def reset(self):
        """Reset the sim to start a new episode."""
        self.sim.reset()
#         state = np.concatenate([self.sim.pose] * self.action_repeat) 
        state = [self.sim.pose[2], self.sim.v[2]]
        return state
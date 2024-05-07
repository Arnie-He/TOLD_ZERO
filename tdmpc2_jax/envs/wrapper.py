import gymnasium as gym
from gymnasium import spaces
from ding.envs import BaseEnv

class DingToGymWrapper(gym.Env):
    """
    A wrapper to convert a Ding environment into an OpenAI Gym environment.
    """
    metadata = {'render.modes': ['human', 'rgb_array']}

    def __init__(self, ding_env: BaseEnv):
        self.ding_env = ding_env
        ding_env.reset()
        self.observation_space = ding_env.observation_space
        self.action_space = ding_env.action_space
        self.reward_range = ding_env._reward_space

    def reset(self):
        """
        Resets the environment and returns the initial observation.
        """
        obs = self.ding_env.reset()
        return obs['observation']

    def step(self, action):
        """
        Steps the environment with the given action.
        Arguments:
            - action: The action to be performed in the environment.
        Returns:
            - obs: New observation after the action.
            - reward: Reward obtained from the action.
            - done: Whether the episode has ended.
            - info: Additional information about the step.
        """
        timestep = self.ding_env.step(action)
        obs = timestep.obs['observation']
        reward = timestep.reward  # Assuming reward is wrapped in an array
        done = timestep.done
        info = timestep.info
        # TODO: Truncated
        truncated = False
        return obs, reward, done, truncated, info

    def render(self):
        """
        Renders the environment. The actual rendering must be handled by the Ding environment if it supports rendering.
        """
        return self.ding_env._env.render()

    def close(self):
        """
        Clean up the environment's resources.
        """
        return self.ding_env._env.close()

    def seed(self, seed=None):
        """
        Sets the seed for this env's random number generator(s).
        """
        super().seed(seed)
        return self.ding_env._env.seed(seed)

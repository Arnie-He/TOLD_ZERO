import gymnasium as gym
from gymnasium.spaces import MultiBinary
import numpy as np
from itertools import product
from gymnasium.core import ObsType, WrapperObsType
from gymnasium.spaces import Box
from gymnasium import Env, ObservationWrapper

class RescaleActions(gym.Wrapper):
  """
  Rescale actions from the range [-1, 1] to the environment action space ranges.
  """

  def __init__(self, env: gym.Env):
    super().__init__(env)
    self.action_scale = (env.action_space.high - env.action_space.low) / 2
    self.action_bias = (env.action_space.high + env.action_space.low) / 2

  def step(self, action):
    action = action * self.action_scale + self.action_bias
    return self.env.step(action)
  
class OneHotAction(gym.Wrapper):
    """
    Wrapper to convert actions to one-hot vectors. Ensures sampled actions are valid one-hot vectors.
    """

    def __init__(self, env: gym.Env):
        super().__init__(env)
        assert isinstance(env.action_space, gym.spaces.Discrete), "Action space must be discrete"
        self.num_actions = env.action_space.n

    def action_sample(self):
        # Custom method to sample a valid one-hot vector
        action_idx = self.env.action_space.sample()  # Sample a scalar action
        one_hot_action = np.zeros(self.num_actions, dtype=int)
        one_hot_action[action_idx] = 1
        return one_hot_action

    def step(self, action):
        # Assuming the action input here is already one-hot encoded
        # Convert one-hot vector back to scalar action
        scalar_action = np.argmax(action)
        return self.env.step(scalar_action)
    
class FourierFeatures(ObservationWrapper):

    def __init__(self, env: Env,
                 min_vals, max_vals,
                 order: int = 2):
        super().__init__(env)
        self.order = order
        terms = product(range(order + 1), repeat=self.observation_space.shape[0])

        # Removing first iterate because it corresponds to the constant bias
        self.multipliers = np.array([list(map(int, x)) for x in terms][1:])
        self.min_vals = min_vals
        self.max_vals = max_vals
        self.observation_space = Box(0, 1, shape=(self.multipliers.shape[0], ))

    def scale(self, values):
        shifted = values - self.min_vals
        if self.max_vals is None:
          return shifted

        return shifted / (self.max_vals - self.min_vals)

    def observation(self, observation: ObsType) -> WrapperObsType:
        scaled = self.scale(observation)
        return np.cos(np.pi * self.multipliers @ scaled)


if __name__ == '__main__':
  # test one-hot action wrapper
  env = gym.make('CartPole-v1')
  env = OneHotAction(env)
  action = env.action_sample()
  print(action)
  


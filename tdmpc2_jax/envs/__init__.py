from atari_lightzero_env import AtariEnvLightZero
from mujoco_lightzero_env import MujocoEnvLightZero
from cartpole_lightzero_env import CartPoleEnv
from mtcar_lightzero_env import MountainCarEnv
from pendulum_lightzero_env import PendulumEnv
from bipedalwalker_env import BipedalWalkerEnv
from lunarlander_env import LunarLanderEnv
from wrapper import DingToGymWrapper

# atari options={'PongNoFrameskip-v4', 'QbertNoFrameskip-v4', 'MsPacmanNoFrameskip-v4', 
# 'SpaceInvadersNoFrameskip-v4', 'BreakoutNoFrameskip-v4', ...}

SUPPORTED_LIST = [
    'atari_lightzero', 'cartpole_lightzero',
    'mountain_car_lightzero', 'pendulum_lightzero',
    # 'bipedalwalker', 
    # 'lunarlander',
    # 'mujoco_lightzero'
]

# Mapping of environment IDs to their corresponding classes
env_list = {
    'atari_lightzero': AtariEnvLightZero,
    # 'mujoco_lightzero': MujocoEnvLightZero,
    'cartpole_lightzero': CartPoleEnv,
    'mountain_car_lightzero': MountainCarEnv,
    'pendulum_lightzero': PendulumEnv,
    # 'bipedalwalker': BipedalWalkerEnv,
    # 'lunarlander': LunarLanderEnv
}

# Function to create and wrap environments
def make_env(env_id, cfg):
    if env_id in SUPPORTED_LIST:
        print(env_id)
        env = env_list[env_id](cfg)
        return DingToGymWrapper(env)
    else:
        raise NotImplementedError(f"Environment {env_id} not supported yet.")
    
# Main execution block
# TODO: mujoco is not supported on macOS
if __name__ == '__main__':
    for env_id in SUPPORTED_LIST:
        env = make_env(env_id, {})
        observation = env.reset()
        action = env.action_space.sample()
        next_observation, reward, done, truncated, info = env.step(action)
        env.close()
        print(f'env_id: {env_id}')
        print(f'observation space: {env.observation_space}')
        print(f'action space: {env.action_space}')
        print(f'reward range: {env.reward_range}')
        print(f'observation: {observation}')
        print(f'action: {action}')
        print(f'next_observation: {next_observation}')
        print(f'reward: {reward}')
        print(f'done: {done}')
        print(f'truncated: {truncated}')
        print(f'info: {info}')

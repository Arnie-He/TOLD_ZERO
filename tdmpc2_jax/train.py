import gymnasium as gym
import numpy as np
import jax
import flax.linen as nn
import tqdm
from tdmpc2_jax.networks import NormedLinear
from tdmpc2_jax.common.activations import mish, simnorm
from functools import partial
from tdmpc2_jax import WorldModel, TDMPC2
from tdmpc2_jax.data import EpisodicReplayBuffer
import os
import hydra
from tdmpc2_jax.wrappers.action_scale import RescaleActions, OneHotAction, FourierFeatures
import jax.numpy as jnp
import math
from mcts import MinMax

os.environ['PYDEVD_DISABLE_FILE_VALIDATION'] = '1'
CARTPOLE_MIN_VALS = np.array([-2.4, -5., -math.pi/12., -math.pi*2.])
CARTPOLE_MAX_VALS = np.array([2.4, 5., math.pi/12., math.pi*2.])


@hydra.main(config_name='config', config_path='.')
def train(cfg: dict):
  seed = cfg['seed']
  max_steps = cfg['max_steps']
  encoder_config = cfg['encoder']
  model_config = cfg['world_model']
  tdmpc_config = cfg['tdmpc2']

  T = 500
  seed_steps = 0
  # HalfCheetah-v4
  env = gym.make("CartPole-v1")
  # check if the environment has a discrete action space
  if not isinstance(env.action_space, gym.spaces.Discrete):
    env = RescaleActions(env)
  # env = RepeatAction(env, repeat=2)
  env = gym.wrappers.RecordEpisodeStatistics(env)
  # env = FourierFeatures(env, CARTPOLE_MIN_VALS, CARTPOLE_MAX_VALS)
  env = OneHotAction(env)
  env.action_space.seed(seed)
  env.observation_space.seed(seed)
  np.random.seed(seed)
  rng = jax.random.PRNGKey(seed)

  dtype = jnp.dtype(model_config['dtype'])
  rng, model_key = jax.random.split(rng, 2)
  encoder = nn.Sequential(
      # [
      #   nn.Conv2D(16, (8, 8), (4, 4), 'SAME'),
      #   nn.relu,
      #   nn.Conv2D(32, (4, 4), (2, 2), 'SAME'),
      #   nn.relu,
      #   nn.Flatten(),
      # ] + 
      [
          NormedLinear(encoder_config['encoder_dim'],
                       activation=mish, dtype=dtype)
          for _ in range(encoder_config['num_encoder_layers']-1)
      ] +
      [
          NormedLinear(
              model_config['latent_dim'],
              activation=partial(
                  simnorm, simplex_dim=model_config['simnorm_dim']),
              dtype=dtype)
      ])

  model = WorldModel.create(
      observation_space=env.observation_space,
      action_space=env.action_space,
      encoder_module=encoder,
      **model_config,
      key=model_key)
  agent = TDMPC2.create(world_model=model, **tdmpc_config)

  replay_buffer = EpisodicReplayBuffer(
      capacity=max_steps,
      dummy_input=dict(
          observation=env.observation_space.sample(),
          action=env.action_sample(),
          reward=1.0,
          next_observation=env.observation_space.sample(),
          terminated=True,
          truncated=True,
      ),
      seed=seed,
      respect_episode_boundaries=False)

  # Training loop
  ep_info = {}
  ep_count = 0
  prev_plan = None
  minmax = MinMax()
  observation, _ = env.reset(seed=seed)
  script_dir = os.path.dirname(os.path.abspath(__file__))
  log_file_path = os.path.join(script_dir, "log_result_reward.txt")
  returns = []
  losses = []
  with open(log_file_path, "a") as log_file:
    log_file.write("Step,Loss,Return,Return Std,Loss Std\n")
    for i in tqdm.tqdm(range(max_steps), smoothing=0.1):
      if i <= seed_steps:
        action = env.action_sample()
      else:
        rng, action_key = jax.random.split(rng)
        action, prev_plan = agent.act(
            observation, minmax, prev_plan, train=True, key=action_key)
      # print(f"Action: {action}")

      next_observation, reward, terminated, truncated, info = env.step(action)
      replay_buffer.insert(dict(
          observation=observation,
          action=action,
          reward=reward,
          next_observation=next_observation,
          terminated=terminated,
          truncated=truncated),
          episode_index=ep_count)
      observation = next_observation

      if terminated or truncated:
        observation, _ = env.reset()
        prev_plan = None

        r = info['episode']['r']
        l = info['episode']['l']
        returns.append(r[0])
        return_std = np.std(returns)
        print(f"Episode: r = {r}, l = {l}")
        # Print the mean of all episode info
        if len(ep_info) > 0:
          print(jax.tree_map(lambda x: np.mean(x), ep_info))
          # log episode info to file
          avg_loss = np.mean(ep_info['total_loss'])
          losses = ep_info['total_loss']
          loss_std = np.std(losses) 
          log_file.write(f"{i}, {avg_loss}, {r[0]}, {return_std}, {loss_std}\n") 
          log_file.flush() 
        ep_count += 1

      if i >= seed_steps:
        if i == seed_steps:
          print('Pre-training on seed data...')
          num_updates = seed_steps
        else:
          num_updates = 1

        rng, *update_keys = jax.random.split(rng, num_updates+1)
        for j in range(num_updates):
          # print(f"Update {j}...")
          batch = replay_buffer.sample(agent.batch_size, agent.horizon)
          # batch['action] should be horizon x batch_size x action_dim, but it is batch_size x horizon
          agent, train_info = agent.update(
              observations=batch['observation'],
              actions=batch['action'],
              rewards=batch['reward'],
              next_observations=batch['next_observation'],
              terminated=batch['terminated'],
              truncated=batch['truncated'],
              key=update_keys[j])
          # Append all episode info
          if len(ep_info) == 0:
            ep_info = train_info

          if i % 100 == 0:
            ep_info = jax.tree_map(
                lambda x, y: np.append(np.array(x), np.array(y)), ep_info, train_info)


if __name__ == '__main__':
  train()
import gym
import gym_mupen64plus
import ray
import time
from argparse import ArgumentParser
from ray import tune
from ray.tune.registry import register_env
from ray.rllib.agents.ppo import PPOTrainer
from ray.rllib.env.atari_wrappers import (MonitorEnv,
                                          NoopResetEnv,
                                          WarpFrame,
                                          FrameStack)


def parse_args():
    parser = ArgumentParser(description='Train an agent to beat N64 Games')
    parser.add_argument('--checkpoint', help='Specify an existing checkpoint '
                        'which can be used to restore progress from a previous'
                        ' training run.')
    parser.add_argument('--environment', help='The Super Mario Bros level to '
                        'train on.', type=str,
                        default='Mario-Kart-Discrete-Luigi-Raceway-v0')
    parser.add_argument('--gpus', help='Number of GPUs to include in the '
                        'cluster.', type=float, default=0)
    parser.add_argument('--workers', help='Number of workers to launch on the '
                        'cluster. Hint: Must be less than the number of CPU '
                        'cores available.', type=int, default=4)
    return parser.parse_args()


def env_creator(env_name, config):
    import gym_mupen64plus
    env = gym.make(env_name)
    env = MonitorEnv(env)
    env = NoopResetEnv(env, noop_max=30)
    env = WarpFrame(env, 84)
    env = FrameStack(env, 4)
    return env


def track_name(env):
    env = env.strip('Mario-Kart-')
    env = env.strip('Discrete-')
    env = env.strip('-v0')
    env = env.lower()
    env = env.replace('-', '_')
    env = 'mario_kart_' + env
    return env


def main():
    def env_creator_lambda(env_config):
        return env_creator(args.environment,
                           config)

    args = parse_args()
    env_name = track_name(args.environment)
    config = {
        'env': env_name,
        'framework': 'torch',
        'lr': 0.0003,
        'lambda': 0.95,
        'gamma': 0.99,
        'sgd_minibatch_size': 256,
        'vf_clip_param': 1000,
        'clip_param': 0.2,
        'num_sgd_iter': 10,
        'num_workers': args.workers,
        'num_envs_per_worker': 1,
        'num_gpus': args.gpus,
        'model': {
            'fcnet_hiddens': [512, 512]
        }
    }
    ray.init(address='head:6379', _redis_password='5241590000000000')

    register_env(env_name, env_creator_lambda)
    time.sleep(5)
    tune.run('PPO',
             stop={'training_iteration': 1000000},
             config=config,
             restore=args.checkpoint,
             checkpoint_freq=25,
             reuse_actors=True)


if __name__ == "__main__":
    main()

import gymnasium as gym
import torch
import numpy as np
from skrl.agents.torch.dqn import DQN
from skrl.agents.torch.ppo import PPO

# SKRL API: use model_instantiators directly
from skrl.utils.model_instantiators.torch import deterministic_model, categorical_model

# SKRL API change: wrap_env may be in different modules
try:
    from skrl.envs.wrappers import wrap_env
except ImportError:
    try:
        from skrl.envs.wrappers.torch import wrap_env
    except ImportError:
        from skrl.envs.torch import wrap_env
from skrl.memories.torch import RandomMemory  # used by DQN; PPO ignores
from skrl.trainers.torch import SequentialTrainer

# Select device (CPU or GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_dqn_models(env):
    """
    Create and return DQN models: Q-network and target Q-network.
    """
    network = [
        {
            "name": "hidden",
            "input": "OBSERVATIONS",
            "layers": [64, 64],
            "activations": ["relu"],
        }
    ]
    q_network = deterministic_model(
        observation_space=env.observation_space,
        action_space=env.action_space,
        device=device,
        network=network,
        output="ACTIONS",
    )
    target_q_network = deterministic_model(
        observation_space=env.observation_space,
        action_space=env.action_space,
        device=device,
        network=network,
        output="ACTIONS",
    )
    # Ensure both networks are on the selected device
    q_network.to(device)
    target_q_network.to(device)
    return {"q_network": q_network, "target_q_network": target_q_network}


def build_ppo_models(env):
    """
    Create and return PPO models: categorical policy and value function.
    """
    network = [
        {
            "name": "shared",
            "input": "OBSERVATIONS",
            "layers": [64, 64],
            "activations": ["relu"],
        }
    ]
    policy = categorical_model(
        observation_space=env.observation_space,
        action_space=env.action_space,
        device=device,
        network=network,
        output="ACTIONS",
    )
    value_function = deterministic_model(
        observation_space=env.observation_space,
        action_space=env.action_space,
        device=device,
        network=network,
        output="ONE",
    )
    # Ensure networks are on the selected device
    policy.to(device)
    value_function.to(device)
    return {"policy": policy, "value_function": value_function}


def train_cartpole(algorithm_name, timesteps, seed):
    """
    Train CartPole-v1 using specified algorithm (DQN or PPO), timesteps, and seed.
    """
    # reproducibility
    np.random.seed(seed)
    torch.manual_seed(seed)

    # environment setup
    env = gym.make("CartPole-v1")
    env = wrap_env(env)

    # memory (DQN uses it; PPO ignores)
    memory = RandomMemory(
        memory_size=50000, num_envs=env.num_envs, device=device, replacement=False
    )

    if algorithm_name == "DQN":
        models = build_dqn_models(env)
        cfg = {"learning_starts": 1000}
        agent = DQN(
            models=models,
            memory=memory,
            cfg=cfg,
            observation_space=env.observation_space,
            action_space=env.action_space,
            device=device,
        )
    else:
        models = build_ppo_models(env)
        cfg = {"learning_starts": 1000}
        agent = PPO(
            models=models,
            memory=memory,
            cfg=cfg,
            observation_space=env.observation_space,
            action_space=env.action_space,
            device=device,
        )

    # trainer and training
    trainer_cfg = {"timesteps": timesteps, "headless": True}
    trainer = SequentialTrainer(cfg=trainer_cfg, env=env, agents=[agent])
    print(
        f"Training {algorithm_name} for {timesteps} timesteps (seed={seed}) on {device}"
    )
    trainer.train()


if __name__ == "__main__":
    seeds = [0, 42, 123]
    timesteps_list = [500_000, 1_000_000, 2_000_000]
    for algo in ["DQN", "PPO"]:
        for ts in timesteps_list:
            for sd in seeds:
                train_cartpole(algo, ts, sd)

#!/usr/bin/env python3
import gymnasium as gym
import torch
import numpy as np
from skrl.agents.torch.ppo import PPO
from skrl.models.torch import Model, CategoricalMixin
from skrl.utils.model_instantiators.torch import deterministic_model
from skrl.envs.wrappers.torch import wrap_env
from skrl.memories.torch import RandomMemory
from skrl.trainers.torch import SequentialTrainer
from skrl.utils import set_seed

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class StablePolicy(CategoricalMixin, Model):
    def __init__(self, observation_space, action_space, device, **kwargs):
        Model.__init__(self, observation_space, action_space, device)
        CategoricalMixin.__init__(self, device)

        self.net = torch.nn.Sequential(
            torch.nn.Linear(observation_space.shape[0], 64),
            torch.nn.Tanh(),
            torch.nn.Linear(64, 64),
            torch.nn.Tanh(),
            torch.nn.Linear(64, action_space.n)
        )

        # Initialize weights
        for layer in self.net:
            if isinstance(layer, torch.nn.Linear):
                torch.nn.init.orthogonal_(layer.weight, gain=0.1)
                torch.nn.init.constant_(layer.bias, 0.0)

    def compute(self, inputs, role):
        # Use correct input key: "states" instead of "observations"
        x = inputs["states"]
        logits = self.net(x)
        
        if torch.isnan(logits).any() or torch.isinf(logits).any():
            print("Invalid logits detected, resetting to zero")
            logits = torch.zeros_like(logits)
            
        return logits, {}

def build_ppo_models(env):
    policy = StablePolicy(
        observation_space=env.observation_space,
        action_space=env.action_space,
        device=device
    )
    
    value = deterministic_model(
        observation_space=env.observation_space,
        action_space=env.action_space,
        device=device,
        network=[
            {"name": "input", "input": "states", "layers": [64, 64], "activations": ["tanh", "tanh"]},
        ],
        output="ONE",
    )
    
    return {"policy": policy, "value": value}

def train_cartpole(algorithm_name, timesteps, seed):
    set_seed(seed)
    
    env = gym.make("CartPole-v1")
    env = gym.wrappers.NormalizeObservation(env)
    env = wrap_env(env)
    
    models = build_ppo_models(env)
    
    agent = PPO(
        models=models,
        memory=RandomMemory(memory_size=5000, num_envs=env.num_envs, device=device),
        cfg={"learning_starts": 1_000,
            "rollouts": 5000,
            "learning_epochs": 4,
            "mini_batches": 32,
            "discount_factor": 0.99,
            "lambda": 0.95,
            "learning_rate": 3e-4,
            "grad_norm_clip": 0.5,
            "ratio_clip": 0.2,
            "value_clip": 0.2,
            "entropy_loss_scale": 0.01,
            "kl_threshold": 0.008,
            "rewards_shaper": lambda rewards, *_: rewards * 0.01,
        },
        observation_space=env.observation_space,
        action_space=env.action_space,
        device=device,
    )

    trainer = SequentialTrainer(
        cfg={
            "timesteps": timesteps,
            "headless": True,
            "progress_interval": 1000,
        },
        env=env,
        agents=agent,
    )
    
    print(f"Training {algorithm_name} for {timesteps} timesteps (seed={seed}) on {device}")
    try:
        trainer.train()
    except Exception as e:
        print(f"Training failed: {str(e)}")
    finally:
        env.close()

if __name__ == "__main__":
    seeds = [123]
    timesteps_list = [2_000_000]
    
    for algo in ["PPO"]:
        for ts in timesteps_list:
            for sd in seeds:
                train_cartpole(algo, ts, sd)
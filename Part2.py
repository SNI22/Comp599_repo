import gymnasium as gym
import torch
import numpy as np
from skrl.agents.torch.dqn import DQN
from skrl.agents.torch.ppo import PPO
from skrl.utils.model_instantiators.torch import deterministic_model, categorical_model

# SKRL API change: wrap_env may be in different modules
try:
    from skrl.envs.wrappers import wrap_env
except ImportError:
    try:
        from skrl.envs.wrappers.torch import wrap_env
    except ImportError:
        from skrl.envs.torch import wrap_env
from skrl.memories.torch import RandomMemory
from skrl.trainers.torch import SequentialTrainer
from torchvision.models import resnet18
from torchvision import transforms as T
import timm
from transformers import CLIPModel, CLIPProcessor

# Select device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Wrapper for image features - must be defined before use
class ImageFeatureWrapper(gym.Wrapper):
    """
    Gym wrapper to replace observations with feature vectors from RGB-rendered frames.
    Requires the inner env to use render_mode="rgb_array".
    """

    def __init__(self, env, extractor, dim):
        super().__init__(env)
        self.extractor = extractor
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(dim,), dtype=np.float32
        )

    def reset(self, **kwargs):
        # Reset underlying env
        result = self.env.reset(**kwargs)
        # Render to get RGB image
        img = self.env.render()
        feat = self.extractor(img)
        # Gymnasium API: reset returns (obs, info)
        if isinstance(result, tuple) and len(result) == 2:
            _, info = result
            return feat, info
        else:
            return feat

    def step(self, action):
        # Step environment
        result = self.env.step(action)
        # Unpack step result
        obs, reward, terminated, truncated, info = result
        # Render to get RGB image
        img = self.env.render()
        feat = self.extractor(img)
        return feat, reward, terminated, truncated, info


# ---------- Vision feature extraction ----------
# ResNet-18 encoder
resnet_encoder = resnet18(weights="IMAGENET1K_V1")
resnet_encoder = torch.nn.Sequential(*list(resnet_encoder.children())[:-1]).to(device)
resnet_encoder.eval()
# CLIP encoder
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
clip_model.eval()

# Common image transform
tform = T.Compose(
    [
        T.ToPILImage(),
        T.Resize((224, 224)),
        T.ToTensor(),
    ]
)


def extract_resnet(img):
    x = tform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        feat = resnet_encoder(x)
    return feat.view(-1).cpu().numpy()


def extract_clip(img):
    inputs = clip_processor(images=img, return_tensors="pt").to(device)
    with torch.no_grad():
        feat = clip_model.get_image_features(**inputs)
    return feat.squeeze().cpu().numpy()


# DINOv2 encoder (if available)
encoders = [("ResNet18", extract_resnet, 512), ("CLIP", extract_clip, 512)]
try:
    dino_model = timm.create_model("dino_vitbase16", pretrained=True)
    dino_model.head = torch.nn.Identity()
    dino_model = dino_model.to(device)
    dino_model.eval()

    def extract_dino(img):
        x = tform(img).unsqueeze(0).to(device)
        with torch.no_grad():
            feat = dino_model(x)
        return feat.squeeze().cpu().numpy()

    encoders.append(
        (
            "DINOv2",
            extract_dino,
            (
                dino_model.head.in_features
                if hasattr(dino_model.head, "in_features")
                else feat.shape[-1]
            ),
        )
    )
except Exception as e:
    print(f"Warning: could not load DINOv2 encoder: {e}")


# ---------- Model builders ----------
def build_dqn_models(env):
    net = [
        {
            "name": "hidden",
            "input": "OBSERVATIONS",
            "layers": [64, 64],
            "activations": ["relu"],
        }
    ]
    qnet = deterministic_model(
        env.observation_space,
        env.action_space,
        device=device,
        network=net,
        output="ACTIONS",
    )
    target = deterministic_model(
        env.observation_space,
        env.action_space,
        device=device,
        network=net,
        output="ACTIONS",
    )
    qnet.to(device)
    target.to(device)
    return {"q_network": qnet, "target_q_network": target}


def build_ppo_models(env):
    net = [
        {
            "name": "shared",
            "input": "OBSERVATIONS",
            "layers": [64, 64],
            "activations": ["relu"],
        }
    ]
    policy = categorical_model(
        env.observation_space,
        env.action_space,
        device=device,
        network=net,
        output="ACTIONS",
    )
    value = deterministic_model(
        env.observation_space,
        env.action_space,
        device=device,
        network=net,
        output="ONE",
    )
    policy.to(device)
    value.to(device)
    return {"policy": policy, "value": value}


# ---------- Training function for image-based RL ----------
def train_image_cartpole(
    encoder_name, extractor, feature_dim, algorithm, timesteps, seed
):
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Build wrapped environment
    raw_env = gym.make("CartPole-v1", render_mode="rgb_array")
    wrapped_env = ImageFeatureWrapper(raw_env, extractor, feature_dim)
    env = wrap_env(wrapped_env)

    # Replay memory
    memory = RandomMemory(
        memory_size=50000, num_envs=env.num_envs, device=device, replacement=False
    )

    # Agent config
    if algorithm == "DQN":
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
        cfg = {
            "learning_starts": 1000,
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
        }
        agent = PPO(
            models=models,
            memory=memory,
            cfg=cfg,
            observation_space=env.observation_space,
            action_space=env.action_space,
            device=device,
        )

    # Trainer
    trainer_cfg = {"timesteps": timesteps, "headless": True}
    trainer = SequentialTrainer(cfg=trainer_cfg, env=env, agents=[agent])
    print(
        f"[Image:{encoder_name}] Training {algorithm} for {timesteps} steps (seed={seed}) on {device}"
    )
    trainer.train()


# ---------- Main ----------
if __name__ == "__main__":
    seeds = [0, 42, 123]
    timesteps_list = [2_000_000, 1_000_000, 500_000]
    algorithms = ["DQN", "PPO"]  # run both DQN and PPO

    for enc_name, fn, dim in encoders:
        for algorithm in algorithms:
            for ts in timesteps_list:
                for sd in seeds:
                    train_image_cartpole(enc_name, fn, dim, algorithm, ts, sd)

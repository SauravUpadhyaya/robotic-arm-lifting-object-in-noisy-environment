import gym
import torch as th
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from typing import Dict


class CustomCombinedExtractor(BaseFeaturesExtractor):
    """Custom features extractor that accepts either a Dict observation space
    containing an image and a small vector, or attempts to infer the correct
    branches when a slightly different observation_space is provided.

    The image branch is a small CNN and the vector branch is an MLP. The
    combined features dim is returned as the extractor output.
    """

    def __init__(self, observation_space, cnn_output_dim: int = 256):
        # Call parent init first with a placeholder features_dim so that
        # nn.Module.__init__ is called and registering submodules is allowed.
        super().__init__(observation_space, features_dim=1)

        # Try to find the image and vector spaces inside the provided observation_space
        img_space = None
        vec_space = None

        if hasattr(observation_space, "spaces"):
            # common for gym.spaces.Dict
            # prefer explicit keys
            if "image" in observation_space.spaces:
                img_space = observation_space.spaces["image"]
            else:
                # find first 3D Box for image
                for k, v in observation_space.spaces.items():
                    if isinstance(v, gym.spaces.Box) and len(v.shape) == 3:
                        img_space = v
                        break

            if "cube_pos_noisy" in observation_space.spaces:
                vec_space = observation_space.spaces["cube_pos_noisy"]
            else:
                # find first 1D Box of length 3
                for k, v in observation_space.spaces.items():
                    if isinstance(v, gym.spaces.Box) and len(v.shape) == 1 and v.shape[0] == 3:
                        vec_space = v
                        break

 
        if img_space is None or vec_space is None:
            raise ValueError(
                "CustomCombinedExtractor expects an observation_space containing an 'image' Box (C,H,W) "
                "and a 'cube_pos_noisy' Box (3,). Got: %r" % (observation_space,)
            )

        C, H, W = img_space.shape

        # build small CNN
        self.cnn = nn.Sequential(
            nn.Conv2d(C, 16, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        # infer flattened size
        with th.no_grad():
            sample = th.zeros(1, C, H, W)
            n_flatten = self.cnn(sample).shape[1]

        self.cnn_mlp = nn.Sequential(nn.Linear(n_flatten, cnn_output_dim), nn.ReLU())

        # vector branch
        self.vec_mlp = nn.Sequential(nn.Linear(vec_space.shape[0], 32), nn.ReLU(), nn.Linear(32, 32), nn.ReLU())
        # combined feature dim
        features_dim = cnn_output_dim + 32
        # set the correct features dim
        self._features_dim = features_dim

    def forward(self, observations: Dict[str, th.Tensor]) -> th.Tensor:
        # observations come as tensors
        img = observations["image"].float()
        vec = observations["cube_pos_noisy"].float()

        cnn_out = self.cnn(img)
        cnn_feat = self.cnn_mlp(cnn_out)
        vec_feat = self.vec_mlp(vec)

        return th.cat([cnn_feat, vec_feat], dim=1)
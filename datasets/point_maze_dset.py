import torch
import decord
import numpy as np
from pathlib import Path
from einops import rearrange
from typing import Callable, Optional
from .traj_dset import TrajDataset, get_train_val_sliced
decord.bridge.set_bridge("torch")

class PointMazeDataset(TrajDataset):
    def __init__(
        self,
        data_path: str = "data/point_maze",
        n_rollout: Optional[int] = None,
        transform: Optional[Callable] = None,
        normalize_action: bool = False,
        action_scale=1.0,
        use_preprocessed: bool = False,
        use_frame_files: bool = False,
    ):
        self.data_path = Path(data_path)
        self.transform = transform
        self.normalize_action = normalize_action
        self.use_preprocessed = use_preprocessed
        self.use_frame_files = use_frame_files

        states = torch.load(self.data_path / "states.pth").float()
        self.states = states
        self.actions = torch.load(self.data_path / "actions.pth").float()
        self.actions = self.actions / action_scale  # scaled back up in env
        self.seq_lengths = torch.load(self.data_path /'seq_lengths.pth')

        self.n_rollout = n_rollout
        if self.n_rollout:
            n = self.n_rollout
        else:
            n = len(self.states)

        self.states = self.states[:n]
        self.actions = self.actions[:n]
        self.seq_lengths = self.seq_lengths[:n]
        self.proprios = self.states.clone()
        print(f"Loaded {n} rollouts")

        self.action_dim = self.actions.shape[-1]
        self.state_dim = self.states.shape[-1]
        self.proprio_dim = self.proprios.shape[-1]

        if normalize_action:
            self.action_mean, self.action_std = self.get_data_mean_std(self.actions, self.seq_lengths)
            self.state_mean, self.state_std = self.get_data_mean_std(self.states, self.seq_lengths)
            self.proprio_mean, self.proprio_std = self.get_data_mean_std(self.proprios, self.seq_lengths)
        else:
            self.action_mean = torch.zeros(self.action_dim)
            self.action_std = torch.ones(self.action_dim)
            self.state_mean = torch.zeros(self.state_dim)
            self.state_std = torch.ones(self.state_dim)
            self.proprio_mean = torch.zeros(self.proprio_dim)
            self.proprio_std = torch.ones(self.proprio_dim)

        self.actions = (self.actions - self.action_mean) / self.action_std
        self.proprios = (self.proprios - self.proprio_mean) / self.proprio_std
    
    def get_data_mean_std(self, data, traj_lengths):
        all_data = []
        for traj in range(len(traj_lengths)):
            traj_len = traj_lengths[traj]
            traj_data = data[traj, :traj_len]
            all_data.append(traj_data)
        all_data = torch.vstack(all_data)
        data_mean = torch.mean(all_data, dim=0)
        data_std = torch.std(all_data, dim=0)
        return data_mean, data_std

    def get_seq_length(self, idx):
        return self.seq_lengths[idx]

    def get_all_actions(self):
        result = []
        for i in range(len(self.seq_lengths)):
            T = self.seq_lengths[i]
            result.append(self.actions[i, :T, :])
        return torch.cat(result, dim=0)

    def get_frames(self, idx, frames):
        visual = self.load_visual_frames(idx, frames)
        proprio = self.proprios[idx, frames]
        act = self.actions[idx, frames]
        state = self.states[idx, frames]
        obs = {
            "visual": visual,
            "proprio": proprio
        }
        return obs, act, state, {} # env_info

    def __getitem__(self, idx):
        return self.get_frames(idx, range(self.get_seq_length(idx)))

    def __len__(self):
        return len(self.seq_lengths)

    def preprocess_imgs(self, imgs):
        if isinstance(imgs, np.ndarray):
            raise NotImplementedError
        elif isinstance(imgs, torch.Tensor):
            return rearrange(imgs, "b h w c -> b c h w") / 255.0

    def load_visual_frames(self, episode_idx: int, frame_indices):
        """Load only requested visual frames for one episode."""
        if isinstance(frame_indices, range):
            frame_indices = list(frame_indices)
        elif isinstance(frame_indices, torch.Tensor):
            frame_indices = frame_indices.tolist()
        else:
            frame_indices = list(frame_indices)

        if self.use_frame_files:
            raw_visual = self._load_frames_from_frame_files(episode_idx, frame_indices)
        else:
            episode_visual = self._load_episode_visual_tensor(episode_idx)
            raw_visual = episode_visual[frame_indices]

        if self.use_preprocessed:
            return raw_visual

        visual = raw_visual / 255.0
        visual = rearrange(visual, "T H W C -> T C H W")
        if self.transform:
            visual = self.transform(visual)
        return visual

    def _load_episode_visual_tensor(self, idx):
        """Load one full episode tensor from `obses/episode_xxx.pth`."""
        obs_dir = self.data_path / "obses"
        image_path = obs_dir / f"episode_{idx:03d}.pth"
        if not image_path.exists():
            raise ValueError(f"Failed to load image for episode {idx}")
        return torch.load(image_path, map_location="cpu")

    def _load_frames_from_frame_files(self, episode_idx: int, frame_indices):
        """Load raw frame tensors from individual frame files."""
        frames = []
        frame_dir = self.data_path / "obses"
        for frame_idx in frame_indices:
            frame_path = frame_dir / f"episode_{episode_idx:03d}_frame_{frame_idx:03d}.pth"
            if not frame_path.exists():
                raise ValueError(
                    f"Failed to load frame {frame_idx} for episode {episode_idx}"
                )
            frame = torch.load(frame_path, map_location="cpu")
            frames.append(frame)
        return torch.stack(frames, dim=0)
        
def load_point_maze_slice_train_val(
    transform,
    n_rollout=50,
    data_path='data/pusht_dataset',
    normalize_action=False,
    split_ratio=0.8,
    num_hist=0,
    num_pred=0,
    frameskip=0,
    use_preprocessed=False,
    use_frame_files=False,
):
    dset = PointMazeDataset(
        n_rollout=n_rollout,
        transform=transform,
        data_path=data_path,
        normalize_action=normalize_action,
        use_preprocessed=use_preprocessed,
        use_frame_files=use_frame_files,
    )
    dset_train, dset_val, train_slices, val_slices = get_train_val_sliced(
        traj_dataset=dset, 
        train_fraction=split_ratio, 
        num_frames=num_hist + num_pred, 
        frameskip=frameskip
    )

    datasets = {}
    datasets['train'] = train_slices
    datasets['valid'] = val_slices
    traj_dset = {}
    traj_dset['train'] = dset_train
    traj_dset['valid'] = dset_val
    return datasets, traj_dset

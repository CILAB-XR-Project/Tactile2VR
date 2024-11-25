from torch.utils.data import Dataset, DataLoader, Subset
from collections import OrderedDict
import numpy as np
import os
import torch
import math
from pathlib import Path

from const import TACTILE_SIZE, ACTIVITY_LIST
from utils import get_action_type

def pad_tactile(a, shape):
    assert a.ndim == 3
    if tuple(a.shape[1:]) == tuple(shape):
        return a
    z = np.zeros((a.shape[0], shape[0], shape[1]))
    z[:, :a.shape[1], :a.shape[2]] = a
    return z

def split_list(array, ratio):
    ratio_sum = round(sum(ratio), 7)
    assert ratio_sum == 1, f"{ratio_sum}"
    rr = (np.array(ratio) * len(array)).astype(int).tolist()
    for i in range(1, len(rr)):
        rr[i] += rr[i-1]
    rr.insert(0, 0)

    result = []
    for i in range(len(rr)-2):
        start_idx = rr[i]
        end_idx = rr[i+1]
        result.append(array[start_idx:end_idx])
    result.append(array[end_idx:])
    assert len(result)==len(ratio)
    return result

def generate_heatmap(keypoint):  # (B, VR_kps, 3) -> (B, VR_kps, 20, 20, 18) 
    def _gaussian(dis, mu, sigma):
        return 1 / (mu * torch.sqrt(torch.tensor(2 * math.pi))) * torch.exp(-0.5 * ((dis - mu) / sigma)**2)

    def _softmax(x):
        x_max = torch.amax(x, dim=(-3, -2, -1), keepdim=True)  # torch.amax는 여러 차원에 대해 작동
        exp_x = torch.exp(x - x_max)  # 오버플로우 방지
        return exp_x / torch.sum(exp_x, dim=(-3, -2, -1), keepdim=True)
    
    size = [20, 20, 18]
    # 3D voxel space 생성
    pos_x, pos_y, pos_z = torch.meshgrid(
        torch.linspace(0., 1., int(size[0])).to(keypoint.device),
        torch.linspace(0., 1., int(size[1])).to(keypoint.device),
        torch.linspace(0., 1., int(size[2])).to(keypoint.device)  
    )
    pos_x = pos_x.unsqueeze(0).unsqueeze(0) # (1, 1, 20, 20, 18)
    pos_y = pos_y.unsqueeze(0).unsqueeze(0) # (1, 1, 20, 20, 18)
    pos_z = pos_z.unsqueeze(0).unsqueeze(0) # (1, 1, 20, 20, 18)

    # keypoint 확장
    keypoint = keypoint.unsqueeze(-2).unsqueeze(-2).unsqueeze(-2)  # (B, window, VR_kps, 1, 1, 1, 3)

    # 거리 계산 (Broadcasting)
    dis = torch.sqrt(
        (pos_x - keypoint[..., 0])**2 +
        (pos_y - keypoint[..., 1])**2 +
        (pos_z - keypoint[..., 2])**2
    )  # (B, VR_kps, 20, 20, 18)

    # Gaussian 계산
    g = _gaussian(dis, 0.001, 1)  # (B, VR_kps, 20, 20, 18)

    # Softmax 계산
    heatmap = _softmax(g)  # (B, VR_kps, 20, 20, 18)

    return heatmap

def split_by_file(dataset, train_ratio=0.7, valid_ratio=0.2, test_ratio=0.1):
    train_indices = []
    valid_indices = []
    test_indices = []
    train_mappings = []
    valid_mappings = []
    test_mappings = []

    cur_train_start = 0
    cur_valid_start = 0
    cur_test_start = 0
    
    for start, end, data_dir in dataset.file_mappings:
        file_indices = list(range(start, end))
        total_len = len(file_indices)
        
        train_len = int(total_len * train_ratio)
        valid_len = int(total_len * valid_ratio)
        test_len = total_len - train_len - valid_len
        
        train_indices.extend(file_indices[:train_len])
        valid_indices.extend(file_indices[train_len:train_len + valid_len])
        test_indices.extend(file_indices[-test_len:])
        
        # 각각의 매핑을 업데이트
        train_mappings.append((cur_train_start, cur_train_start + train_len, data_dir))
        valid_mappings.append((cur_valid_start, cur_valid_start + valid_len, data_dir))
        test_mappings.append((cur_test_start, cur_test_start + test_len, data_dir))

        cur_train_start += train_len
        cur_valid_start += valid_len
        cur_test_start += test_len
    return (train_indices, valid_indices, test_indices), (train_mappings, valid_mappings, test_mappings)


class SlidingWindowDataset(Dataset):
    def __init__(self, all_paths, cache_size, config, **kwargs):
        self.all_paths = all_paths
        self.cache_size = cache_size
        self.config = config
        self.kwargs = kwargs

        self.buffers = OrderedDict()  # Dictionary to hold the loaded data files
        self.file_indices, self.file_mappings = self._prepare_indices()
        # self.shuffle_indices()
        
        if "filter_by_action" in self.kwargs:
            if self.kwargs["filter_by_action"]:
                print(f"Only use {self.kwargs['action_list']}")
        if "filter_by_person" in self.kwargs:
            if self.kwargs["filter_by_person"]:
                print(f"Only use {self.kwargs['person_list']}")

    def _prepare_indices(self):
        indices = []
        mappings = []
        for data_dir in self.all_paths:
            folder = data_dir.split("/")[-1]
            action_type = get_action_type(folder)
            person_id = folder[:2]
            if "filter_by_action" in self.kwargs:
                if self.kwargs["filter_by_action"]:
                    if action_type not in self.kwargs["action_list"]:
                        continue
            if "filter_by_person" in self.kwargs:
                if self.kwargs["filter_by_person"]:
                    if person_id not in self.kwargs["person_list"]:
                        continue

            kp_path = os.path.join(data_dir, "keypoints.npy")
            kp = np.load(kp_path)
            data_len = kp.shape[0]
            num_windows = data_len - self.config.WINDOW_SIZE + 1
            start = len(indices)
            indices.extend([(kp_path, i) for i in range(num_windows)])
            end = len(indices)
            mappings.append((start, end, data_dir))
            # if "test" in data_dir:
            #     print(f"Data: {data_dir} - {num_windows} windows")
        return indices, mappings

    
    def _load_data(self, file_path):
        # Check if the data is already in the buffer
        if file_path in self.buffers:
            # Move to the end to show that it was recently used
            self.buffers.move_to_end(file_path)
            return self.buffers[file_path]

        # Load data and put it into the buffer
        kp_path = file_path
        # hm_path = file_path.replace("keypoints", "heatmaps")
        tac_left_path = file_path.replace("keypoints", "tactile_left")
        tac_right_path = file_path.replace("keypoints", "tactile_right")

        kp = np.load(open(kp_path, 'rb'))
        # hm = np.load(open(hm_path, 'rb'))
        tac_left = np.load(open(tac_left_path, 'rb'))
        tac_right = np.load(open(tac_right_path, 'rb'))
        action = get_action_type(file_path)
        action_idx = ACTIVITY_LIST.index(action)

        tac_left = pad_tactile(tac_left, TACTILE_SIZE)
        tac_right = pad_tactile(tac_right, TACTILE_SIZE)

        data = (kp, tac_left, tac_right, action_idx)
        self.buffers[file_path] = data

        # Move to the end to show that it was recently used
        self.buffers.move_to_end(file_path)

        # If the buffer exceeds the cache size, remove the oldest item
        if len(self.buffers) > self.cache_size:
            self.buffers.popitem(last=False)  # Remove the first item

        return data
    
    def shuffle_indices(self):
        np.random.seed(6897)
        np.random.shuffle(self.file_indices)
    
    def __len__(self):
        return len(self.file_indices)

    
    def __getitem__(self, idx):
        file_path, start_index = self.file_indices[idx]
        data_buffer = self._load_data(file_path)  # Ensure the data is loaded
        kp, tac_left, tac_right, action_idx = data_buffer

        target_idx = start_index + self.config.WINDOW_SIZE - 1
        
        # input: tactile_left, tactile_right, heatmap
        _input_tac_left = tac_left[start_index:start_index + self.config.WINDOW_SIZE]
        _input_tac_right = tac_right[start_index:start_index + self.config.WINDOW_SIZE]
        _input_kp = kp[start_index:start_index + self.config.WINDOW_SIZE]
        # output: kp
        
     
    
        return _input_tac_left, _input_tac_right, _input_kp, action_idx


class ShuffleDataloader(DataLoader):
    def shuffle_indices(self):
        self.dataset.dataset.shuffle_indices()       


def get_tactile_dataloaders(data_dir, config):
    data_paths = sorted([str(path) for path in Path(data_dir).iterdir()])
    dataset = SlidingWindowDataset(data_paths, config.CACHE_SIZE, config)
    
    (train_indices, valid_indices, test_indices), (train_mappings, valid_mappings, test_mappings) = split_by_file(
        dataset, train_ratio=0.7, valid_ratio=0.2, test_ratio=0.1
    )    
    train_dataset = Subset(dataset, train_indices)
    valid_dataset = Subset(dataset, valid_indices)
    test_dataset = Subset(dataset, test_indices)
    
    print(f"Train/Valid/Test frames: {len(train_dataset)}/{len(valid_dataset)}/{len(test_dataset)}")
    print(f"CacheSize/FileNum: {config.CACHE_SIZE}/{len(dataset.file_mappings)}")

    train_dataloader = ShuffleDataloader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
    )
    valid_dataloader = ShuffleDataloader(
        valid_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
    )
    test_dataloader = ShuffleDataloader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
    )
    
    return (train_dataloader, valid_dataloader, test_dataloader), (train_dataset, valid_dataset, test_dataset), (train_mappings, valid_mappings, test_mappings)


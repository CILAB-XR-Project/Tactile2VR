from torch.utils.data import Dataset, DataLoader, Subset
from collections import OrderedDict
from utils import get_action_type
import numpy as np
from pathlib import Path

from const import TACTILE_SIZE, ACTIVITY_LIST

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
            num_windows = data_len - self.config.WINDOW_SIZE*2 + 1
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
        hm_path = file_path.replace("keypoints", "heatmaps")
        tac_left_path = file_path.replace("keypoints", "tactile_left")
        tac_right_path = file_path.replace("keypoints", "tactile_right")

        kp = np.load(open(kp_path, 'rb'))
        hm = np.load(open(hm_path, 'rb'))
        tac_left = np.load(open(tac_left_path, 'rb'))
        tac_right = np.load(open(tac_right_path, 'rb'))
        action = get_action_type(file_path)
        action_idx = ACTIVITY_LIST.index(action)

        tac_left = pad_tactile(tac_left, TACTILE_SIZE)
        tac_right = pad_tactile(tac_right, TACTILE_SIZE)

        data = (kp, hm, tac_left, tac_right, action_idx)
        self.buffers[file_path] = data

        # Move to the end to show that it was recently used
        self.buffers.move_to_end(file_path)

        # If the buffer exceeds the cache size, remove the oldest item
        if len(self.buffers) > self.cache_size:
            self.buffers.popitem(last=False)  # Remove the first item

        return data
    
    def shuffle_indices(self):
        np.random.shuffle(self.file_indices)
        
    def __len__(self):
        return len(self.file_indices)

    def __getitem__(self, idx):
        file_path, start_index = self.file_indices[idx]
        data_buffer = self._load_data(file_path)  # Ensure the data is loaded
        kp, hm, tac_left, tac_right, action_idx = data_buffer

        # indexing data
        if self.config.PREDICT_MIDDLE:
            target_idx = start_index + self.config.WINDOW_SIZE//2 - 1
        else:
            target_idx = start_index + self.config.WINDOW_SIZE - 1
        
        # input: tactile_left, tactile_right, heatmap
        _input_tac_left = tac_left[start_index:start_index + self.config.WINDOW_SIZE]
        _input_tac_right = tac_right[start_index:start_index + self.config.WINDOW_SIZE]
        _input_hm = hm[start_index:start_index + self.config.WINDOW_SIZE]
        
        future_start_index = start_index + self.config.WINDOW_SIZE + self.config.WINDOW_SIZE//2

        # output: future_tactile_left, future_tactile_right, future_heatmap, kp
        _output_kp = kp[target_idx]  # 단일 프레임 
        
        _output_tac_left = tac_left[future_start_index:future_start_index+self.config.WINDOW_SIZE//2] #미래 프레임
        _output_tac_right = tac_right[future_start_index:future_start_index+self.config.WINDOW_SIZE//2] #미래 프레임
        _output_hm = hm[future_start_index:future_start_index+self.config.WINDOW_SIZE//2] #미래 프레임
    
        return _input_tac_left, _input_tac_right, _input_hm, _output_tac_left, _output_tac_right, _output_hm, _output_kp, action_idx


class ShuffleDataloader(DataLoader):
    def shuffle_indices(self):
        self.dataset.dataset.shuffle_indices()       


def get_tactile_dataloaders(data_dir, config):
    data_paths = sorted([str(path) for path in Path(data_dir).iterdir()])
    dataset = SlidingWindowDataset(data_paths, config.CACHE_SIZE, config)
    indices = list(range(len(dataset)))
    
    train_indices, valid_indices, test_indices = split_list(indices, [0.7, 0.2, 0.1])
    train_dataset = Subset(dataset, train_indices)
    valid_dataset = Subset(dataset, valid_indices)
    test_dataset = Subset(dataset, test_indices)
    
    print(f"Train/Valid/Test frames: {len(train_dataset)}/{len(valid_dataset)}/{len(test_dataset)}")
    print(f"CacheSize/FileNum: {config.CACHE_SIZE}/{len(dataset.file_mappings)}")

    train_dataloader = ShuffleDataloader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
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
    
    return train_dataloader, valid_dataloader, test_dataloader

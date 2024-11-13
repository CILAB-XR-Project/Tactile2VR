import os, sys
import inspect
import torch
import torch.nn as nn
import json
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from tqdm import tqdm
import copy

from config import Tactile2PoseConfig
from model import Tactile2PoseFeatureModel, SpatialSoftmax3D
from utils import DualOutput
from dataloader import get_tactile_dataloaders


def get_spatial_keypoint(keypoint):
    spatial_keypoint = copy.deepcopy(keypoint)
    spatial_keypoint[:, :, 2] *= 2
    spatial_keypoint *= 100
    return spatial_keypoint

def get_keypoint_spatial_dis(keypoint_GT, keypoint_pred):
    dis = get_spatial_keypoint(keypoint_pred) - get_spatial_keypoint(keypoint_GT)
    eud = np.sum(dis**2, axis=-1) ** 0.5
    xyz_diff = np.abs(dis)
    return eud, xyz_diff

def run_feature_epoch(config, model, dataloader, softmax, optimizer,  writer, epoch, visualize=False, test_mode=False, device="cuda", name="train"):
    if config.NAME == "tactile2pose":
        history = {
            "heatmap_loss": [],
            "keypoint_loss": [],
            "total_loss": [],
            "cm_L2_keypoint": [],
        }
        is_tactile2pose = True
    elif config.NAME == "pose2tactile":
        history = {
            "tactile_left_loss": [],
            "tactile_right_loss": [],
            "total_loss": []
        }
        is_tactile2pose = False
    else:
        raise ValueError(f"Invalid config name: {config.NAME}")
            
    if test_mode:
        model.eval()
    else:
        model.train()
        # dataloader.shuffle_indices()
            
    if config.PREDICT_MIDDLE:
        target_idx = config.WINDOW_SIZE//2 - 1
    else:
        target_idx = config.WINDOW_SIZE - 1
    mse_loss = nn.MSELoss()
    pbar = tqdm(dataloader, total=len(dataloader), desc=f"{name} epoch: {epoch}")
    for i, (input_tac_left, input_tac_right, input_hm, _, _, _, output_kp, _) in enumerate(pbar):
        #Heatmap은?
        heatmap = input_hm.clone()
        heatmap[heatmap < 0.01] = 0
        heatmap[heatmap > 0.01] *= 100
        
        tactile_left = input_tac_left.to(device).float()
        tactile_right = input_tac_right.to(device).float()
        heatmap = heatmap.to(device).float()
        keypoint_label =  output_kp.to(device).float()

        
        #일단 쪼갰다고 가정 후 loss 계산
        if is_tactile2pose:
            heatmap_pred = model(tactile_left, tactile_right)
            heatmap_label = heatmap[:, target_idx]
            heatmap_loss = mse_loss(heatmap_pred, heatmap_label) * 100
            
            heatmap_pred = torch.clip(heatmap_pred, 0, None)
            keypoint_out, _ = softmax(heatmap_pred)
            keypoint_loss = mse_loss(keypoint_out, keypoint_label) * 1000
            
            total_loss = heatmap_loss + keypoint_loss
            
            eud, _ = get_keypoint_spatial_dis(keypoint_out.detach().cpu().numpy(), keypoint_label.detach().cpu().numpy())
            history["heatmap_loss"].append(heatmap_loss.item())
            history["keypoint_loss"].append(keypoint_loss.item())
            history["cm_L2_keypoint"].append(np.mean(eud))
            
        else:
            tactile_left_pred, tactile_right_pred = model(heatmap)
            tectile_left_label = tactile_left[:, target_idx]
            tactile_right_label = tactile_right[:, target_idx]
            tactile_left_loss = mse_loss(tactile_left_pred, tectile_left_label) * 100
            tactile_right_loss = mse_loss(tactile_right_pred, tactile_right_label) * 100
            total_loss = tactile_left_loss + tactile_right_loss
            
            history["tactile_left_loss"].append(tactile_left_loss.item())
            history["tactile_right_loss"].append(tactile_right_loss.item())
            
        history["total_loss"].append(total_loss.item())
            

        if not test_mode:
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            
        pbar.set_description(
            f"[{name}] Epoch={epoch} Iter={i} Loss={total_loss.item():.2f}")

    if visualize:
        #TODO: visualize feature train
        pass
        #writer.
        
    for key in history.keys():
        history[key] = np.mean(history[key])
    
    return history


if __name__ == "__main__":
    # load config
    config = Tactile2PoseConfig()
    
    # create log directory
    log_name = input("Enter Run name: ")
    current_time = datetime.now().strftime('%b%d_%H-%M-%S')
    log_dir = os.path.join('runs', f"{current_time}_{log_name}")
    writer = SummaryWriter(log_dir=log_dir)
    
    #load model
    model = Tactile2PoseFeatureModel(config)
    softmax = SpatialSoftmax3D(20,20,18,19)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)    
    model.to(device)
    softmax.to(device)
        
    print("Model created")
    print("======Configurations======")
    print(f"Model Name: {config.NAME}")
    print(f"Batch Size: {config.BATCH_SIZE}")
    print(f"Window Size: {config.WINDOW_SIZE}")
    print(f"Predict Middle: {config.PREDICT_MIDDLE}")
    print(f"Log directory: {log_dir}")
    print("==========================")
    
    #load data
    data_dir = "/app/raid/isaac/InsoleData/dataset/"
    train_dataloader, valid_dataloader, test_dataloader = get_tactile_dataloaders(data_dir, config)
    
    # save config and train code
    config.save_to_json(os.path.join(log_dir, "config.json"))
    current_code = inspect.getsource(sys.modules[__name__])
    with open(os.path.join(writer.log_dir, "train_code.py"), 'w') as f:
        f.write(current_code)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    
    # Start train
    with DualOutput(os.path.join(writer.log_dir, "train_log.txt")):
        # multimodal feature train
        print(f"===Start {config.NAME} Multimodal Feature Training===")      
        best_feature_val_loss = np.inf

        for epoch in range(config.EPOCHS):
            train_history =  run_feature_epoch(config, model, train_dataloader, softmax, optimizer, writer, epoch, device=device, name="train")
            valid_history = run_feature_epoch(config, model, valid_dataloader, softmax, optimizer, writer, epoch, test_mode=True, device=device, name="valid")
            test_history = run_feature_epoch(config, model, test_dataloader,  softmax, optimizer, writer, epoch, test_mode=True, device=device, name="test")
        
     
            feature_train_history = {}
            for key in train_history.keys():
                feature_train_history["feature_train/" + key] = train_history[key]
            for key in valid_history.keys():
                feature_train_history["feature_valid/" + key] = valid_history[key]
            for key in test_history.keys():
                feature_train_history["feature_test/" + key] = test_history[key]

            for key in feature_train_history.keys():
                writer.add_scalar(key, feature_train_history[key], epoch)

            
            val_loss = feature_train_history["feature_valid/total_loss"]
            if val_loss < best_feature_val_loss:
                print(f"Best model found at epoch {epoch}. val_loss: {val_loss}")
                best_feature_val_loss = val_loss
                torch.save(model, os.path.join(writer.log_dir, "best_model.pth"))
        print(f"Best model val_loss: {best_feature_val_loss}")
        print(f"===End Training===\n\n")
        
        
    
    writer.close()

        
            
            
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
import random

from config import Tactile2PoseConfig

from model import Tactile2PoseFeatureModel, SpatialSoftmax3D
from utils import DualOutput
from dataloader import get_tactile_dataloaders
from visualize import plotMultiKeypoint, plotTactile, plot3Dheatmap
from const import VR_INDEXS

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

def run_epoch(config, model, dataloader, softmax, optimizer,  writer, epoch, visualize=False, test_mode=False, device="cuda", name="train"):
    history = {
        "keypoint_loss": [],
        "total_loss": [],
        "cm_L2_keypoint": [],
    }
            
    if test_mode:
        model.eval()
    else:
        model.train()
        # dataloader.shuffle_indices()
            
    mse_loss = nn.MSELoss()
    pbar = tqdm(dataloader, total=len(dataloader), desc=f"{name} epoch: {epoch}")
    for i, (input_tac_left, input_tac_right, input_kp, _) in enumerate(pbar):

        tactile_left = input_tac_left.to(device).float()
        tactile_right = input_tac_right.to(device).float()
        keypoint_vr = input_kp[:,:,VR_INDEXS, :].to(device).float()
        keypoint_label =  input_kp[:,-1,:,:].to(device).float()

        heatmap_pred = model(tactile_left, tactile_right, keypoint_vr)
        
        heatmap_pred = torch.clip(heatmap_pred, 0, None)
        keypoint_out, _ = softmax(heatmap_pred)
        keypoint_loss = mse_loss(keypoint_out, keypoint_label) * 1000
        
        total_loss = keypoint_loss
        
        eud, _ = get_keypoint_spatial_dis(keypoint_out.detach().cpu().numpy(), keypoint_label.detach().cpu().numpy())
        history["keypoint_loss"].append(keypoint_loss.item())
        history["cm_L2_keypoint"].append(np.mean(eud))

        history["total_loss"].append(total_loss.item())
            

        if not test_mode:
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

        pbar.set_description(
            f"[{name}] Epoch={epoch} Iter={i} Loss={total_loss.item():.2f} L2={np.mean(eud):.2f}cm")

        
    if visualize:
        keypoint_np = keypoint_label.detach().cpu().numpy()
        keypoint_pred_np = keypoint_out.detach().cpu().numpy()
        tactile_left_np = tactile_left.detach().cpu().numpy()
        tactile_right_np = tactile_right.detach().cpu().numpy()
        heatmap_pred_np = heatmap_pred.detach().cpu().numpy()

        k = random.randint(0, keypoint_np.shape[0]-1)

        img1 = plotMultiKeypoint([keypoint_np[k]], limit=[(0, 1), (0, 1), (0, 1)])
        img2 = plotMultiKeypoint([keypoint_pred_np[k]], limit=[(0, 1), (0, 1), (0, 1)])

        if config.PREDICT_MIDDLE:
            tactile_idx = config.WINDOW_SIZE//2 - 1
        else:
            tactile_idx = -1

        img3 = plotTactile(tactile_left_np[k][tactile_idx])
        img4 = plotTactile(tactile_right_np[k][tactile_idx])
        img5 = plot3Dheatmap(heatmap_pred_np[k])

        # Log images
        writer.add_image(f'{name}/Images/Keypoint_True', img1, epoch, dataformats='HWC')
        writer.add_image(f'{name}/Images/Keypoint_Pred', img2, epoch, dataformats='HWC')
        writer.add_image(f'{name}/Images/TactileLeft', img3, epoch, dataformats='HWC')
        writer.add_image(f'{name}/Images/TactileRight', img4, epoch, dataformats='HWC')
        writer.add_image(f'{name}/Images/Heatmap_Pred', img5, epoch, dataformats='HWC')

        
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
    loaders, datasets, mappings = get_tactile_dataloaders(data_dir, config)
    train_dataloader, valid_dataloader, test_dataloader = loaders
    
    # save config and train code
    config.save_to_json(os.path.join(log_dir, "config.json"))
    current_code = inspect.getsource(sys.modules[__name__])
    with open(os.path.join(writer.log_dir, "train_code.py"), 'w') as f:
        f.write(current_code)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    
    # Start train
    with DualOutput(os.path.join(writer.log_dir, "train_log.txt")):
        # multimodal feature train
        print(f"===Start {config.NAME} Tactile2Pose Training===")      
        best_feature_val_loss = np.inf

        for epoch in range(config.EPOCHS):
            train_history =  run_epoch(config, model, train_dataloader, softmax, optimizer, writer, epoch, device=device, name="train")
            valid_history = run_epoch(config, model, valid_dataloader, softmax, optimizer, writer, epoch, test_mode=True, device=device, visualize=True, name="valid")
            test_history = run_epoch(config, model, test_dataloader,  softmax, optimizer, writer, epoch, test_mode=True, device=device, visualize=True, name="test")
        
     
            feature_train_history = {}
            for key in train_history.keys():
                feature_train_history["train/" + key] = train_history[key]
            for key in valid_history.keys():
                feature_train_history["valid/" + key] = valid_history[key]
            for key in test_history.keys():
                feature_train_history["test/" + key] = test_history[key]

            for key in feature_train_history.keys():
                writer.add_scalar(key, feature_train_history[key], epoch)

            
            val_loss = feature_train_history["valid/total_loss"]
            if val_loss < best_feature_val_loss:
                print(f"Best model found at epoch {epoch}. val_loss: {val_loss}")
                best_feature_val_loss = val_loss
                torch.save(model, os.path.join(writer.log_dir, "best_model.pth"))
        print(f"Best model val_loss: {best_feature_val_loss}")
        print(f"===End Training===\n\n")
        
        
    
    writer.close()

        
            
            
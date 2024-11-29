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

from model import Tactile2PoseFeatureModel, Tactile2PoseVRHeatmap, Tactile2PoseAction, SpatialSoftmax3D
from utils import DualOutput
from dataloader import get_tactile_dataloaders, generate_heatmap
from visualize import plotMultiKeypoint, plotTactile, plot_action_confusion_matrix
from const import VR_INDEXS, SCALED_WEIGHT_INDEX

def calc_keypoint_loss(keypoint_GT, keypoint_pred, eud):
    # scaling_factor = eud
    keypoint_len = keypoint_GT.shape[1]
    # scaling_factor = [5.0 if i in SCALED_WEIGHT_INDEX else 1.0 for i in range(keypoint_len)]
    
    scaling_factor = torch.ones(keypoint_len).to(keypoint_GT.device)
    scaling_factor[SCALED_WEIGHT_INDEX] = 5.0
    
    mse_losses = torch.mean((keypoint_GT - keypoint_pred) ** 2, axis=-1)
    scaled_mse_losses = torch.mean(mse_losses * scaling_factor) * 10000
    
    return scaled_mse_losses, torch.mean(mse_losses)*10000

def get_spatial_keypoint(keypoint):
    spatial_keypoint = keypoint.clone()
    spatial_keypoint *= 2
    spatial_keypoint *= 100
    return spatial_keypoint

def get_keypoint_spatial_dis(keypoint_GT, keypoint_pred):
    dis = get_spatial_keypoint(keypoint_pred) - get_spatial_keypoint(keypoint_GT)
    eud = torch.sqrt(torch.sum(dis**2, dim=-1))
    xyz_diff = torch.abs(dis) 
    return eud, xyz_diff

    

def run_epoch(config, model, dataloader, softmax, optimizer,  writer, epoch, visualize=False, test_mode=False, device="cuda", name="train"):
    history = {
        "keypoint_loss": [],
        "action_loss" : [],
        "total_loss": [],
        "action_accuracy": [],
        "cm_L2_keypoint": [],
    }
            
    if test_mode:
        model.eval()
    else:
        model.train()
        # dataloader.shuffle_indices()
    action_len = len(config.ACTION_LIST)
    class_history = np.zeros((action_len, action_len),dtype=int)
            
    mse_loss = nn.MSELoss()
    pbar = tqdm(dataloader, total=len(dataloader), desc=f"{name} epoch: {epoch}")
    for i, (input_tac_left, input_tac_right, input_kp, action_idx) in enumerate(pbar):

        tactile_left = input_tac_left.to(device).float()
        tactile_right = input_tac_right.to(device).float()
        keypoint_label =  input_kp[:,-1,:,:].to(device).float()
        action_idx = action_idx.cuda()

        # if config.MODEL == "Tactile2PoseVRHeatmap":
        #     keypoint_vr = input_kp[:,-1,VR_INDEXS, :].to(device).float()
        #     vr_keypoint_heatmap = generate_heatmap(keypoint_vr)
        #     heatmap_pred, action_pred = model(tactile_left, tactile_right, vr_keypoint_heatmap)
            
        # else:
        #     keypoint_vr = input_kp[:,:,VR_INDEXS, :].to(device).float()
        #     heatmap_pred, action_pred = model(tactile_left, tactile_right, keypoint_vr)
        
        heatmap_pred, action_pred = model(tactile_left, tactile_right)
            
        heatmap_pred = torch.clip(heatmap_pred, 0, None)
        keypoint_out, _ = softmax(heatmap_pred)
        eud, _ = get_keypoint_spatial_dis(keypoint_out.detach(), keypoint_label.detach())
        scaled_keypoint_loss, keypoint_loss = calc_keypoint_loss(keypoint_label, keypoint_out, eud)
        # keypoint_loss = mse_loss(keypoint_out, keypoint_label) * 10000
        action_loss = torch.nn.functional.cross_entropy(action_pred, action_idx)
        total_loss = scaled_keypoint_loss + action_loss
            
        if not test_mode:
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
              
        _, predicted = torch.max(action_pred, 1)     
        correct_predictions = (predicted == action_idx).sum().item()
        total_predictions = action_idx.shape[0]
        
        accuracy = correct_predictions / total_predictions
        
        history["action_loss"].append(action_loss.item())
        history["keypoint_loss"].append(keypoint_loss.item())
        history["cm_L2_keypoint"].append(torch.mean(eud).item())
        history["total_loss"].append(total_loss.item())
        history["action_accuracy"].append(accuracy)
        
        for idx, action_label in enumerate(action_idx):
            class_history[action_label, predicted[idx]] += 1

        pbar.set_description(
            f"[{name}] Epoch={epoch} Iter={i} Loss={total_loss.item():.2f} L2={torch.mean(eud).item():.2f}cm  Acc={accuracy:.2f}")
        
    if visualize:
        keypoint_np = keypoint_label.detach().cpu().numpy()
        keypoint_pred_np = keypoint_out.detach().cpu().numpy()
        tactile_left_np = tactile_left.detach().cpu().numpy()
        tactile_right_np = tactile_right.detach().cpu().numpy()

        k = random.randint(0, keypoint_np.shape[0]-1)

        img1 = plotMultiKeypoint([keypoint_np[k]], limit=[(0, 1), (0, 1), (0, 1)])
        img2 = plotMultiKeypoint([keypoint_pred_np[k]], limit=[(0, 1), (0, 1), (0, 1)])
        

        if config.PREDICT_MIDDLE:
            tactile_idx = config.WINDOW_SIZE//2 - 1
        else:
            tactile_idx = -1

        img3 = plotTactile(tactile_left_np[k][tactile_idx])
        img4 = plotTactile(tactile_right_np[k][tactile_idx])
        img5 = plot_action_confusion_matrix(class_history)

        # Log images
        writer.add_image(f'{name}/Images/Keypoint_True', img1, epoch, dataformats='HWC')
        writer.add_image(f'{name}/Images/Keypoint_Pred', img2, epoch, dataformats='HWC')
        writer.add_image(f'{name}/Images/TactileLeft', img3, epoch, dataformats='HWC')
        writer.add_image(f'{name}/Images/TactileRight', img4, epoch, dataformats='HWC')
        writer.add_image(f'{name}/Images/Action_Confusion_matrix', img5, epoch, dataformats='HWC')
     
        
    for key in history.keys():
        history[key] = np.mean(history[key])
    
    return history


if __name__ == "__main__":
    # load config
    model_dict ={
        "Tactile2PoseFeatureModel": Tactile2PoseFeatureModel,
        "Tactile2PoseVRHeatmap": Tactile2PoseVRHeatmap,
        # "Tactile2PoseVRLinear": Tactile2PoseVRLinear
    }
    
    config = Tactile2PoseConfig()
    # config.MODEL = "Tactile2PoseFeatureModel"
    config.MODEL = "Tactile2PoseAction"
    # config.MODEL = "Tactile2PoseVRLinear"
    
    # create log directory
    log_name = input("Enter Run name: ")
    current_time = datetime.now().strftime('%b%d_%H-%M-%S')
    log_dir = os.path.join('runs', f"{current_time}_{log_name}")
    writer = SummaryWriter(log_dir=log_dir)
    
    #load model
    
    # model = model_dict[config.MODEL](config)
    model = Tactile2PoseAction(config)
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
        best_val_loss = np.inf
        best_l2_dist = np.inf
        

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
            if val_loss < best_val_loss:
                print(f"Best model found at epoch {epoch}. val_loss: {val_loss}")
                best_val_loss = val_loss
                torch.save(model, os.path.join(writer.log_dir, "best_model.pth"))
                
            l2_dist = feature_train_history["valid/cm_L2_keypoint"]
            if l2_dist < best_l2_dist:
                print(f"Best model found at epoch {epoch}. L2 dist: {l2_dist}")
                best_l2_dist = l2_dist
                torch.save(model, os.path.join(writer.log_dir, "best_model_l2.pth"))    
            
        print(f"Best model val_loss: {best_val_loss}")
        print(f"===End Training===\n\n")
        
        
    
    writer.close()

        
            
            
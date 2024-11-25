import os, sys
import socket
import json
import numpy as np


from model import Tactile2PoseFeatureModel, Tactile2PoseVRHeatmap, Tactile2PoseVRLinear, SpatialSoftmax3D
from config import Tactile2PoseConfig
from dataloader import get_tactile_dataloaders, generate_heatmap
import const
import torch

HOST = "127.0.0.1"
PORT = 12345


def load_model(config, model_path, device="cuda"):
    model_dict ={
        "Tactile2PoseFeatureModel": Tactile2PoseFeatureModel,
        "Tactile2PoseVRHeatmap": Tactile2PoseVRHeatmap,
        "Tactile2PoseVRLinear": Tactile2PoseVRLinear
    }
    
    model = model_dict[config.MODEL](config)
    model.to(device)
    softmax = SpatialSoftmax3D(20, 20, 18, 19)
    softmax.to(device)
    
    #load model
    try:
        model.load_state_dict(torch.load(model_path).state_dict())
    except:
        model = torch.nn.DataParallel(model)
        model.load_state_dict(torch.load(model_path).state_dict())

    return model, softmax

def get_pose_aciton_data(config, data, model, softmax, device="cuda"):
        
    input_tac_left, input_tac_right, input_kp, action_idx = data    
    
    tactile_left = input_tac_left.to(device).float()
    tactile_right = input_tac_right.to(device).float()
    
    if config.MODEL == "Tactile2PoseVRHeatmap":
        keypoint_vr = input_kp[:,-1, const.VR_INDEXS, :].to(device).float()
        vr_keypoint_heatmap = generate_heatmap(keypoint_vr)
        heatmap_pred = model(tactile_left, tactile_right, vr_keypoint_heatmap)
    
    else:
        keypoint_vr = input_kp[:,:,const.VR_INDEXS, :].to(device).float()
        heatmap_pred = model(tactile_left, tactile_right, keypoint_vr)
    heatmap_pred = torch.clip(heatmap_pred, 0, None)
    keypoint_out, _ = softmax(heatmap_pred)
    
    keypoints = denormalize_keypoints(keypoint_out[0, const.UNITY_INDEXS, :]).detach().cpu().numpy().tolist()
    
    # keypoints = denormalize_keypoints(input_kp[0, -1,const.UNITY_INDEXS, :]).numpy().tolist()
    
    action_class = action_idx[0].item()
    return {'keypoints': keypoints, 'action_class': action_class}


def denormalize_keypoints(normalized_keypoints):
    restored_keypoints = normalized_keypoints.clone()
    restored_keypoints[:,:2] -= 0.5
    restored_keypoints *= 2.0
    # x,y좌표 뒤집기
    restored_keypoints[:,0] *= -1
    restored_keypoints[:,1] *= -1
    return restored_keypoints


def main():
    #get config
    config = Tactile2PoseConfig()
    config.BATCH_SIZE = 1
    
    # set dataloader
    data_dir = "D:\\Desktop\\dataset"
    loaders, datasets, mappings = get_tactile_dataloaders(data_dir, config)
    train_dataloader, valid_dataloader, test_dataloader = loaders
    test_data_iterator = iter(test_dataloader)
    
    #load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    
    # config.MODEL = "Tactile2PoseFeatureModel"
    config.MODEL = "Tactile2PoseVRHeatmap"
    # config.MODEL = "Tactile2PoseVRLinear"
    model_path = ".\\models\\best_model_heatmap_lowerbody_weighted.pth"
    model, softmax = load_model(config, model_path, device)
    
    data = next(test_data_iterator)
    model_output_data = get_pose_aciton_data(config, data, model, softmax, device)
    # start server
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((HOST, PORT))
        s.listen()
        print(f"Python 서버가 {PORT}:{PORT}에서 대기 중 입니다.")
        conn, addr = s.accept()
        with conn:
            print(f"{addr}가 연결되었습니다.")
            while True:
                data = conn.recv(4096)
                if not data:
                    break
                
                
                # unity_vr_data = json.loads(data.decode('utf-8'))
                # print(f"Unity에서 받은 vr 좌표 데이터: {unity_vr_data}")
                try:
                    data = next(test_data_iterator)
                except StopIteration:
                    test_data_iterator = iter(test_dataloader)
                    data = next(test_data_iterator)
                model_output_data = get_pose_aciton_data(config, data, model, softmax, device)

                conn.sendall(json.dumps(model_output_data).encode('utf-8'))
                print(f"Unity에 보낼 model 데이터: {model_output_data['keypoints'][0]}, {model_output_data['keypoints'][1]}")
                

if __name__ == "__main__":
    main()
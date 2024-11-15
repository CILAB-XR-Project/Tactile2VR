import os, sys
import socket
import json
import numpy as np


from model import Tactile2PoseFeatureModel
from config import Tactile2PoseConfig
from dataloader import get_tactile_dataloaders



HOST = "127.0.0.1"
PORT = 12345

config = Tactile2PoseConfig()
config.BATCH_SIZE = 1
data_dir = "/app/raid/isaac/InsoleData/dataset/"
loaders, datasets, mappings = get_tactile_dataloaders(data_dir, config)
train_dataloader, valid_dataloader, test_dataloader = loaders


def get_pose_aciton_data():
    data = next(iter(test_dataloader))
    tac_left, tac_right, kp, action_idx = data
    
    keypoints = denormalize_keypoints(kp[0, -1, :, :]).numpy().tolist()
    action_class = action_idx[0].item()
    return {'keypoints': keypoints, 'action_class': action_class}


def denormalize_keypoints(normalized_keypoints):
    restored_keypoints = normalized_keypoints.clone()
    restored_keypoints[:,:2] -= 0.5
    restored_keypoints *= 2
    return restored_keypoints


def main():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((HOST, PORT))
        s.listen()
        print(f"Python 서버가 {PORT}:{PORT}에서 대기 중 입니다.")
        model_output_data = get_pose_aciton_data()
        conn, addr = s.accept()
        with conn:
            print(f"{addr}가 연결되었습니다.")
            while True:
                data = conn.recv(4096)
                if not data:
                    break
                
                
                unity_vr_data = json.loads(data.decode('utf-8'))
                print(f"Unity에서 받은 vr 좌표 데이터: {unity_vr_data}")
                
                model_output_data = get_pose_aciton_data()
                conn.sendall(json.dumps(model_output_data).encode('utf-8'))
                print(f"Unity에 보낼 model 데이터: {model_output_data}")
                

if __name__ == "__main__":
    main()
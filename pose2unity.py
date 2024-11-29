import os, sys
import socket
import json
import numpy as np


from model import Tactile2PoseFeatureModel, Tactile2PoseVRHeatmap, Tactile2PoseVRLinear, SpatialSoftmax3D
from config import Tactile2PoseConfig
from dataloader import get_tactile_dataloaders, generate_heatmap
import const
import torch

from WifiSensor import WifiSensor, align_pressure

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


def get_pose_aciton_data_from_dataset(config, data, model, softmax, device="cuda"):
        
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

def get_pose_aciton_data_from_realdata(config, tactile_data, vr_kps, model, softmax, device="cuda"):
    input_tac_left, input_tac_right = tactile_data
    
    input_kp = torch.tensor(vr_kps).unsqueeze(0).to(device).float()
    input_kp = normalize_keypoints(input_kp)
    tactile_left = input_tac_left.to(device).float()
    tactile_right = input_tac_right.to(device).float()
    
    if config.MODEL == "Tactile2PoseVRHeatmap":
        vr_keypoint_heatmap = generate_heatmap(input_kp)
        heatmap_pred, action_prd = model(tactile_left, tactile_right, vr_keypoint_heatmap)
    
    else:
        heatmap_pred, action_prd = model(tactile_left, tactile_right, input_kp)
    heatmap_pred = torch.clip(heatmap_pred, 0, None)
    keypoint_out, _ = softmax(heatmap_pred)
    keypoints = denormalize_keypoints(keypoint_out[0, const.UNITY_INDEXS, :]).detach().cpu().numpy().tolist()
    
    action_idx = torch.argmax(action_prd, dim=1)
    
    action_class = action_idx[0].item()
    return {'keypoints': keypoints, 'action_class': action_class}


def get_next_test_data(test_data_iterator, test_dataloader):
    try:
        return next(test_data_iterator), test_data_iterator
    except StopIteration:
        test_data_iterator = iter(test_dataloader) 
        return next(test_data_iterator) ,test_data_iterator
    
def get_tactile_data():
    pass

def denormalize_keypoints(normalized_keypoints):
    restored_keypoints = normalized_keypoints.clone()
    restored_keypoints[:,:2] -= 0.5
    restored_keypoints *= 2.0
    # x,y좌표 뒤집기
    restored_keypoints[:,0] *= -1
    restored_keypoints[:,1] *= -1
    return restored_keypoints


def normalize_keypoints(keypoints):
    normalized_keypoints = keypoints.clone()
    normalized_keypoints[:,:2] /= 2.0
    normalized_keypoints[:,:2] += 0.5
    # x,y좌표 뒤집기
    normalized_keypoints[:,0] *= -1
    normalized_keypoints[:,1] *= -1
    return normalized_keypoints

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
   
    
    # config.MODEL = "Tactile2PoseFeatureModel"
    config.MODEL = "Tactile2PoseVRHeatmap"
    # config.MODEL = "Tactile2PoseVRLinear"

    model_path = ".\\models\\best_model_heatmap_lowerbody_weighted.pth"
    model, softmax = load_model(config, model_path, device)
    
    num_client = 2
    sensor = WifiSensor(
        host='192.168.0.2',  # Localhost
        port=7000,  # Port to listen on (non-privileged ports are > 1023)
        num_client=num_client,
    )
    sensor.start()

    for _, sender_ID, _, ts, data_matrix in sensor.get_all():
        if sender_ID == 1:
            data_matrixL = data_matrix
        elif sender_ID == 2:
            data_matrixR = data_matrix
        else:
            raise RuntimeError
    tactile_left, tactile_right = align_pressure(data_matrixL, data_matrixR, Insole_ID)
    #    save_queue.put((tactile_left, tactile_right, timestamp))
    #  normalize(tactile_left, 0, 255, 2900, 3100).astype(np.uint8),    

    keypoint_data, test_data_iterator= get_next_test_data(test_data_iterator, test_dataloader)
    model_output_data = get_pose_aciton_data_from_dataset(config, keypoint_data, model, softmax, device)
    # start server
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((HOST, PORT))
        s.listen()
        print(f"Python 서버가 {PORT}:{PORT}에서 대기 중 입니다.")
        conn, addr = s.accept()
        with conn:
            print(f"{addr}가 연결되었습니다.")
            buffer = ""
            while True:
                data = conn.recv(4096).decode('utf-8')
                if not data:
                    break

                # 받은 데이터를 버퍼에 추가
                buffer += data
                
                
                while "\n" in buffer:
                    message, buffer = buffer.split("\n", 1)  # 첫 번째 메시지와 나머지 데이터 분리
                    try:
                        unity_vr_data = json.loads(message)  # JSON 파싱
                        print(f"Unity에서 받은 VR 좌표 데이터: {unity_vr_data}")
                        
                    
                        # tactile_data = get_tactile_data()
                        #model_output_data = get_pose_aciton_data_from_realdata(config, tactile_data, unity_vr_data, model, softmax, device)
                        
                        # 테스트 데이터 준비
                        keypoint_data, test_data_iterator = get_next_test_data(test_data_iterator, test_dataloader)

                        # 모델 데이터 생성
                        model_output_data = get_pose_aciton_data_from_dataset(config, keypoint_data, model, softmax, device)

                        # Unity로 데이터 전송
                        conn.sendall((json.dumps(model_output_data) + "\n").encode('utf-8'))
                        print(f"Unity에 보낼 Model 데이터: {model_output_data['keypoints'][0]}, {model_output_data['keypoints'][1]}")
                    except json.JSONDecodeError as e:
                        print(f"JSON 디코드 에러: {e}")
                    except Exception as e:
                        print(f"예외 발생: {e}")

    
if __name__ == "__main__":
    main()
import socket
import multiprocessing as mp
import time
import json
import numpy as np
import torch

from config import Tactile2PoseConfig
from const import UNITY_INDEXS, VR_INDEXS
from model import SpatialSoftmax3D, Tactile2PoseVRHeatmap
from dataloader import get_tactile_dataloaders, generate_heatmap
from WifiSensor import WifiSensor, align_pressure

class UnityCommunicator:
    def __init__(self, model, host, unity_port, tactile_sensor) -> None:
        self.host = host
        self.port = unity_port
        self.model = model
        self.device = next(self.model.parameters()).device
        self.softmax = SpatialSoftmax3D(20, 20, 18, 19).to(self.device)
        self.tactile_sensor = tactile_sensor
        
    def _denormalize_keypoints(self, normalized_keypoints) -> torch.Tensor:
        restored_keypoints = normalized_keypoints.clone()
        restored_keypoints[:,:2] -= 0.5
        restored_keypoints *= 2.0
        # x,y좌표 뒤집기
        restored_keypoints[:,0] *= -1
        restored_keypoints[:,1] *= -1
        return restored_keypoints

    
    def _normalize_keypoints(self, keypoints) -> torch.Tensor:
        normalized_keypoints = keypoints.clone()
        normalized_keypoints[:,0] *= -1
        normalized_keypoints[:,1] *= -1
        
        normalized_keypoints /= 2.0
        normalized_keypoints[:,:2] += 0.5
        return normalized_keypoints
    
    def model_inference(self, tactile_left, tactile_right, vr_kps) -> dict:
        vr_kps_heatmap = generate_heatmap(vr_kps)
        heatmap_pred, action_prd = self.model(tactile_left, tactile_right, vr_kps_heatmap)

        heatmap_pred = torch.clip(heatmap_pred, 0, None)
        keypoint_out, _ = self.softmax(heatmap_pred)
        keypoints = self._denormalize_keypoints(keypoint_out[0, UNITY_INDEXS, :]).detach().cpu().numpy().tolist()
    
        action_idx = torch.argmax(action_prd, dim=1)
        
        action_class = action_idx[0].item()
        return {'keypoints': keypoints, 'action_class': action_class}

        
    def run_with_testdata(self) -> None:
        def _load_test_data():
            config = Tactile2PoseConfig()
            config.BATCH_SIZE = 1
    
            data_dir = "D:\\Desktop\\dataset"
            loaders, _, _ = get_tactile_dataloaders(data_dir, config)
            _,_, test_dataloader = loaders
            test_data_iterator = iter(test_dataloader)
            return test_dataloader, test_data_iterator
        def _get_next_data(test_data_iterator, test_dataloader):
            try:
                data = next(test_data_iterator)
            except StopIteration:
                test_data_iterator = iter(test_dataloader) 
                data = next(test_data_iterator)
                
            tactile_left, tactile_right, keypoint_data, _ = data
            tactile_left = tactile_left.to(self.device).float()
            tactile_right = tactile_right.to(self.device).float()
            keypoint_vr = keypoint_data[:,-1, VR_INDEXS, :].to(self.device).float()
            return (tactile_left, tactile_right, keypoint_vr), test_data_iterator
        
        data_loader, data_iterator = _load_test_data()
        (tactile_left, tactile_right, keypoint_vr), data_iterator = _get_next_data(data_iterator, data_loader)
        self.model_inference(tactile_left, tactile_right, keypoint_vr)
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as unity_socket:
            unity_socket.bind((self.host, self.port))
            unity_socket.listen(1)
            print(f"Python 서버가 {self.host}:{self.port}에서 대기 중 입니다.")
            conn, addr = unity_socket.accept()
            print(f"{addr}, Unity가 연결되었습니다.")
            
            buffer = ""
            while True:
                data = conn.recv(1024).decode('utf-8')
                if not data:
                    print("No data")
                    break
                
                buffer += data
                
                while "\n" in buffer:
                    message, buffer = buffer.split("\n", 1)  
                    try:
                        unity_vr_kps_str = json.loads(message)
                        unity_vr_kps = list(list(map(float, kp)) for kp in unity_vr_kps_str)
                        print(f"Unity에서 받은 VR 좌표 데이터: {unity_vr_kps}")
                        
                        keypoint_data = torch.tensor(unity_vr_kps).unsqueeze(0).to(self.device).float()
                        keypoint_data = self._normalize_keypoints(keypoint_data)

                        (tactile_left, tactile_right, keypoint_vr), data_iterator = _get_next_data(data_iterator, data_loader)
                              
                        model_output_data = self.model_inference(tactile_left, tactile_right, keypoint_data)
                        print(f"Unity에 보낼 Model 데이터: {model_output_data['keypoints'][0]}, {model_output_data['action_class']}")
                        conn.sendall((json.dumps(model_output_data) + "\n").encode('utf-8'))
                        
                    except json.JSONDecodeError as e:
                        print(f"JSON 디코드 에러: {e}")
                    except Exception as e:
                        print(f"예외 발생: {e}")
                
                
    def run_with_realtime(self) -> None:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as unity_socket:
            unity_socket.bind((self.host, self.port))
            unity_socket.listen(1)
            print(f"Python 서버가 {self.host}:{self.port}에서 대기 중 입니다.")
            conn, addr = unity_socket.accept()
            print(f"{addr}, Unity가 연결되었습니다.")
            
            buffer = ""
            while True:
                data = conn.recv(1024).decode('utf-8')
                if not data:
                    print("No data")
                    break
                
                buffer += data
                
                while "\n" in buffer:
                    message, buffer = buffer.split("\n", 1)  
                    try:
                        unity_vr_kps_str = json.loads(message)
                        unity_vr_kps = list(list(map(float, kp)) for kp in unity_vr_kps_str)
                        print(f"Unity에서 받은 VR 좌표 데이터: {unity_vr_kps}")
                        
                        keypoint_data = torch.tensor(unity_vr_kps).unsqueeze(0).to(self.device).float()
                        keypoint_data = self._normalize_keypoints(keypoint_data)
                        
                        #큐에서 데이터 가져오기.
                        tactile_left, tactile_right = self.tactile_sensor.get_window_data()
                        tactile_left = torch.tensor(tactile_left).to(self.device).float()
                        tactile_right = torch.tensor(tactile_right).to(self.device).float()
                        
                        model_output_data = self.model_inference(tactile_left, tactile_right, keypoint_data)

                        conn.sendall((json.dumps(model_output_data) + "\n").encode('utf-8'))
                        print(f"Unity에 보낼 Model 데이터: {model_output_data['keypoints'][0]}, {model_output_data['action_class']}")
                    except json.JSONDecodeError as e:
                        print(f"JSON 디코드 에러: {e}")
                    except Exception as e:
                        print(f"예외 발생: {e}")
    
    
    
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    
    # Load model
    config = Tactile2PoseConfig()
    model = Tactile2PoseVRHeatmap(config)
    model_path = ".\\models\\best_model_action.pth"
    try:
        model.load_state_dict(torch.load(model_path).state_dict())
    except:
        model = torch.nn.DataParallel(model)
        model.load_state_dict(torch.load(model_path).state_dict())
    model.to(device)
    
    num_client = 2
    sensor = WifiSensor(
        # host='192.168.0.2',  # Localhost
        host='127.0.0.1',
        port=7000,  # Port to listen on (non-privileged ports are > 1023)
        num_client=num_client,
        insole_ID=1,
        window_size=20
    )
    
    unity_communicator = UnityCommunicator(model, '127.0.0.1', 12345, sensor)
    
    # sensor.start()
    
    unity_communicator.run_with_testdata()
    
    
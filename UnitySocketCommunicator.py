import socket
import multiprocessing as mp
import time
import json
import numpy as np
import torch
from collections import deque
import cv2

from config import Tactile2PoseConfig
from const import UNITY_INDEXS, VR_INDEXS
from model import SpatialSoftmax3D, Tactile2PoseVRHeatmap
from dataloader import get_tactile_dataloaders, generate_heatmap
from WifiSensor import WifiSensor, align_pressure, minmax_normalization

def normalize(array, alpha, beta, minv, maxv):
    new_array = (array - minv) / (maxv - minv)
    new_array = np.clip(new_array, 0, 1)
    new_array *= beta - alpha
    new_array += alpha
    return new_array
def visualize_insole(tactile_left, tactile_right, fps=None):
    img = np.concatenate((
        # cv2.normalize(tactile_left, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8),
        # cv2.normalize(tactile_right, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        normalize(tactile_left, 0, 255, 2900, 3100).astype(np.uint8),
        normalize(tactile_right, 0, 255, 2900, 3100).astype(np.uint8)
    ), axis=1)

    img = cv2.resize(img.astype(np.uint8), (160 * 2, 160 * 3),
                     interpolation=cv2.INTER_NEAREST)  # Scale up for better visibility

    # Convert image to color (grayscale to BGR)
    img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)


    # Draw a red vertical line in the middle to separate left and right
    mid_x = img_color.shape[1] // 2
    cv2.line(img_color, (mid_x, 0), (mid_x, img_color.shape[0]), (0, 0, 255), 2)

    if fps != None:
        # Put the timestamp text on the image
        cv2.putText(img_color, f'FPS: {fps:.1f}', (10, img_color.shape[0] - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1,
                    cv2.LINE_AA)

    # Put the labels "Left" and "Right" on the image
    cv2.putText(img_color, 'Left', (mid_x // 2 - 30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(img_color, 'Right', (mid_x + mid_x // 2 - 30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2,
                cv2.LINE_AA)

    return img_color

class UnityCommunicator:
    def __init__(self, model, host, unity_port, tactile_sensor, window_size=20) -> None:
        self.host = host
        self.port = unity_port
        self.model = model
        self.device = next(self.model.parameters()).device
        self.softmax = SpatialSoftmax3D(20, 20, 18, 19).to(self.device)
        self.tactile_sensor = tactile_sensor
        self.left_tactile_window = deque(maxlen=window_size)
        self.right_tactile_window = deque(maxlen=window_size)

        self.tactile_min = float("inf") 
        self.tactile_max = float("-inf") 
        
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
        # normalized_keypoints[:,0] *= -1
        # normalized_keypoints[:,1] *= -1
        
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
            start_time = time.time()
            fps=0
            
            frame_count =0 
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
                    for _, sender_ID, _, ts, data_matrix in sensor.get_all():
                        if sender_ID == 1:
                            data_matrixL = data_matrix
                        elif sender_ID == 2:
                            data_matrixR = data_matrix
                        else:
                            raise RuntimeError

                    tactile_left, tactile_right = align_pressure(data_matrixL, data_matrixR, self.tactile_sensor.insole_ID) 
                    # img_color = visualize_insole(tactile_left, tactile_right, fps)

                    tactile_val = np.concatenate((tactile_left, tactile_right),axis=0)
                    if self.tactile_min > np.min(tactile_val):
                        self.tactile_min = np.min(tactile_val)
                    if self.tactile_max < np.max(tactile_val):
                        self.tactile_max = np.max(tactile_val)

                    tactile_left = minmax_normalization(tactile_left, self.tactile_min, self.tactile_max)
                    tactile_right = minmax_normalization(tactile_right, self.tactile_min, self.tactile_max)
                    # tactile_left = minmax_normalization(tactile_left)
                    # tactile_right = minmax_normalization(tactile_right)

                    #큐에서 데이터 가져오기.
                    self.left_tactile_window.append(tactile_left)
                    self.right_tactile_window.append(tactile_right)
                    if len(self.right_tactile_window) < 20 or len(self.left_tactile_window) < 20:
                        print("모으는중")
                        continue

                    tactile_left = torch.tensor(list(self.left_tactile_window)).unsqueeze(0).to(self.device).float()
                    tactile_right = torch.tensor(list(self.right_tactile_window)).unsqueeze(0).to(self.device).float()

                    unity_vr_kps_str = json.loads(message)
                    unity_vr_kps = list(list(map(float, kp)) for kp in unity_vr_kps_str)
                    # print(f"Unity에서 받은 VR 좌표 데이터: {unity_vr_kps}")
                    
                    keypoint_data = torch.tensor(unity_vr_kps).unsqueeze(0).to(self.device).float()
                    keypoint_data = self._normalize_keypoints(keypoint_data)
                    
                    model_output_data = self.model_inference(tactile_left, tactile_right, keypoint_data)

                    conn.sendall((json.dumps(model_output_data) + "\n").encode('utf-8'))
                    # print(f"Unity에 보낼 Model 데이터: {model_output_data['keypoints'][0]}, {model_output_data['action_class']}")
                except json.JSONDecodeError as e:
                    print(f"JSON 디코드 에러: {e}")
                except Exception as e:
                    print(f"예외 발생: {e}")
                    pass
                # cv2.imshow("Pressure Matrix Visualization", img_color)

                # if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit the loop
                #     break
                    
                frame_count += 1
                # print(frame_count)
                current_time = time.time()
                elapsed_time = current_time - start_time

                if elapsed_time > 1:
                    fps = frame_count / elapsed_time
                    print(f"FPS: {fps:.2f}")
                    frame_count = 0
                    start_time = time.time()
    
    
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
        host='192.168.0.2',  # Localhost
        # host='127.0.0.1',
        port=7000,  # Port to listen on (non-privileged ports are > 1023)
        num_client=num_client,
        insole_ID=1
    )
    
    unity_communicator = UnityCommunicator(model, '127.0.0.1', 12345, sensor, window_size=20)
    
    sensor.start()
    
    # unity_communicator.run_with_testdata()
    unity_communicator.run_with_realtime()
    
import os, sys
import socket
import json
import numpy as np


from model import Tactile2PoseFeatureModel
from config import Tactile2PoseConfig



HOST = "127.0.0.1"
PORT = 12345


def get_pose_aciton_data():
    # 일단 랜덤값 전송
    keypoints = np.random.rand(19, 3).tolist()  # 19개의 3D 좌표
    action_class = np.random.randint(0, 4)      # 4개의 action class 중 하나 선택
    return {'keypoints': keypoints, 'action_class': action_class}



def main():
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
                
                
                unity_vr_data = json.loads(data.decode('utf-8'))
                print(f"Unity에서 받은 vr 좌표 데이터: {unity_vr_data}")
                
                model_output_data = get_pose_aciton_data()
                conn.sendall(json.dumps(model_output_data).encode('utf-8'))
                print(f"Unity에 보낼 model 데이터: {model_output_data}")
                

if __name__ == "__main__":
    main()
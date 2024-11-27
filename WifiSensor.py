import socket
import numpy as np
import cv2
import time
import multiprocessing as mp
from multiprocessing import Manager

def normalize(array, alpha, beta, minv, maxv):
    new_array = (array - minv) / (maxv - minv)
    new_array = np.clip(new_array, 0, 1)
    new_array *= beta - alpha
    new_array += alpha
    return new_array

def align_pressure(tactile_left, tactile_right, Insole_ID):
    insole_shape = [22, 32]
    # 270 size
    if Insole_ID == 1:
        tactile_left = tactile_left[:insole_shape[1], :insole_shape[0]]
        tactile_right = tactile_right[:insole_shape[1], -insole_shape[0]:]

        tactile_left = np.flip(tactile_left, axis=1)
        tactile_right = np.flip(tactile_right, axis=1)
    # 240 size
    elif Insole_ID == 2:
        tactile_left = np.rot90(tactile_left, k=1, axes=(1, 0))
        tactile_left = np.flip(tactile_left, axis=1)
        tactile_left = tactile_left[:insole_shape[1], -insole_shape[0]:]

        # tactile_right = np.rot90(tactile_right, k=1, axes=(1, 0))
        tactile_right = np.flip(tactile_right, axis=1)
        tactile_right = tactile_right[:insole_shape[1], -insole_shape[0]:]

    return tactile_left, tactile_right

def parse_data(data):
    assert len(data)==2051
    data_shape = (32, 32)
    matrix_index = data[0] - (ord('\n') + 1)
    sender_ID = data[1] - (ord('\n') + 1)
    _sensor_bitshift = 6
    data_matrix = data[2:-1]  # ignore the newline character at the end
    data_matrix = np.frombuffer(data_matrix, dtype=np.uint8).astype(np.uint16)

    data_matrix = data_matrix - (ord('\n') + 1)
    data_matrix = data_matrix[0::2] * (2 ** _sensor_bitshift) + data_matrix[1::2]
    data_matrix = data_matrix.reshape(data_shape)
    return matrix_index, sender_ID, data_matrix

class WifiSensor:
    def __init__(self, host, port, num_client):
        self.host = host
        self.port = port
        self.num_client = num_client

        self.fps = 0
        self.exit = mp.Event()
        self.queues = [Manager().Queue() for _ in range(self.num_client)]

    def close(self):
        self.exit.set()
        for p in self.processes:
            p.join()

    def get_all(self):
        result = []
        for i in range(self.num_client):
            re = self.get(i)
            result.append(re)
        return result
    def get(self, idx):
        result = None
        if self.queues[idx].empty():
            result = self.queues[idx].get()
        else:
            while not self.queues[idx].empty():
                result = self.queues[idx].get()
        return result

    def start(self):
        # Create a TCP/IP socket
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_socket:
            server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            # Bind the socket to the address and port
            server_socket.bind((self.host, self.port))
            print(f"Server started and listening on {self.host}:{self.port}")

            # Listen for incoming connections (can queue up to NUM_SENSORS requests)
            server_socket.listen(self.num_client)

            processes = []
            for i in range(self.num_client):
                print(f"{len(processes)} clients are connected. Waiting for {self.num_client - len(processes)} more")
                conn, client_address = server_socket.accept()
                print(f"Connection found from {client_address}")
                p = mp.Process(target=self.receive_data, args=(conn, self.queues[i]))
                p.start()
                processes.append(p)
                print(f"Start process {i}")

            server_socket.setblocking(False)
            print(f"{len(processes)} clients are connected.")
            self.processes = processes

    def receive_data(self, conn, queue):
        data_length = 2051

        # Receive data from the client (buffer size 1024 bytes)
        total_data = b''
        frame_count = 0
        fps = 0
        start_time = time.time()  # Start time for FPS calculation

        with conn:
            while not self.exit.is_set():
                while len(total_data) < data_length:
                    total_data += conn.recv(data_length)
                timestamp = time.time()
                data = total_data[:data_length]
                total_data = total_data[data_length:]

                matrix_index, sender_ID, data_matrix = parse_data(data)
                queue.put((matrix_index, sender_ID, fps, timestamp, data_matrix))

                # FPS calculation
                frame_count += 1
                elapsed_time = time.time() - start_time

                if elapsed_time >= 1.0:  # Every second
                    fps = frame_count / elapsed_time
                    frame_count = 0  # Reset the frame count
                    start_time = time.time()  # Reset the start time

if __name__ == "__main__":
    num_client = 2
    sensor = WifiSensor(
        host='192.168.0.2',  # Localhost
        port=7000,  # Port to listen on (non-privileged ports are > 1023)
        num_client=num_client,
    )
    sensor.start()

    fps = 0
    frame_count = 0
    start_time = time.time()
    while True:
        data_matrixs = []
        for i in range(num_client):
            matrix_index, sender_ID, sens_fps, ts, data_matrix = sensor.get(i)
            print(ts)
            data_matrixs.append(data_matrix)
        data_matrix = np.concatenate(data_matrixs, axis=1)

        # Normalize the data for visualization (scale to 8-bit image)
        data_matrix_normalized = cv2.normalize(data_matrix, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        # Resize the data matrix to 640x640 pixels
        data_matrix_resized = cv2.resize(data_matrix_normalized, (320 * num_client, 320), interpolation=cv2.INTER_LINEAR)
        # FPS calculation
        frame_count += 1
        elapsed_time = time.time() - start_time

        if elapsed_time >= 1.0:  # Every second
            fps = frame_count / elapsed_time
            print(f"FPS: {fps:.2f}")
            frame_count = 0  # Reset the frame count
            start_time = time.time()  # Reset the start time

        # Display FPS on the image
        fps_text = f"FPS: {fps:.2f}"
        cv2.putText(data_matrix_resized, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Show the image with FPS
        cv2.imshow("Sensor Data with FPS", data_matrix_resized)

        # Check if 'q' is pressed to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Clean up OpenCV windows
    cv2.destroyAllWindows()

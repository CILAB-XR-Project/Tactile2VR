import numpy as np
import torch
from tqdm import tqdm
import cv2
from config import Tactile2PoseConfig
from dataloader import get_tactile_dataloaders
from model import Tactile2PoseFeatureModel, SpatialSoftmax3D
from visualize import plotMultiKeypointVideo, plotTactileVideo, plot3DheatmapVideo
import matplotlib.pyplot as plt
import gc
import os
from torch.utils.data import DataLoader, Subset
from const import VR_INDEXS, ACTIVITY_LIST
from PIL import Image

def main(load_dir):
    config = Tactile2PoseConfig()
    config.load_from_json(os.path.join(load_dir, "config.json"))


    model = Tactile2PoseFeatureModel(config)
    frame_num_by_file = 15
    video_batch_size = 32

    softmax = SpatialSoftmax3D(20, 20, 18, 19)
   
    data_dir = "/app/raid/isaac/InsoleData/dataset/"
    loaders, datasets, mappings = get_tactile_dataloaders(data_dir, config)
    train_dataloader, valid_dataloader, test_dataloader = loaders
    train_dataset, valid_dataset, test_dataset = datasets
    train_mapping, valid_mapping, test_mapping = mappings


    # create file mappings

    test_file_mappings = []
    for start, end, filename in test_mapping:
        testd_end = min(end, start+frame_num_by_file)
        test_file_mappings.append((start, testd_end, filename.split("/")[-1]))


    print(f"Filenum Test: {len(test_file_mappings)}")

    # GPU 사용 설정
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    softmax.to(device)

    #load model
    try:
        model.load_state_dict(torch.load(os.path.join(load_dir, "best_model.pth")).state_dict())
    except:
        model = torch.nn.DataParallel(model)
        model.load_state_dict(torch.load(os.path.join(load_dir, "best_model.pth")).state_dict())
    print(f"loaded model from {load_dir}")

    make_video_by_file(get_prediction(model, test_dataset, softmax, test_file_mappings, config), os.path.join(load_dir, "pred_img"), video_batch_size)

def get_prediction(model, dataset, softmax, file_mappings, config, device="cuda"):
    model.eval()

    frame_num = sum([e-s for s, e, _ in file_mappings])

    result = []
    with torch.no_grad():
        pbar = tqdm(total=frame_num, desc="Running inference")
        for f_start, f_end, filename in file_mappings:
            subset = Subset(dataset, range(f_start, f_end))
            data_loader = DataLoader(subset, batch_size=config.BATCH_SIZE, shuffle=False)

            total_tactile_left = []
            total_tactile_right = []
            total_heatmap_pred = []
            total_keypoint = []
            total_keypoint_out = []

            for i, (input_tac_left, input_tac_right, input_kp, _) in enumerate(data_loader):
                tactile_left = input_tac_left.to(device).float()
                tactile_right = input_tac_right.to(device).float()
                keypoint_vr = input_kp[:,:,VR_INDEXS, :].to(device).float()
                keypoint_label =  input_kp[:,-1,:,:].to(device).float()
                
                heatmap_pred = model(tactile_left, tactile_right, keypoint_vr)
                heatmap_pred = torch.clip(heatmap_pred, 0, None)
                # get keypoint prediction
                keypoint_out, _ = softmax(heatmap_pred)

                total_tactile_left.append(tactile_left.detach().cpu().numpy()[:, -1])
                total_tactile_right.append(tactile_right.detach().cpu().numpy()[:, -1])
                total_heatmap_pred.append(heatmap_pred.detach().cpu().numpy())
                total_keypoint.append(keypoint_label.detach().cpu().numpy())
                total_keypoint_out.append(keypoint_out.detach().cpu().numpy())

                pbar.update(keypoint_label.shape[0])


            total_tactile_left = np.concatenate(total_tactile_left, axis=0)
            total_tactile_right = np.concatenate(total_tactile_right, axis=0)
            total_heatmap_pred = np.concatenate(total_heatmap_pred, axis=0)
            total_keypoint = np.concatenate(total_keypoint, axis=0)
            total_keypoint_out = np.concatenate(total_keypoint_out, axis=0)

            data_tuple = (total_tactile_left, total_tactile_right, total_heatmap_pred, total_keypoint, total_keypoint_out)
            result.append((filename, data_tuple))
    return result

def make_video_by_file(data, save_dir, batch_size):
    for i, (filename, data_tuple) in enumerate(data):
        print(f"Making img for file {filename}, {i+1}/{len(data)}")
        save_path = os.path.join(save_dir, filename)
        os.makedirs(save_path, exist_ok=True)
        make_img(data_tuple, save_path, batch_size)

def make_img(data, path, batch_size):
    total_tactile_left, total_tactile_right, total_heatmap_pred, total_keypoint, total_keypoint_out = data
    data_len = total_tactile_left.shape[0]

    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    out = cv2.VideoWriter(path, fourcc, 15, (640 * 6, 480))

    for i in range(0, data_len, batch_size):
        start, end = i, min(data_len, i + batch_size)

        images1 = plotTactileVideo(total_tactile_left)
        images2 = plotTactileVideo(total_tactile_right)
        images3 = plotMultiKeypointVideo(total_keypoint[start:end, np.newaxis, :, :], limit=[(0, 1), (0, 1), (0, 1)])
        images4 = plotMultiKeypointVideo(total_keypoint_out[start:end, np.newaxis, :, :], limit=[(0, 1), (0, 1), (0, 1)])
        images5 = plot3DheatmapVideo(total_heatmap_pred[start:end])
        
        for idx, (img1, img2, img3, img4, img5) in enumerate(zip(images1, images2, images3, images4, images5)):
            Image.fromarray(img1, 'RGB').save(f'{path}/{idx}_tactile_left.png')
            Image.fromarray(img2, 'RGB').save(f'{path}/{idx}_tactile_right.png')
            Image.fromarray(img3, 'RGB').save(f'{path}/{idx}_label_kps.png')
            Image.fromarray(img4, 'RGB').save(f'{path}/{idx}_pred_kps.png')
            Image.fromarray(img5, 'RGB').save(f'{path}/{idx}_pred_heatmap.png')

        del images1, images2, images3, images4
        gc.collect()
    print(f"Saved video in {path}")

if __name__=="__main__":
    main(load_dir="./runs/Nov13_10-37-10_Main_1")

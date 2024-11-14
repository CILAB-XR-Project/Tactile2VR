import matplotlib.pyplot as plt
from const import BODY_18_PAIRS, BODY_25_color, ACTIVITY_LIST
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from tqdm import tqdm
import seaborn as sns
from matplotlib.patches import Rectangle

def plotMultiKeypoint(keypoints, limit=None):
    fig = plt.figure()

    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

    if limit != None:
        ax.set_xlim(*limit[0])
        ax.set_ylim(*limit[1])
        ax.set_zlim(*limit[2])

    ax.view_init(-149, 33)

    for i in range(len(keypoints)):
        data = keypoints[i]

        xs = data[:, 0]
        ys = data[:, 1]
        zs = -data[:, 2]
        z_offset = (zs.max() + zs.min())  # z축 중앙값 계산
        zs -= z_offset  # z축 중앙으로 평행 이동
    
        for i in range(len(BODY_18_PAIRS)):
            index_1 = BODY_18_PAIRS[i][0]
            index_2 = BODY_18_PAIRS[i][1]

            xs_line = [xs[index_1], xs[index_2]]
            ys_line = [ys[index_1], ys[index_2]]
            zs_line = [zs[index_1], zs[index_2]]
            ax.plot3D(xs_line, ys_line, zs_line, color=BODY_25_color[i] / 255.0)

        ax.scatter(xs, ys, zs, s=20, c=BODY_25_color[:19] / 255.0)

    fig.canvas.draw()
    # plt.show()
    frame = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    img = frame.reshape(fig.canvas.get_width_height()[::-1] + (3,))[..., ::-1]
    
    ax.clear()
    plt.close()
    return img


def plotMultiKeypointVideo(keypoints, limit=None, name=None):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    canvas = FigureCanvas(fig)

    images = []

    for frame_data in tqdm(keypoints, desc="plotMultiKeypointVideo"):
        ax.clear()
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        if limit is not None:
            ax.set_xlim(*limit[0])
            ax.set_ylim(*limit[1])
            ax.set_zlim(*limit[2])

        ax.view_init(-149, 33)

        for kp in frame_data:
            xs = kp[:, 0]
            ys = kp[:, 1]
            zs = -kp[:, 2]
            z_offset = (zs.max() + zs.min())  # z축 중앙값 계산
            zs -= z_offset  # z축 중앙으로 평행 이동

            for i, pair in enumerate(BODY_18_PAIRS):
                index_1, index_2 = pair
                ax.plot3D([xs[index_1], xs[index_2]],
                          [ys[index_1], ys[index_2]],
                          [zs[index_1], zs[index_2]],
                          color=BODY_25_color[i] / 255.0)

            ax.scatter(xs, ys, zs, s=20, c=BODY_25_color[:len(kp)] / 255.0)

        # Draw the canvas, cache the buffer
        if name is not None:
            ax.set_title(name)
        canvas.draw()
        image = np.frombuffer(canvas.tostring_rgb(), dtype='uint8')
        image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))

        images.append(image)

    ax.clear()
    plt.close()
    return images

def plotTactile(data):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.imshow(data)
    fig.canvas.draw()
    frame = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    img = frame.reshape(fig.canvas.get_width_height()[::-1] + (3,))[..., ::-1]
    
    ax.clear()
    plt.close()
    return img

def plotTactileVideo(data_sequence, name=None):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    canvas = FigureCanvas(fig)

    images = []

    for data in tqdm(data_sequence, desc="plotTactileVideo"):
        ax.clear()
        im = ax.imshow(data, cmap='viridis')
        if name is not None:
            ax.set_title(name)
        # Draw the canvas and convert to an image
        canvas.draw()
        image = np.frombuffer(canvas.tostring_rgb(), dtype='uint8')
        image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))

        images.append(image)

    ax.clear()
    plt.close()
    return images

def plot3Dheatmap(_data):
    data = round_to_1(_data, 2)
    colors = ['Greys', 'Purples', 'Blues', 'Greens', 'Oranges',
              'Reds', 'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd',
              'RdPu', 'BuPu', 'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn',
              'BuGn', 'YlGn', 'Greys', 'Purples', 'Blues']

    fig = plt.figure()
    # ax = fig.gca(projection='3d')
    ax = fig.add_subplot(projection='3d')

    ax.clear()
    ax.set_xlim(0, data.shape[1])
    ax.set_ylim(0, data.shape[2])
    ax.set_zlim(0, data.shape[3])
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.view_init(-149, 33)

    for j in range(19):
        frame = data[j]
        x,y,z = np.where(frame>0)
        ax.scatter(x, y, z, c=frame[x,y,z]*255, cmap=colors[j])

    fig.canvas.draw()
    frame = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    img = frame.reshape(fig.canvas.get_width_height()[::-1] + (3,))[..., ::-1]

    ax.clear()
    plt.close()
    return img

def plot3DheatmapVideo(_data):
    data = round_to_1(_data, 2)
    colors = ['Greys', 'Purples', 'Blues', 'Greens', 'Oranges',
              'Reds', 'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd',
              'RdPu', 'BuPu', 'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn',
              'BuGn', 'YlGn', 'Greys', 'Purples', 'Blues']

    fig = plt.figure(dpi=300)
    # ax = fig.gca(projection='3d')
    ax = fig.add_subplot(projection='3d')

    images = []
    for i in tqdm(range(len(data)), desc="plot3DheatmapVideo"):
        ax.clear()
        ax.set_xlim(0, data.shape[1])
        ax.set_ylim(0, data.shape[2])
        ax.set_zlim(0, data.shape[3])
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        ax.view_init(-149, 33)

        for j in range(19):
            frame = data[i, j]
            x,y,z = np.where(frame>0)
            ax.scatter(x, y, z, c=frame[x,y,z]*255, cmap=colors[j])

        fig.canvas.draw()
        frame = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
        img = frame.reshape(fig.canvas.get_width_height()[::-1] + (3,))[..., ::-1]

        images.append(img)
    ax.clear()
    plt.close()
    return images

def round_to_1(data, sig):
    # Create a mask for values significantly greater than zero
    significant_mask = data > 1e-2

    # Calculate the scale factor for rounding based on `sig`
    scales = np.floor(np.log10(np.abs(data[significant_mask]))).astype(int)
    precision = sig - scales - 1

    # Apply rounding to each element according to its calculated precision
    # We need to handle precision per-element
    rounded_values = np.array(
        [np.around(val, dec) if dec >= 0 else np.around(val, 0) for val, dec in zip(data[significant_mask], precision)])
    data[significant_mask] = rounded_values

    # Set very small values to zero
    data[data <= 1e-2] = 0

    return data


def plot_action_confusion_matrix(class_history):
    max_sum = np.max(np.sum(class_history, axis=1))
    ACTIVITY_LIST = [
        "Squat",
        "Warm-up",
        "Walking",  # walking should be in front of other walking variants
        "Walking-in-place",
        "Side walking",
        "Backward walking",
        "Lunge",
    ]
    fig, ax = plt.subplots(dpi=300)
    sns.heatmap(class_history, annot=True, fmt="d", cmap="viridis",
                xticklabels=ACTIVITY_LIST, yticklabels=ACTIVITY_LIST,
                ax=ax, square=True, vmax=max_sum, annot_kws={"weight": "bold"})
    
    height, width = class_history.shape
    # 외곽 테두리 (좌상단부터 우하단까지의 직사각형)
    ax.add_patch(Rectangle((0, 0), width, height, fill=False, edgecolor='black', lw=3))  

    cbar = ax.collections[0].colorbar
    if max_sum % 5 == 0:
        cbar.set_ticks(np.linspace(0, max_sum, num=6))
    else:
        cbar.set_ticks(np.linspace(0, max_sum, num=5))
    cbar.outline.set_edgecolor('black')  # 컬러바 테두리 색상 설정
    cbar.outline.set_linewidth(1) 
    cbar.ax.yaxis.label.set_weight('bold')
    for label in cbar.ax.get_yticklabels():
        label.set_fontweight('bold')
    
    plt.xlabel('Predicted Label', fontsize=15)
    plt.ylabel('True Label', fontsize=15)
    plt.xticks(rotation=25, ha='right', fontsize=13)  # x축 레이블 회전 및 정렬
    plt.yticks(rotation=0, fontsize=13)  # y축 레이블 정렬
    plt.tight_layout() 
    
    fig.canvas.draw()  
    img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)  
    img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))  
    
    plt.close()
    return img


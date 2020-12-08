import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm

def Video2Npy(addr):
    vidcap = cv2.VideoCapture(addr)
    count = 0
    frames=[]
    success=True
    while success and count<20:
        vidcap.set(cv2.CAP_PROP_POS_MSEC,(count*250))
        success,frame= vidcap.read()
        frame = cv2.resize(frame,(224,224),    interpolation=cv2.INTER_AREA)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = np.reshape(frame, (224,224,3))
        frames.append(frame)
        count += 1
    frames = np.array(frames)
    vidcap.release()
    return frames

def Save2Npy(file_dir, save_dir):
    """Transfer all the videos and save them into specified directory
    Args:
        file_dir: source folder of target videos
        save_dir: destination folder of output .npy files
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    # List the files
    videos = os.listdir(file_dir)
    for v in tqdm(videos):
        # Split video name
        video_name = v.split('.')[0]
        # Get src
        video_path = os.path.join(file_dir, v)
        # Get dest
        save_path = os.path.join(save_dir, video_name + '.npy')
        # Load and preprocess video
        data = Video2Npy(addr=video_path)
        data = np.uint8(data)
        # Save as .npy file
        np.save(save_path, data)

    return None

source_path = os.getcwd() + "/RWF-2000"
target_path = os.getcwd() + "/RWF-2000_simple_preprocessed"

for f1 in ['train', 'val']:
    for f2 in ['Fight', 'NonFight']:
        path1 = os.path.join(source_path, f1, f2)
        path2 = os.path.join(target_path, f1, f2)
        Save2Npy(file_dir=path1, save_dir=path2)



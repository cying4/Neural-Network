import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
#%%
def getOpticalFlow(video):
    gray_video = []
    for i in range(len(video)):
        img = cv2.cvtColor(video[i], cv2.COLOR_RGB2GRAY)
        gray_video.append(np.reshape(img, (50, 50, 1)))
    flows = []
    for i in range(0, len(video) - 1):
        flow = cv2.calcOpticalFlowFarneback(gray_video[i], gray_video[i + 1], None, 0.5, 3, 15, 3, 5, 1.2,
                                            cv2.OPTFLOW_FARNEBACK_GAUSSIAN)
        flow[..., 0] -= np.mean(flow[..., 0])
        flow[..., 1] -= np.mean(flow[..., 1])
        flow[..., 0] = cv2.normalize(flow[..., 0], None, 0, 255, cv2.NORM_MINMAX)
        flow[..., 1] = cv2.normalize(flow[..., 1], None, 0, 255, cv2.NORM_MINMAX)
        flows.append(flow)
    flows.append(np.zeros((50, 50, 2)))
    return np.array(flows, dtype=np.float32)
def Video2Npy(file_path, resize=(50, 50)):
    cap = cv2.VideoCapture(file_path)
    len_frames = int(cap.get(7))
    try:
        frames = []
        for i in range(len_frames - 1):
            _, frame = cap.read()
            frame = cv2.resize(frame, resize, interpolation=cv2.INTER_AREA)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = np.reshape(frame, (50, 50, 3))
            frames.append(frame)
    except:
        print("Error: ", file_path, len_frames, i)
    finally:
        frames = np.array(frames)
        cap.release()
    flows = getOpticalFlow(frames)
    result = np.zeros((len(flows), 50, 50, 5))
    result[..., :3] = frames
    result[..., 3:] = flows
    return result

def Save2Npy(file_dir, save_dir):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    videos = os.listdir(file_dir)
    for v in tqdm(videos):
        video_name = v.split('.')[0]
        video_path = os.path.join(file_dir, v)
        save_path = os.path.join(save_dir, video_name + '.npy')
        data = Video2Npy(file_path=video_path, resize=(50,50))
        data = np.uint8(data)
        np.save(save_path, data)
    return None
#%%
source_path = "/home/ubuntu/mlproj/data/"
target_path = "/home/ubuntu/mlproj/video/"

for f1 in ['train', 'val']:
    for f2 in ['Fight', 'NonFight']:
        path1 = os.path.join(source_path, f1, f2)
        path2 = os.path.join(target_path, f1, f2)
        Save2Npy(file_dir=path1, save_dir=path2)

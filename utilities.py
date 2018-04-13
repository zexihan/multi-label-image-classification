import numpy as np
import pandas as pd
import cv2

NUM_CLASSES = 228
IMAGE_WIDTH = 224
IMAGE_HEIGHT = 224

def load_images(addrs_list):   
    images = np.empty((len(addrs_list), IMAGE_WIDTH, IMAGE_HEIGHT, 3), dtype=np.float32)
    for i, fpath in enumerate(addrs_list):
        img = cv2.imread(fpath, cv2.IMREAD_COLOR)
        images[i, ...] = img#.transpose(2, 0, 1) 
        if i % 1000 == 0:
            print('Loading images: {}'.format(i))
    return images

def get_multi_hot_labels(df, index_list):
    label_id = [df['labelId'][i] for i in index_list]
    
    labels_matrix = np.zeros([len(index_list), NUM_CLASSES], dtype=np.uint8())
    
    for i in range(len(label_id)):
        for j in range(len(label_id[i].split(' '))):
            row, col = i, int(label_id[i].split(' ')[j]) - 1
            labels_matrix[row][col] = 1
    
    return labels_matrix

#%%
import argparse
import tensorflow as tf
import numpy as np
import cv2
import time
import matplotlib.pyplot as plt
from PIL import Image
from tensorflow.python.tools import freeze_graph 

import argparse
from tqdm import trange
from utils.config import Config
from model import ICNet, ICNet_BN

#%% [markdown]
# # Setup configurations

#%%

parser = argparse.ArgumentParser()
parser.add_argument("--input", default='bedroom.jpg')
parser.add_argument("--model", default='snapshots/model.ckpt-30')
a = parser.parse_args()
model_config = {'train': ICNet, 'trainval': ICNet, 'train_bn': ICNet_BN, 'trainval_bn': ICNet_BN, 'others': ICNet_BN}

# Choose dataset here, but remember to use `script/downlaod_weight.py` first
dataset = 'ade_relabeled'
filter_scale = 2
    
class InferenceConfig(Config):
    def __init__(self, dataset, is_training, filter_scale):
        Config.__init__(self, dataset, is_training, filter_scale)
    
    # You can choose different model here, see "model_config" dictionary. If you choose "others", 
    # it means that you use self-trained model, you need to change "filter_scale" to 2.
    model_type = 'trainval_bn'

    # Set pre-trained weights here (You can download weight from Google Drive) 
    model_weight = a.model
    
    # Define default input size here
    INFER_SIZE = (512, 1024, 3)
                  
cfg = InferenceConfig(dataset, is_training=False, filter_scale=filter_scale)

#%% [markdown]
# # Create graph, session, and restore weights

#%%
# Create graph here 
model = model_config[cfg.model_type]
net = model(cfg=cfg, mode='inference')

# Create session & restore weight!
net.create_session()
net.restore(cfg.model_weight)


#%%
im1 = cv2.imread(a.input)
if im1.shape != cfg.INFER_SIZE:
    im1 = cv2.resize(im1, (cfg.INFER_SIZE[1], cfg.INFER_SIZE[0]),cv2.INTER_CUBIC)

results1 = net.predict(im1)


#%%


#%%
result = plt.imsave('out.jpg', results1[0]/255.0)



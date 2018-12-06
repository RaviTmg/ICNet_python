import argparse
import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
from tensorflow.python.tools import freeze_graph 

from tqdm import trange
from utils.config import Config
from model import ICNet, ICNet_BN

model_config = {'train': ICNet, 'trainval': ICNet, 'train_bn': ICNet_BN, 'trainval_bn': ICNet_BN, 'others': ICNet_BN}

# Choose dataset here, but remember to use `script/downlaod_weight.py` first
dataset = 'ade_relabeled'
filter_scale = 2
    
class ExportGraphConfig(Config):
    def __init__(self, dataset, is_training, filter_scale):
        Config.__init__(self, dataset, is_training, filter_scale)
    
    # You can choose different model here, see "model_config" dictionary. If you choose "others", 
    # it means that you use self-trained model, you need to change "filter_scale" to 2.
    model_type = 'trainval_bn'

    # Set pre-trained weights here (You can download weight from Google Drive) 
    model_weight = 'model/model.ckpt-0'
    
    # Define default input size here
    INFER_SIZE = (512, 1024, 3)
                  
def main():
    cfg = ExportGraphConfig(dataset, is_training=False, filter_scale=filter_scale)

    # Create graph here 
    model = model_config[cfg.model_type]
    net = model(cfg=cfg, mode='inference')

    # Create session & restore weight!
    net.create_session()
    net.restore(cfg.model_weight)

    ckpt_num = cfg.model_weight.split('-')[1]

    #to print graph nodes
    # [print(n.name) for n in tf.get_default_graph().as_graph_def().node]
    #print(net.output)

    tf.train.write_graph(net.sess.graph.as_graph_def(), '.', 'tensorflowModel.pbtxt', as_text=True)

    freeze_graph.freeze_graph(input_graph='tensorflowModel.pbtxt',input_saver="", input_binary=False, 
                            input_checkpoint=cfg.model_weight, output_node_names='Reshape_1',
                            restore_op_name="save/restore_all",filename_tensor_name="save/Const:0",
                            output_graph='frozentensorflowModel-' + ckpt_num +'.pb', clear_devices=True, initializer_nodes=""  
                            )

if __name__ == "__main__":
    main()
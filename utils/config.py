import numpy as np
import os

class Config(object):
    # Setting dataset directory
    ADE_RELABEL_DATA_DIR = './ADERelabelled/'
      
    ADERELABEL_eval_list = os.path.join('ICNet_python/data/list/aderelabel_val_list.txt')

    ADERELABEL_train_list = os.path.join('ICNet_python/data/list/aderelabel_train_list.txt')
    
    IMG_MEAN = np.array((103.939, 116.779, 123.68), dtype=np.float32)

    model_paths = {'train': './model/cityscapes/icnet_cityscapes_train_30k.npy', 
              'trainval': './model/cityscapes/icnet_cityscapes_trainval_90k.npy',
              'train_bn': './model/cityscapes/icnet_cityscapes_train_30k_bnnomerge.npy',
              'trainval_bn': './model/cityscapes/icnet_cityscapes_trainval_90k_bnnomerge.npy',
              'others': 'snapshots/model.ckpt-3000'}
    
    ## If you want to train on your own dataset, try to set these parameters.
    others_param = {'name': 'ade_relabeled',
                    'num_classes': 2,
                    'ignore_label': 0,
                    'eval_size': [480, 480],
                    'eval_steps': 800,
                    'eval_list': ADERELABEL_eval_list,
                    'train_list': ADERELABEL_train_list,
                    'data_dir': ADE_RELABEL_DATA_DIR}

    ## You can modify following lines to train different training configurations.
    INFER_SIZE = [512, 1024, 3] 
    TRAINING_SIZE = [720, 720] 
    TRAINING_STEPS = 201
    
    N_WORKERS = 8
    BATCH_SIZE = 16
    LEARNING_RATE = 1e-4
    MOMENTUM = 0.9
    POWER = 0.9
    RANDOM_SEED = 1234
    WEIGHT_DECAY = 0.0001
    SNAPSHOT_DIR = './snapshots/'
    SAVE_NUM_IMAGES = 4
    SAVE_PRED_EVERY = 40
    
    # Loss Function = LAMBDA1 * sub4_loss + LAMBDA2 * sub24_loss + LAMBDA3 * sub124_loss
    LAMBDA1 = 0.16
    LAMBDA2 = 0.4
    LAMBDA3 = 1.0
    
    def __init__(self, dataset, is_training=False, filter_scale=1, random_scale=False, random_mirror=False):
        print('Setup configurations...')
        
        self.param = self.others_param

        self.dataset = dataset
        self.random_scale = random_scale
        self.random_mirror = random_mirror
        self.is_training = is_training
        self.filter_scale = filter_scale
        
    def display(self):
        """Display Configuration values."""
        print("\nConfigurations:")
        for a in dir(self):
            if not a.startswith("__") and not callable(getattr(self, a)) and not isinstance(getattr(self, a), dict):
                print("{:30} {}".format(a, getattr(self, a)))

            if a == ("param"):
                print(a)
                for k, v in getattr(self, a).items():
                    print("   {:27} {}".format(k, v))

        print("\n")
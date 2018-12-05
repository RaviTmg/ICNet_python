import cv2
import numpy as np
import tensorflow as tf

INPUT_TENSOR_NAME = 'InputNode:0'
OUTPUT_TENSOR_NAME = 'Reshape_1:0'
frozen_graph_filename = 'frozentensorflowModel.pb'
INFER_SIZE = (256, 512, 3)

class Inference:
    
    def __init__(self):
        self.graph = tf.Graph()

        with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
            self.graph_def = tf.GraphDef()
            self.graph_def.ParseFromString(f.read())

        if self.graph_def is None:
            raise RuntimeError('Cannot find inference graph')

        with self.graph.as_default():
            tf.import_graph_def(self.graph_def, name='')
            print('importing graph')

        self.sess = tf.Session(graph=self.graph)  

    def get_markers(self, out_image):
        marker = []
        itemindex = np.where(out_image * 255.0 == [80,50,50])

        for i in range(0, len(itemindex[0])):
            marker.append([itemindex[0][i], itemindex[1][i]])
        
        return marker

    def run(self, image):
        shp = image.shape
        
        if shp != INFER_SIZE:
            print('image not of required size. resizing.....')
            image = cv2.resize(image, (INFER_SIZE[1], INFER_SIZE[0]))
        
        output = self.sess.run(OUTPUT_TENSOR_NAME, 
        feed_dict={INPUT_TENSOR_NAME: image})

        #op = cv2.resize(output[0]/255.0,(shp[1], shp[0]))

        return output
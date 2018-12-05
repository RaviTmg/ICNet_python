import cv2
from inference import Inference
import flask

im1 = cv2.imread('bed (1).jpg')
infer = Inference()
#for i in range(1,9):

output = infer.run(im1)
cv2.imshow("output", output[0]/255.0)
cv2.waitKey(0)
cv2.destroyAllWindows()
import io
import cv2
from PIL import Image
from depth_refiner import DepthPredictor

# pil_img = Image.open('raw_depth0195.png')
pil_img = Image.open('/home/dataset/alldata/yogurt_indoor/depth/00000078.png')
d_predictor = DepthPredictor(gpu_id=0)

e1 = cv2.getTickCount()
res = d_predictor(pil_img)
e2 = cv2.getTickCount()

t = (e2 - e1)/cv2.getTickFrequency()
print('time elapsed:', t)

cv2.imwrite('test_color.png', res)

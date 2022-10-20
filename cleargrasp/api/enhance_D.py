from depth_completion_api import DepthToDepthCompletion
from utils import save_uint16_png
import cv2

input_image = "./test_dir/00000078.jpg"
input_depth = "./test_dir/00000078_d.png"

image_ary = cv2.imread(input_image)
depth_ary = cv2.imread(input_depth, -1)


completion = DepthToDepthCompletion()
output = completion._complete_depth(image_ary, depth_ary)

save_uint16_png('./test_dir/output.png',output)

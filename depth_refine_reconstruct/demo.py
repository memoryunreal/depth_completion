import io
import cv2
from PIL import Image
from depth_refiner import DepthPredictor
import os
import argparse
parser = argparse.ArgumentParser(description='Run tracker on sequence or dataset.')
parser.add_argument('--hole_depth', type=str, help='the directory of depth map which needs completion.')
parser.add_argument('--output_depth', type=str, help='the directory of completed depth map.')

args = parser.parse_args()


# pil_img = Image.open('raw_depth0195.png')
# data_path = "/home/dataset4/cvpr2023/depth_completion/depth_refine_reconstruct/data"
data_path = args.hole_depth

completion = args.output_depth
if not os.path.exists(completion):
    os.makedirs(completion)
    os.makedirs(os.path.join(completion, 'colormap'))

depth_image = [i  for i in os.listdir(data_path) if os.path.isfile(os.path.join(data_path, i)) and i.split(".")[1]=="png"]

for dep in depth_image:
    full_path = os.path.join(data_path, dep)
    pil_img = Image.open(full_path)
    d_predictor = DepthPredictor(gpu_id=0)

    e1 = cv2.getTickCount()
    res = d_predictor(pil_img)
    e2 = cv2.getTickCount()

    t = (e2 - e1)/cv2.getTickFrequency()
    print('time elapsed:', t)

    cv2.imwrite(os.path.join(completion, dep), res)
    colormap = cv2.applyColorMap(cv2.convertScaleAbs(cv2.imread(os.path.join(completion, dep), -1), alpha=0.06), cv2.COLORMAP_JET)
    cv2.imwrite(os.path.join(completion, "colormap", dep), colormap)
    colormap1 = cv2.applyColorMap(cv2.convertScaleAbs(cv2.imread(full_path, -1), alpha=0.03), cv2.COLORMAP_JET)
    # cv2.imwrite(os.path.join(completion, "colormap", "ori_{}".format(dep)), colormap1)

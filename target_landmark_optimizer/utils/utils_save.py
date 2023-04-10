import cv2
from PIL import Image
import matplotlib.pyplot as plt
import copy
import numpy as np
import os

def save_plot(img, img_name, path):
    os.makedirs(path, exist_ok=True)
    plt.imshow(img)
    plt.savefig(path+img_name)

def save_cv_img(img, img_name, path):
    os.makedirs(path, exist_ok=True)
    cv2.imwrite(path+img_name, img)

def save_pts(pts, pts_name, path):
    os.makedirs(path, exist_ok=True)
    with open(path + pts_name, 'w+') as f:
        for p2, p1 in pts:
            f.write('%i, %i\n' % (p2, p1))  # orig : p2, p1


def save_options(dict_options, option_num, path):
    with open(path + 'xfof_class_style_w_opt'+str(option_num) + '.txt', 'w',  encoding='UTF-8') as f:
        for component, weight in dict_options.items():
            f.write(f'{component} : {weight}\n') # https://seong6496.tistory.com/108




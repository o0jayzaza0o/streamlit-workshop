import os
import cv2
import numpy as np
base_dir="D:\dataseg"
def load_img(folder):
    image=[]
    for filename in os.listdir(folder):
        img=cv2.imread(os.path.join(folder,filename))
        if img is not None:
            image.append(img)
    return image
apple_img=load_img(os.path.join(base_dir,"apple"))
orange_img=load_img(os.path.join(base_dir,"orange"))


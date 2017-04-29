# -*- coding: utf-8 -*-
from os import listdir
import os

import numpy as np 
import scipy as sp 
import matplotlib.pyplot as plt  
import skimage as sk
import skimage.io as io
from skimage import data as da
from tools import *
   
IMG_DIR = r'E:\3-experiment\SalBenchmark-master\Data\DataSet1\Imgs/'
COARSE_DIR =r'E:\3-experiment\SalBenchmark-master\Data\DataSet1\Saliency/'

IMG_DIR =  '../DataSet1/Imgs/'
COARSE_DIR ='../DataSet1/Saliency/'

LABEL_DATA_DIR = os.path.dirname(IMG_DIR[:-1])+'/LabelData/'
if not os.path.isdir(LABEL_DATA_DIR):
    os.mkdir(LABEL_DATA_DIR)
    
IMG_NAME_LIST = filter(lambda x:x[-3:]=='jpg',listdir(IMG_DIR))
allMethods = list(set(map(lambda x: x[x.rindex('_')+1:-4],listdir(COARSE_DIR))))
  
# 自动画图

import numpy as np
import cv2
from matplotlib import pyplot as plt

if 0:
    img = da.astronaut()             # img.shape : (413, 620, 3)
    mask = np.zeros(img.shape[:2],np.uint8)   # img.shape[:2] = (413, 620)
    
    bgdModel = np.zeros((1,65),np.float64)
    fgdModel = np.zeros((1,65),np.float64)
    
    rect = (300,120,470,350)
    
    # this modifies mask 
    cv2.grabCut(img,mask,rect,bgdModel,fgdModel,5,cv2.GC_INIT_WITH_RECT)
    
    # If mask==2 or mask== 1, mask2 get 0, other wise it gets 1 as 'uint8' type.
    mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
    
    # adding additional dimension for rgb to the mask, by default it gets 1
    # multiply it with input image to get the segmented image
    img_cut = img*mask2[:,:,np.newaxis]
    
    plt.subplot(211),plt.imshow(img)
    plt.title('Input Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(212),plt.imshow(img_cut)
    plt.title('Grab cut'), plt.xticks([]), plt.yticks([])
    plt.show()

import numpy as np
import cv2
from matplotlib import pyplot as plt

img = cv2.imread('bolt.jpg')              # img.shape : (413, 620, 3)
mask = np.zeros(img.shape[:2],np.uint8)   # img.shape[:2] = (413, 620)

bgdModel = np.zeros((1,65),np.float64)
fgdModel = np.zeros((1,65),np.float64)

rect = (300,120,470,350)

# this modifies mask 
cv2.grabCut(img,mask,rect,bgdModel,fgdModel,5,cv2.GC_INIT_WITH_RECT)

# If mask==2 or mask== 1, mask2 get 0, other wise it gets 1 as 'uint8' type.
mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')

# adding additional dimension for rgb to the mask, by default it gets 1
# multiply it with input image to get the segmented image
img_cut = img*mask2[:,:,np.newaxis]

plt.subplot(221),plt.imshow(img[:,:,[2,1,0]])
plt.title('Input Image'), plt.xticks([]), plt.yticks([])
plt.subplot(222),plt.imshow(img_cut[:,:,[2,1,0]])
plt.title('Grab cut'), plt.xticks([]), plt.yticks([])
plt.show()







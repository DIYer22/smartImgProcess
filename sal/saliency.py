# -*- coding: utf-8 -*-
from __future__ import unicode_literals
from os import listdir
import os
import multiprocessing
from copy import deepcopy

import numpy as np 
import scipy as sp 
import matplotlib.pyplot as plt  
import skimage as sk
import skimage.io as io
from skimage import data as da
from skimage.segmentation import mark_boundaries
import cv2

import cProfile as profile
crun = lambda x:profile.run(x,sort='time')

from tools import mapp,show,getPoltName,performance,normalizing,random,loga
from tools import saveData,loadData

from warnings import filterwarnings
filterwarnings('ignore')



r=random(3,5)
G = {} # G is a global var to save value for DEBUG in funcation



#IMG_DIR =  '../DataSet1/Imgs/'
#COARSE_DIR ='../DataSet1/Saliency/'

#IMG_DIR =  r'G:\Data\HKU-IS/Imgs/'
#COARSE_DIR = r'G:\Data\HKU-IS/Saliency/'

IMG_DIR = r'G:\experiment\Data\DUT-OMRON-image\imgs/'
COARSE_DIR = r'G:\experiment\Data\DUT-OMRON-image\Saliency/'

#IMG_DIR = r'E:\3-experiment\SalBenchmark-master\Data\DataSet1\Imgs/'
#COARSE_DIR =r'E:\3-experiment\SalBenchmark-master\Data\DataSet1\Saliency/'
    
#IMG_DIR = r'E:\3-experiment\SalBenchmark-master\Data\DataSet2\Imgs/'
#COARSE_DIR =r'E:\3-experiment\SalBenchmark-master\Data\DataSet2\Saliency/'

#IMG_DIR = r'E:\3-experiment\SalBenchmark-master\Data\HKU-IS\Imgs/'
#COARSE_DIR =r'E:\3-experiment\SalBenchmark-master\Data\HKU-IS\Saliency/'

#IMG_DIR = r'E:\3-experiment\SalBenchmark-master\Data\MSRA\Imgs/'
#COARSE_DIR =r'E:\3-experiment\SalBenchmark-master\Data\MSRA\Saliency/'

IMG_NAME_LIST = []
LABEL_DATA_DIR = []
#LABEL_DATA_DIR = os.path.dirname(IMG_DIR[:-1])+'/LabelData/'
#if not os.path.isdir(LABEL_DATA_DIR):
#    os.mkdir(LABEL_DATA_DIR)
#    
#IMG_NAME_LIST = filter(lambda x:x[-3:]=='jpg',listdir(IMG_DIR))
#allMethods = list(set(map(lambda x: x[x.rindex('_')+1:-4],[imgName for imgName in listdir(COARSE_DIR) if '.png' in imgName])))

if __name__ == '__main__':
    from algorithm import readImg,getCoarseDic,showpr
    import algorithm as alg  # main algorithm
    from algorithm import buildImgs,mergeImgs
    
    from analysis import plotMethods,getPrCurve

    
    IMG_NAME_LIST=IMG_NAME_LIST[::]
    # build MY* 创造
    coarseMethods=['QCUT','DISC2']
    coarseMethods=['QCUT']
    
    buildMethods=['MY4']
    num = len(IMG_NAME_LIST)
#    num = 1
    for name in IMG_NAME_LIST[:num]:
        buildImgs(name,buildMethods,coarseMethods)
    
    
    # merge methods to make ME* 融合
    mergeMethods=['MY4','DISC2']
    for name in IMG_NAME_LIST[:num]:
        mergeImgs(name,mergeMethods)
        
    
    #  画图分析
    #raise LookupError,u'结束'
    #showMethods = ["MY1","MY2","MY3","MY4","MY5","ME1","ME2","ME3","MEAN", "DRFI", "GMR","QCUT","DISC2"]
    
    showMethods = ["ME1","FT","GC","RC", "DRFI", "GMR","QCUT","DISC2"]
    showMethods += filter(lambda x:x not in showMethods,buildMethods)
    
    data = plotMethods(showMethods,
                       num=num,
                       save=getPoltName(coarseMethods,IMG_DIR)
                       )













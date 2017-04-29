# -*- coding: utf-8 -*-
"""
Created on Thu Jul 28 00:36:24 2016

@author: yl
"""
from __future__ import unicode_literals


def log(x,strr='log'):
    print strr+": ",x
    g[0] = x
    return x
    
def mapp(f, matrix, need_i_j=False):
    '''
    for each it of a matrix
    return a new matrix consist of f:f(it) or f(it, i, j)
    '''
    m, n = matrix.shape[:2]
    if not need_i_j:
        return np.array(map(lambda row :
                            map(lambda i:f(i), row), 
                        matrix), matrix.dtype)
    listt = [[None]*n for i in range(m)]
    for i in range(m):
        for j in range(n):
            it = matrix[i][j]
            listt[i][j] = f(it,i,j) if need_i_j else f(it)
    return np.array(listt, matrix.dtype)

def loga(array):
    if isinstance(array,list):
        array = np.array(array)
    if isinstance(array,str) or isinstance(array,unicode):
        print 'info and histogram of',array
        l=[]
        eval('l.append('+array+')')
        array = l[0]
    print 'shape:%s ,max: %s, min: %s'%(str(array.shape),str(array.max()),str(array.min()))
    
    unique = np.unique(array)
    if len(unique)<10:
        dic=dict([(i*1,0) for i in unique])
        for i in array.ravel():
            dic[i] += 1
        listt = dic.items()
        listt.sort(key=lambda x:x[0])
        data,x=[v for k,v in listt],np.array([k for k,v in listt]).astype(float)
        width = (x[0]-x[1])*0.7
        x -=  (x[0]-x[1])*0.35
    else:
        data, x = np.histogram(array.ravel(),8)
        x=x[1:]
        width = (x[0]-x[1])
    plt.plot(x, data, color = 'orange')
    plt.bar(x, data,width = width, alpha = 0.5, color = 'b')
    plt.show()
    return 
    
def show(l,lab=False):
    '''
    do io.imshow to a list of imgs or one img
    lab,means if img`color is lab
    '''
    if isinstance(l,dict):
        l = l.values()
    if not isinstance(l,list) and (not isinstance(l,tuple) ) :
        l = [l]
    n = len(l)
    if n > 3:
        show(l[:3],lab)
        show(l[3:],lab)
        return 
    fig, axes = plt.subplots(ncols=n)
    count = 0
    axes = [axes] if n==1 else axes 
    for img in l:
        axes[count].imshow(
        sk.color.lab2rgb(img) if len(img.shape)==3 
        and lab else img,
        cmap='gray')
        count += 1
    plt.show()
    

def getImgBase64(arr):
    io.imsave('tmp.png',arr)
    with open('tmp.png','rb') as f:
        imgBase64 = base64.b64encode(f.read())
    return imgBase64

def setCount(img):
    '''
    统计各个像素值的数目 并将其存入countDic中
    '''
    colorPixelCount = np.array([np.histogram(img[...,ind],range(257))[0] for ind in range(3)])
#    pixelCountMax = colorPixelCount.max()
    pixelCountMax = colorPixelCount[:,3:-3].max()
    #pixelCountMax = int(pixelCountMax*0.3)
    countDic['counts'] = colorPixelCount
    countDic['maxx'] = pixelCountMax
    
    
def addColorToHis(his, inds):
    '''
    生成 inds 通道的颜色直方图
    '''
    counts, maxx = countDic['counts'], countDic['maxx']
    xs,ys,zs=[],[],[]
    for x in range(256):
        for z in inds:
            num = stand(256 * counts[z, x]/maxx)
            ys += range(256-num, 256)
            xs += [x]*num
            zs += [z]*num
    his[ys, xs, zs] = 220

def applyColorTable(colorTable, img, inds):
    '''
    colorTable 颜色函数表
    inds 待处理的通道
    his 通道直方图与函数面板
    '''
    m,n,_ = img.shape
    for row in range(m):
        for col in range(n):
            for ind in inds:
                img[row,col,ind] = colorTable[img[row,col,ind]]
#    return img
#    inds = np.array([i in inds for i in range(3)])
#    img = mapp(lambda x:x*(inds==False)+inds*([colorTable[i] for i in x]),img)
def changeLineToHis(img, inds, colorTable):
    '''
    当颜色改变时候
    '''
    setCount(img)
    his = np.zeros((256,256,3),np.uint8)+90
    addColorToHis(his, inds)
    for ind in inds:
        his[255-colorTable,range(256),[ind]*256] = 255 
    return his

     
import numpy as np
import matplotlib.pyplot as plt  
import skimage as sk
import skimage.io as io
from skimage import data as da
import skimage

import base64
import cProfile
#from line_profiler import  LineProfiler as lp 
run = lambda cmd:cProfile.run(cmd,sort='time') 
from math import tan

array,arange = np.array, np.arange

random = lambda shape,maxx:(np.random.random(
shape if (isinstance(shape,tuple) or isinstance(shape,list)
)else (shape,shape))*maxx).astype(int)

normalizing = lambda a:(a.astype(float)-a.min())/(a.max() - a.min())
stand = lambda num:max(min(255, int(round(num))),0)

r = random(3,5)
g = [0] * 3   


img = da.astronaut()
#img = da.coffee()
inds = [0,1,2]

raw = img.copy()


import cv2


# -*- coding: utf-8 -*-
import scipy.weave as weave
import numpy as np
import time

def my_sum(a):
    n=int(len(a))
    code="""
    int i;

    double counter;
    counter =0;
    for(i=0;i<n;i++){
        counter=counter+a(i);
    }
    return_val=counter;
    """

    err=weave.inline(
        code,['a','n'],
        type_converters=weave.converters.blitz,
        compiler="gcc"
    )
    return err

a = np.arange(0, 10000000, 1.0)
# 先挪用一次my_sum，weave会主动对C说话举行编译，今后直接运行编译之后的代码
my_sum(a)

start = time.clock()
for i in xrange(100):
    my_sum(a)  # 直接运行编译之后的代码
print "my_sum:", (time.clock() - start) / 100.0

start = time.clock()
for i in xrange(100):
    np.sum( a ) # numpy中的sum，实在现也是C说话级别
print "np.sum:", (time.clock() - start) / 100.0

start = time.clock()
print sum(a) # Python内部函数sum经由过程数组a的迭代接口会见其每个元素，是以速率很慢
print "sum:", time.clock() - start
























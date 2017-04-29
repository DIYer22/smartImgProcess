# -*- coding: utf-8 -*-
"""
Created on Thu Jul 28 00:36:24 2016

@author: yl
"""
from __future__ import unicode_literals

from imgProcessConfig import *
import os
#%%
def log(x,strr='log'):
    print (strr+": ",x)
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
    if isinstance(array,str) or isinstance(array,str):
        print ('info and histogram of',array)
        l=[]
        eval('l.append('+array+')')
        array = l[0]
    print ('shape:%s ,max: %s, min: %s'%(str(array.shape),str(array.max()),str(array.min())))
    
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



    
import numpy as np
import matplotlib.pyplot as plt  
import skimage as sk
import skimage.io as io
from skimage import data as da
import skimage
import cv2

import base64
import cProfile
#from line_profiler import  LineProfiler as lp 
crun = run = lambda cmd:cProfile.run(cmd,sort='time') 
from math import tan
from math import log
import math

array,arange = np.array, np.arange

random = lambda shape,maxx:(np.random.random(
shape if (isinstance(shape,tuple) or isinstance(shape,list)
)else (shape,shape))*maxx).astype(int)

normalizing = lambda a:(a.astype(float)-a.min())/(a.max() - a.min())
stand = lambda num:max(min(255, int(round(num))),0)
roundInt = lambda x: int(round(x))


from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

SHOW_SHAPE= (500,500)
HIS_BACKGROUND_COLOR = 40 #51
HIS_MAX = 8



class View:
    speed = 0.05  # 缩放速度
    def __init__(self, img, shape=(400,400)):
        self.img = img
        self.m, self.n = img.shape[:2]
#        self.zeros = np.zeros((self.m*2, self.n*2,3),np.uint8)
        # 显示中点对应的图像像素 不可以离开图像
        self.py, self.px = self.m//2, self.n//2
        self.shape = np.array(shape)
        self.up = self.speed + 1
        self.down = 1./self.up 
        maxLevel = max(log(self.m/float(shape[0]),self.up),log(self.n/float(shape[1]),self.up))
        minLevel = log(0.5, self.up)
        self.ratios = [self.up**level for level in [minLevel]+range(int(minLevel),int(maxLevel)+1)+[maxLevel]]
        self.levels = [ np.array([roundInt(shape[0]*ratio), roundInt(shape[1]*ratio)]) for ratio in self.ratios]
#        self.levels = [ np.array([roundInt(shape[0]*self.up**level), roundInt(shape[1]*self.up**level)])
#                        for level in range(int(maxLevel)+1)+[maxLevel]]
        self.maxx = len(self.levels)-1
        self.now = self.maxx
    def getView(self,onlyImg=False):
        '''
        创造一个2m*2n的黑色背景图片 将img叠在中间
        '''
        y_, x_ = self.levels[self.now]
        m, n = self.m, self.n 
        py, px = self.py, self.px
        shape = self.shape
        img = self.img
        black = max(m,n)
        zeros = np.zeros((m+black,n+black,3),np.uint8)
        zeros[black//2:black//2+m,black//2:black//2+n] = img
#        print ([black//2+py-y_//2, black//2+py+y_//2,
#                     black//2+px-x_//2, black//2+px+x_//2, ])
        imgg = zeros[black//2+py-y_//2: black//2+py+y_//2,
                     black//2+px-x_//2: black//2+px+x_//2, :]
#        print (black//2,py,px,y_,x_,self.now)
#        show(imgg) 
        imgg = cv2.resize(imgg,tuple(shape))
        if onlyImg:
            return imgg
        
        zeros[black//2:black//2+m,black//2:black//2+n] = g['rraw']
        raww = zeros[black//2+py-y_//2: black//2+py+y_//2,
                     black//2+px-x_//2: black//2+px+x_//2, :]
        raww = cv2.resize(raww,tuple(shape))
        return imgg,raww
    @property
    def v(self):
        return self.getView(True)
    def move(self,xy):
        xy = np.array(xy)*self.levels[self.now][0]//self.shape[0]
        m, n = self.m, self.n 
        middle = lambda x, maxx:max(0,min(x,maxx))
        self.py = middle(self.py-xy[1], m-1)
        self.px = middle(self.px-xy[0], n-1)
        return self.getView()
        
    def zoomDown(self):
        if self.now == self.maxx:
            self.py,self.px = self.py+roundInt((self.m/2-self.py)*0.3), self.px+roundInt((self.n/2-self.px)*0.3)
            return self.getView()
        self.now = min(self.now+1, self.maxx)
        return self.getView()        
    def zoomUp(self):
        self.now = max(self.now-1, 0)
        return self.getView()    
    def zoomLevel(self, level):
        self.now = min(max(level, 0), self.maxx)
        return self.getView()

r,c = 1000,800
step = 10
bg = np.zeros((r,c,3),np.uint8)
ind = np.zeros((r,c),np.bool)


def draw3dSurface(X,Y,Z):

    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib import cm
    from matplotlib.ticker import LinearLocator, FormatStrFormatter
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    
    surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=0, antialiased=False)
    #画表面,x,y,z坐标， 横向步长，纵向步长，颜色，线宽，是否渐变
    
    #ax.set_zlim(-1.01, 1.01)#坐标系的下边界和上边界
    ax.zaxis.set_major_locator(LinearLocator(10))#设置Z轴标度
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))#Z轴精度
    fig.colorbar(surf, shrink=0.5, aspect=5)#shrink颜色条伸缩比例（0-1），aspect颜色条宽度（反比例，数值越大宽度越窄）
    
    plt.show()


def polt3dSurface(Z):
    m, n = Z.shape
    X = range(n)
    Y = range(m)
    X, Y = np.meshgrid(X, Y)
    draw3dSurface(X,Y,Z)

def saveAvi(name, arrays, frames=24):
    ''' 将图片集保存为.avi视频
    name: 名字
    arrays: 帧集  
    frames: 帧率
    '''
    import cv2
    from skimage import img_as_ubyte
    fourcc = cv2.VideoWriter_fourcc(*'DIB ')
    log((name+'.avi',fourcc, frames, arrays[0].shape[:2]))
    out = cv2.VideoWriter(name+'.avi',fourcc, frames, arrays[0].shape[:2])
    for frame in arrays:
        show(frame)
        loga(img_as_ubyte(frame))
        out.write(img_as_ubyte(frame))
    out.release()
    cv2.destroyAllWindows()
    

def blackToWhiteVidoEffects(img,frameNum=24,fname='a.gif'):
    img = mapp(lambda x:stand(x.sum()/3.), img)
    show(img)
#    img = cv2.resize(img,(800,600))
    setCount(img)
    count = np.histogram(img,range(257))[0]
    
    per = count.sum()/frameNum
    tmp,cuts = 0, []
    for c in range(256):
        tmp += count[c]
        if tmp >= per:
            cuts += [img>c]
            tmp = 0
    import imageio
    imageio.mimsave(fname, cuts)
#    saveAvi('a', cuts)

#blackToWithVidoEffects(img,60)

def base64Img(arr):
    cnt = cv2.imencode('.jpg',arr[:,:,[2,1,0]])[1]
    if py3:
        return base64.encodebytes(cnt[...,0]).decode('utf-8')
    return base64.encodestring(cnt)
#    io.imsave('tmp.jpg',arr)
#    with open('tmp.jpg','rb') as f:
#        imgBase64 = base64.b64encode(f.read())
#    return imgBase64



def applyColorTable(colorTables, img, inds):
    '''
    colorTable 颜色函数表
    inds 待处理的通道
    his 通道直方图与函数面板
    '''
    m,n = img.shape[:2]
    
    new = np.zeros(img.shape).astype(np.uint8)
    for i,v in enumerate(colorTables[0]):
        new[img==i] = v
    img[:,:,:] = new
    return 
    for row in range(m):
        for col in range(n):
            for ind in inds:
                img[row,col,ind] = colorTables[ind][img[row,col,ind]]
                

    
def getColorTables(fun, inds):
    return [(np.array([stand(fun(x)) for x in range(256)]) if ind in inds else None) for ind in range(3)]


class Fun:
    def __init__(self,f ,name=u'f',maxx=5 ,minn=-5 ,value=0):
        self.dic = {}
        self.dic['f'] = f
        self.dic['max'] = maxx
        self.dic['min'] = minn
        self.dic['value'] = value
        self.dic['name'] = name
    
        self.f = self.dic['f']
        self.maxx = self.dic['max']
        self.minn = self.dic['min']
        self.value = self.dic['value']
        self.name = self.dic['name']
    def __getitem__(self, k):
        return self.dic[k]
    def __call__(self, x, a):
        if a == self.dic['value']:
            return x
        return self.f(x, a)
    

#run('base64Img(vi.zoomUp())')
def getLine(img, thres = 0.05):
    ''' 将img 导数大于 thres 的 值 设为255
    '''
    grey = sk.color.rgb2gray(img)
    grey = normalizing(grey)
    r = np.r_[grey[1:],grey[-1:]]
    d = np.c_[grey[:,1:],grey[:,-1:]]
    line = (abs(grey-d)+abs(grey-r))/2
    black = sk.img_as_ubyte(line > thres)
    return black.repeat(3).reshape(img.shape)

def gaussCore(r):
    '''
    0<r<50 defult 0
    '''
    from math import e 
    sig = 1
    axisLen = 2
#    axisLen = 0.5*sig
    thred = 0.05
    maxR = MAX_FILTER_R
    X = np.linspace(-axisLen, axisLen, maxR*2+1)
    Y = np.linspace(-axisLen, axisLen, maxR*2+1)
    X, Y = np.meshgrid(X, Y)
    
    Z = np.e**(-(X**2+Y**2)/2/sig)/(2*np.pi*sig**2)
#    draw3dSurface(X,Y,Z)
    core = Z[maxR-r:maxR+r+1,maxR-r:maxR+r+1]
    tmp = core>=core[0,r]
    
    core = core*(tmp)
    core = core/core.sum()
    return core

def getFilterImg(img):
    maxr = MAX_FILTER_R
    m,n = img.shape[:2]
    new = np.zeros((m+2*maxr,n+2*maxr,3),np.uint8)
    u,r,d,l = maxr,n+maxr,m+maxr,maxr
    new[u:d,l:r] = img
    new[:u] = new[u:u+maxr][::-1]
    new[:,r:] = new[:,r-maxr:r][:,::-1]
    new[d:] = new[d-maxr:d][::-1]
    new[:,:l] = new[:,l:l+maxr][:,::-1]
#    show(new)
    return new
def gussFilter(img,R):
    '''
    10s 太慢
    '''
    core = gaussCore(R)
    fi = getFilterImg(img)
    maxr = MAX_FILTER_R
    m,n = img.shape[:2]
    u,r,d,l = maxr,n+maxr,m+maxr,maxr
    new = np.zeros((m,n,3),np.uint8)
    for i,y in enumerate(range(maxr,maxr+m)):
        for j,x in enumerate(range(maxr,maxr+n)):
            v = np.round((fi[y-R:y+R+1,x-R:x+R+1]*core[:,:,None]).sum(0).sum(0))
            new[i][j] = v
    return new
  
def maxFilter(img,R):
    '''
    '''
    core = gaussCore(R)
    tmp = (core!=0)[...,None]
    def f(block):
        return (block*tmp).max(0).max(0)
    return filterr(img,R,f)
    
def filterr(img,R,f):
    '''
    10s 太慢
    '''
    fi = getFilterImg(img)
    maxr = MAX_FILTER_R
    m,n = img.shape[:2]
    u,r,d,l = maxr,n+maxr,m+maxr,maxr
    new = np.zeros((m,n,3),np.uint8)
    for i,y in enumerate(range(maxr,maxr+m)):
        for j,x in enumerate(range(maxr,maxr+n)):
            v = f(fi[y-R:y+R+1,x-R:x+R+1])
            new[i][j] = v
    return new

  
import scipy
from scipy import interpolate
#%%
def mdeianFilter(img,R):
    '''
    '''
    core = gaussCore(R)
    tmp = (core!=0)
    def mdeianCore(block):
        rgbs = (block[tmp])
        return np.median(rgbs,axis=0)
#        return np.array([np.median(rgbs[...,0]),np.median(rgbs[...,1]),np.median(rgbs[...,2])],np.uint8)
    return filterr(img,R,mdeianCore)
def avgFilter(img,R):
    '''
    '''
    core = gaussCore(R)
    tmp = (core!=0)
    def avgCore(block):
        return np.mean(block[tmp],0)
    return filterr(img,R,avgCore)
def bilateralFilter(img,R):
#    block = random((2*r+1,2*r+1,3),255)
#    block[r-1:,r-1:] = 10
    r = R
    sd = 100
    sr = sd*0.4
    m,n = 2*r+1,2*r+1
    x,y = np.meshgrid(range(n),range(m))
    dd = -((y-r)**2+(x-r)**2)/2./sd
    tmp = (dd>=dd[r,0])
    def bilateralCore(block):
        rr = -np.linalg.norm(block-block[r,r],axis=2)/2./sr
        core = np.power(np.e,dd+rr)*tmp
        core = (core/core.sum())[...,None]
#        if random(1,1000)[0][0]>990:show([core[...,0],block])
        return (core*block).sum(0).sum(0)
    new = filterr(img,R,bilateralCore)
    return new

def allFilter(r=10):
    show(img)
    show(gussFilter(img,r))
    show(maxFilter(img,r))
    show(mdeianFilter(img,r))
    show(avgFilter(img,r))
    show(bilateralFilter(img,r))
 
#%%
def smallImg(img, maxPixels=512*512):
  m,n = img.shape[:2]
  newM = int((maxPixels*m/float(n))**0.5)
  newShape = (newM*n//m,newM)
  new = cv2.resize(img,newShape)
  return new
if __name__ == '__main__':
    img = da.astronaut()
    R = 10   
#    crun('mdeianFilter(img,R)')
    pass
    
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 28 00:36:24 2016

@author: yl
"""
from __future__ import unicode_literals

from imgProcessConfig import *

from imgProcessTools import *

from aiMain import *

from seamCarving import seamCarve,addBlack,scShow
def getAllImport():
    '''
    自动生成导入所有模块语句 并过滤掉__name__等
    '''
    import imgProcessTools as mod # <= 更改 math 为想导入的模块
    print (("from %s import (%s)"%(mod.__name__,', '.join([i for i in dir(mod) if i.count('_')!=4]))))
def setCount(img):
    '''
    统计各个像素值的数目 并将其存入countDic中
    '''
    colorPixelCount = np.array([np.histogram(img[...,ind],range(257))[0] for ind in range(3)])
#    pixelCountMax = colorPixelCount.max()
#    pixelCountMax = colorPixelCount[:,3:-3].max()
    pixelCountMax = img.shape[0]*img.shape[1]*HIS_MAX/256
    #pixelCountMax = int(pixelCountMax*0.3)
    countDic = {}
    countDic['counts'] = colorPixelCount
    countDic['maxx'] = pixelCountMax
    return countDic
    
def getHisWithLine(img, inds, colorTables):
    '''
    当颜色改变时候
    '''
    countDic = setCount(img)
    his = np.zeros((256,256,3),np.uint8)+HIS_BACKGROUND_COLOR
    addColorToHis(his, inds,countDic)
    for ind in inds:
        his[255-colorTables[ind],range(256),[ind]*256] = 255 
    return his    
def addColorToHis(his, inds, countDic):
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


ra = r = random(3,5)


#path = 'imgs/uce.jpg'
#img = da.astronaut()
#img = io.imread('imgs/uce.jpg')
#img = io.imread('imgs/lenna_top.jpg')
#img = io.imread('imgs/emma.jpg')
#img = da.coffee()
inds = [0,1,2]

#img = mapp(lambda x:[stand(x.sum()/3.)]*3, img)
countDic = {}  # 存储各个像素值的数目 

    #plt.plot(np.arange(0,1,0.01),np.arange(0,1,0.01)**a)
    
#    f = lambda x:x#contrast(x,1)
#    colorTables = getColorTables(f,g['inds'])
#    applyColorTable(colorTables,img,g['inds'])
#    g['resHis'] = getHisWithLine(img, g['inds'], colorTables)





def changeImg(img,data):
    p = img
    for k,v in data.items():
      img = funDic[k](img,v)
    p[:,:,:] = img
    return img


def changeColorTable(img,data):
    fa = data.items()
    def f(x):
        for k,a in fa:
            x = funDic[k](x,a)
        return x
#    f=reduce(lambda x,y:lambda z:funDic[y[0]](funDic[x[0]](z,x[1]),y[1]),data.items())
    colorTables = getColorTables(f,g['inds'])
    applyColorTable(colorTables,img,g['inds'])
    return colorTables


import scipy
from scipy import interpolate


#%%
# -50<a<50 value=0
def heightLight(x, a):
    f = interpolate.pchip([0,76,204-a,255],[0,76,204+a,255])
    return float(f(x))
#crun('[heightLight(i,20) for i in range(255)]')
#%%
def shadow(x, a):
    f = interpolate.pchip([0,52-a,180,255],[0,52+a,180,255])
    return float(f(x))
  
def getHsv(img,satura=0.1):
    hsv = sk.color.rgb2hsv(img)
    hsv[:,:,2]+= satura
    hsv[hsv>1] = 1
    hsv[hsv<0] = 0
#    hsvshow = lambda x:show(sk.color.hsv2rgb(x))
#    hsvshow(hsv)
    new = sk.color.hsv2rgb(hsv)
    return (new*255.9999).astype(np.uint8)
  
funDic = {
#       u'f':Fun(lambda x,a:x+a,'f',128,-128,0),
       u'亮度':Fun(lambda x,a: x+a,
                 u'亮度',128,-128,0),
       u'对比度':Fun(lambda x, a=1: 128.5 + 128.5*(abs(x-128.5)/128.5)**(1./a)*(-1 if x<128.5 else 1),
                      u'对比度',3,0.3,1.002),
       u'高亮':Fun(heightLight,
                      u'高亮',50,-50,0),
       u'阴影':Fun(shadow,
                      u'阴影',50,-50,0),
                 
      u'饱和度':Fun(getHsv,
                 u'饱和度',0.3,-0.3,0),
      u'高斯滤波':Fun(gussFilter,
                 u'高斯滤波',50,0,0),
      u'最大值滤波':Fun(maxFilter,
                 u'最大值滤波',50,0,0),
      u'中值滤波':Fun(mdeianFilter,
                 u'中值滤波',50,0,0),
      u'平均值滤波':Fun(avgFilter,
                 u'平均值滤波',50,0,0),
      u'双边滤波':Fun(bilateralFilter,
                 u'双边滤波',50,0,0),
      
       }

channelFunList = [u'对比度', u'亮度', u'高亮', u'阴影']
imgFunList = [u'饱和度',u'高斯滤波',u'最大值滤波',u'中值滤波',u'平均值滤波',u'双边滤波']
orderList = [ u'亮度', u'对比度',u'饱和度', u'高亮', u'阴影',u'高斯滤波',u'最大值滤波',u'中值滤波',u'平均值滤波',u'双边滤波']

def getSaliencyByPath(path):
    from sal import getSaliency
    
    new = path[:path.rindex('.')]+'_sal.png'
    if os.path.isfile(new):
        sal = io.imread(new)
        if g['img'].shape[:2] == sal.shape:
            return normalizing(sal)
    sal = getSaliency(g['rraw'])
    io.imsave(new,sal)
    sal = normalizing(sal)
    return (sal)

def getMasksByPath(path):
    
    new = path[:path.rindex('.')]+'_SS.png'
    if os.path.isfile(new):
        masks = io.imread(new)
        if g['img'].shape[:2] != masks.shape:
            masks = smallImg(masks)
        g['kinds'] = [i for i in np.unique(masks) if i]
        return (masks)
    getSsMasks = lambda x:None
    print (u'windows无法对接语义分割算法sec https://github.com/kolesman/SEC'),new
    raise OSError,(u'windows无法对接语义分割算法sec https://github.com/kolesman/SEC'),new
    
    masks = getSsMasks(g['rraw'])
    io.imsave(new,masks)
    return (masks)
    
def getHedByPath(path):    
    new = path[:path.rindex('.')]+'_hed.png'
    if os.path.isfile(new):
        hed = io.imread(new)
        if g['img'].shape[:2] != hed.shape:
            hed = smallImg(hed)
        return (hed)
    buildHed = lambda x:None
    print u'windows无法对接Hed算法 https://github.com/s9xie/hed',new
    raise OSError,u'windows无法对接Hed算法 https://github.com/s9xie/hed',new
    hed = buildHed(g['rraw'])
    io.imsave(new,hed)
    return (hed)
    
smallImg = lambda x,y=None:x

def begin(path):
    if isinstance(path, np.ndarray):
        img = path
        path = g['path']
    else:
        img = io.imread(path)
        img = smallImg(img,)
        g.clear()
        g['path'] = path
        g['rraw'] = img.copy()
    g['img'] = img

    g['sal'] = np.ones(img.shape[:2])
#    g['sal'] = getSaliencyByPath(path)
#    g['masks'] = getMasksByPath(path)
#    g['hed'] = getHedByPath(path)

    g['raw'] = img.copy()
    g['bg'] = img.copy()
    g['fg'] = img.copy()
    g['inds'] = [0,1,2]
    g['mouse'] = np.array([0,0])
    import imgProcessConfig as pc

    g['vi'] = View(img,SHOW_SHAPE)
#    pc.vi = g['vi']
    countDic = setCount(img)
    g['rawHis'] = np.zeros((256,256,3),np.uint8)+HIS_BACKGROUND_COLOR
#    g['resHis'] = np.zeros((256,256,3),np.uint8)+HIS_BACKGROUND_COLOR
    addColorToHis(g['rawHis'], g['inds'],countDic)
#    g['fliterImg'] = getFilterImg(img)
#photos = ['./imgs/'+'photos (%d).jpg'%i  for i in range(1,9)]
name = 'sun_aadtmifyuvqcfkcr'
#name = 'emma'
#name = 'photos (4)'
#name = 'lenna_top'
path = './imgs/'+name+'.jpg'
#path = './imgs/uce.jpg'
path = u'./imgs/hku/0084.jpg' or '小孩与狗9'
#path = u'./imgs/hku/0317.jpg' or '狗猫8'
#path = u'./imgs/hku/1206.jpg' or '女孩与马5'
#path = u'./imgs/hku/0302.jpg' or '狗小猫7'

#begin(path)
#begin(img)
#img, raw, inds = g['img'], g['raw'],g['inds']
#counts, maxx = countDic['counts'], countDic['maxx']
#%%
if __name__ == '__main__':
    name = 'emma'
#    path = './imgs/'+name+'.jpg'
    begin(path)
    img,gt = g['rraw'],g['sal']
    pass

    

    
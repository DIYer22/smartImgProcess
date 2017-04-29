# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from imgProcessTools import *
from seamCarving import *

def smoothBinImag(img):
    img = img.astype(np.uint8)
    size = 10
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(size, size))  
    eroded = cv2.erode(img,kernel)  
    
    kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT,(size*2, size*2))  
    dilated = cv2.dilate(eroded,kernel2)  
    new = dilated
    kernel3 = cv2.getStructuringElement(cv2.MORPH_RECT,(int(size*0.2), int(size*0.2)))
    new = cv2.erode(dilated,kernel3)
#    show([img,new])
#    show([eroded,dilated])
#    show(img!=new) 
    return new!=0
    
def getBorders(mask):
    a=mask.sum(1)
    ll = ''.join(['0' if i==0 else '1' for i in a])
    u = ll.index('1')
    d = ll.rindex('1')
    
    a=mask.sum(0)
    ll = ''.join(['0' if i==0 else '1' for i in a])
    l = ll.index('1')
    r = ll.rindex('1')
    #show(mask)
    #mask[u:d,l:r]=1
    #show(mask)
    return u,r,d,l
def getDelnIsRow(mask):
    u,r,d,l = getBorders(mask)
    m,n = mask.shape
    col = d-u
    row = r-l
    MORE_DEL_LINE = 2
    if col/float(m)<row/float(n):
        return col+MORE_DEL_LINE,True
    return row+MORE_DEL_LINE,False


def smartSc(delkinds,masks,img):
    img = img.copy()
    allkinds = set([i for i in np.unique(masks) if i])
    delkinds = set(delkinds)
    savekinds = allkinds.difference(delkinds)
    
    delmask = sum([masks == kind for kind in delkinds])!=0
    savemask = sum([masks == (masks.max()+1)]+[masks == kind for kind in savekinds])!=0 
    
    #show([img,sal,masks])
#    print allkinds,delkinds,savekinds
#    print np.unique(masks)
    
    delmask = smoothBinImag(delmask)
    savemask = smoothBinImag(savemask)
    
    m,n = masks.shape
    #masks[200:,:]=0
    deln,isrow = getDelnIsRow(delmask)
    
    scMask = np.zeros(masks.shape,int)
    scMask[delmask] -= 1
    scMask[savemask] += 1
#    show(scMask.copy())
    newimg = seamCarveBlack(img,deln,mask=scMask,row=isrow)
#    print newimg.shape,img.shape
#    show([img,newimg])
    return newimg

classList = ["背景", "飞机", "自行车", "鸟", "船", "瓶子", "公交车", "汽车", "猫", "椅子", "牛", "桌子", "狗", "马", "摩托车", "人", "飞机", "羊", "沙发", "火车", "屏幕"]
kindToName = dict(enumerate(classList))
nameToKind = dict([(j,i) for i,j in enumerate(classList)])

if __name__ == '__main__':
    1

    dirr = './imgs/hku/'
    paths = [dirr+i for i in  os.listdir(dirr) if '.jpg' in i]
    
    name = '2750'
    
    name = '0302'
    name = '0317'
    name = '0254'
    name = '1206'
    name = '0084'

    path = [i for i in paths if name in i][0]
    img = io.imread(path)
    sal = io.imread(path[:path.rindex('.')]+'_sal.png')
    masks = io.imread(path[:path.rindex('.')]+'_SS.png')
    delkinds = [12]
    newimg=smartSc(delkinds,masks,img)
    show(newimg)

 
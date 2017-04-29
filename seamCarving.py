# -*- coding: utf-8 -*-
"""
Created on Thu Jul 28 00:36:24 2016

@author: yl
"""
from __future__ import unicode_literals

from imgProcessConfig import *

from imgProcessTools import *
from imgProcessTools import (Axes3D, FormatStrFormatter, Fun, HIS_BACKGROUND_COLOR, HIS_MAX, LinearLocator, MAX_FILTER_R, SHOW_SHAPE, View, allFilter, applyColorTable, arange, array, avgFilter, base64, base64Img, bg, bilateralFilter, blackToWhiteVidoEffects, c, cProfile, cm, crun, cv2, da, draw3dSurface, filterr, g, gaussCore, getColorTables, getFilterImg, getLine, gussFilter, ind, interpolate, io, log, loga, mapp, math, maxFilter, mdeianFilter, normalizing, np, os, plt, polt3dSurface, py3, pyv, r, random, roundInt, run, saveAvi, scipy, show, sk, skimage, smallImg, stand, step, sys, tan, unicode_literals)
from skimage import filters
from skimage import transform
import numpy as np
cvSobel = filters.sobel

r = random(3,5)

#show([img,mag])
def getSobel(rgb,mask=None):
    img = rgb.astype(int)
    tmp = np.mean(np.abs(img[:-1,:-1]-img[1:,:-1])+np.abs(img[:-1,:-1]-img[:-1,1:]),2)
    mag = np.append(tmp,tmp[-1:],0)
    mag = np.append(mag,mag[:,-1:],1)
    if mask is not None:
        mag+= mask
    return mag.astype(int)
#mag = mySobel(img)
#show([mag,magg])

def reduceOneLine(img,mag,mask=None):
    m, n = mag.shape
    expand = lambda row: np.append(np.append(row[:1],row),row[-1:])
    accu = [mag[0]]
    for ind in range(1,m):
        tmp = np.zeros((n,3),mag.dtype)
        row = expand(accu[ind-1])
        tmp[...,0] = row[:-2]
        tmp[...,1] = row[1:-1]
        tmp[...,2] = row[2:]
        add = np.min(tmp,1)
        accu += [add+mag[ind]]
    
    accu = np.array(accu)
    accui = np.append([[accu.max()+1]]*m,accu,1)
    
    _ind = np.argmin(accui[-1])
    ind = _ind - 1
    pop = lambda ind,row: np.append(row[:ind],row[ind+1:],0).astype(row.dtype)
    newimg = [pop(ind,img[-1])]
    if mask is not None:
        newmask = [pop(ind,mask[-1])]
    for r in range(m-1)[::-1]:
        d = np.argmin(accui[r][ind:ind+3])-1
        ind = ind+d
        newimg = [pop(ind,img[r])]+newimg
        if mask is not None:
            newmask = [pop(ind,mask[r])]+newmask
    newimg = np.array(newimg)
    if mask is not None:
#        print newmask
        return newimg.astype(img.dtype),np.array(newmask)
    return newimg.astype(img.dtype),None
def seamCarve(img,deln,mask=None,each=False,f=None,row=None):
    if mask is not None:
        if isinstance(mask[0][0],float):
            mask -= skimage.filters.threshold_li(mask)
            mask *= 512
            mask.astype(int)
        else:
            mask[mask==1]=512
            mask[mask==-1]=-512
    if row:
        img=np.transpose(img,[1,0,2])
        mask = None if mask is None else mask.T 
    newimg = img
    for i in range(deln):
        mag = getSobel(newimg,mask)
        newimg,mask = reduceOneLine(newimg,mag,mask)
        if f and i % each==0:
            f(newimg,i+1,row)
    if row:
        newimg=np.transpose(newimg,[1,0,2]) 
    return newimg


def reduceOneLineWithHed(img,mag,mask=None):
    m, n = mag.shape
    expand = lambda row: np.append(np.append(row[:1],row),row[-1:])
    accu = [mag[0]]
    for ind in range(1,m):
        tmp = np.zeros((n,3),mag.dtype)
        row = expand(accu[ind-1])
        tmp[...,0] = row[:-2]
        tmp[...,1] = row[1:-1]
        tmp[...,2] = row[2:]
        add = np.min(tmp,1)
        accu += [add+mag[ind]]
    
    accu = np.array(accu)
    accui = np.append([[accu.max()+1]]*m,accu,1)
    
    _ind = np.argmin(accui[-1])
    ind = _ind - 1
    pop = lambda ind,row: np.append(row[:ind],row[ind+1:],0).astype(row.dtype)
    newimg = [pop(ind,img[-1])]
    hed = mag
    hed = [pop(ind,mag[-1])]
    if mask is not None:
        newmask = [pop(ind,mask[-1])]
    for r in range(m-1)[::-1]:
        d = np.argmin(accui[r][ind:ind+3])-1
        ind = ind+d
        newimg = [pop(ind,img[r])]+newimg
        hed = [pop(ind,mag[r])]+hed
        if mask is not None:
            newmask = [pop(ind,mask[r])]+newmask
    newimg = np.array(newimg)
    hed = np.array(hed)
    if mask is not None:
#        print newmask
        return newimg.astype(img.dtype),hed,np.array(newmask)
    return newimg.astype(img.dtype),hed,None
    
def seamCarveWithHed(img,deln,mask=None,each=False,f=None,row=None,hed=None):
    if mask is not None:
        if isinstance(mask[0][0],float):
            mask -= skimage.filters.threshold_li(mask)
            mask *= 512
            mask.astype(int)
        else:
            mask[mask==1]=512
            mask[mask==-1]=-512
    if row:
        img=np.transpose(img,[1,0,2])
        mask = None if mask is None else mask.T 
    newimg = img
    mag = hed.T if row else hed
    for i in range(deln):
        newimg,mag,mask = reduceOneLineWithHed(newimg,mag,mask)
        if f and i % each==0:
            f(newimg,i+1,row)
    if row:
        newimg=np.transpose(newimg,[1,0,2]) 
    return newimg
    
def addBlack(img,i,row):
    new = np.append(img,[[[0,0,0]]*i]*img.shape[0],1).astype(img.dtype)
#    new = np.append(img,[[[0,0,0] for _ in range(i)] for __ in range(img.shape[0])],1)
    new = np.transpose(new,[1,0,2])  if row else new
    return new
scShow = lambda img,i,row: show(addBlack(img,i,row))

def seamCarveBlack(img,deln,mask=None,each=False,f=None,row=None,):
    imgNoBlack = seamCarve(img,deln,mask=mask,each=each,f=f,row=row,)
    imgNoBlack = np.transpose(imgNoBlack,[1,0,2]) if row else imgNoBlack
    newimg = addBlack(imgNoBlack,deln,row)
    return newimg
    
    

#sc = transform.seam_carve(rgb,mag,'vertical',deln)
#show(sc)
#%%
if __name__ == '__main__':
    from imgProcess import *
    rgb = da.astronaut()
    img = rgb.copy()
    img = g['img']
    sal = g['sal']
    deln = 100
    mask = np.zeros(img.shape[:2],int)
#    mask[:150] = -1
#    mask = None
    mask  = sal
    row = True
    newimg = seamCarveBlack(img,deln,mask,deln//3,scShow,row)
    hed = g['hed']
#    newimg2 = seamCarveWithHed(img,deln,mask,deln//3,scShow,row,hed=hed)
    show(newimg)
#    show(newimg2)
    show([hed,getSobel(img)])
    pass

    

    
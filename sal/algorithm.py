# -*- coding: utf-8 -*-
from __future__ import unicode_literals
'''
most of this module is algorithm ,
and there still has some tools funcation which would use constant like IMG_DIR
'''
from os import listdir
import os
from copy import deepcopy

import numpy as np 
import scipy as sp 
import matplotlib.pyplot as plt  
import skimage as sk
import skimage.io as io
from skimage import data as da
from skimage.feature import local_binary_pattern
from skimage.segmentation import mark_boundaries
import cv2

from tools import mapp,show,getPoltName,performance,normalizing,random,loga
from tools import getElm
from tools import getSlic
from tools import getEdge,getNeighborMatrix,getNeighbor,valueToLabelMap
from tools import loadData,saveData

import saliency as sal
IMG_DIR=sal.IMG_DIR
COARSE_DIR=sal.COARSE_DIR
IMG_NAME_LIST=sal.IMG_NAME_LIST 
LABEL_DATA_DIR=sal.LABEL_DATA_DIR

a = random(3,5)

def showpr(imgName=1,methods=["MY4","ME1","MEAN", "DRFI","QCUT","DISC2"],lab=False):
    if isinstance(imgName, int):
        imgName = IMG_NAME_LIST[imgName]
    img =  io.imread(IMG_DIR+imgName)/255.
    imgGt = io.imread(IMG_DIR+imgName[:-3]+'png')!=0
    coarseDic = getCoarseDic(imgName,methods)
    imgs = []
    methods = []
    for k,coarseImg in coarseDic.items():
        imgs += [mark_boundaries(normalizing(coarseImg),imgGt)]
        methods += [k]
    print '='*40
    print IMG_NAME_LIST.index(imgName),imgName
    print 'methods:',' || '.join(methods)
    show(imgs)
    show(mark_boundaries(img,imgGt))
    from analysis import plotImgPr
    plotImgPr(imgGt,coarseDic)


    
#    img[imgGt] = (1.-img)[imgGt] 
#    show(mark_boundaries(img,imgGt))
    
    

def readImg(imgName=0):
    if isinstance(imgName, int):
        imgName = IMG_NAME_LIST[imgName]
    img =  io.imread(IMG_DIR+imgName)/255.
    imgGt = io.imread(IMG_DIR+imgName[:-3]+'png')/255.    
    return img,imgGt

def readCoarseMap(imgName,method,coarseDir=None):
    coarseDir = coarseDir if coarseDir else COARSE_DIR
    saliency_map = io.imread(coarseDir+imgName[:-4]+'_'+method+'.png')
    return saliency_map.astype(float)/saliency_map.max()

def getCoarseDic(imgName, methods,coarseDir=None):
    '''
    return dic of coarse imgs,
    which key is method
    '''
    coarsesDic = {}
    for method in methods:
        coarsesDic[method] = readCoarseMap(imgName, method,coarseDir) 
    return coarsesDic

def getSumCoarseImg(coarsesDic):
    '''
    merge and normalizing to one coarseImg
    '''
    coarseImg = integratImgsBy3way(coarsesDic.values())
    return coarseImg

def getLbp(img, labelMap, returnRawLbp=False, METHOD = 'uniform'):
    '''
    return N*59 matrix, each super pixel`s lbp histogram
    **img must be lab or grey**
    '''
    img = sk.color.rgb2gray(img)
    if len(img.shape) != 2 :
        img = sk.color.rgb2gray(sk.color.lab2rgb(img))
    
    RADIUS = 3  # LBP radius
    n_points = 57
    lbp = local_binary_pattern(img, n_points, RADIUS, METHOD)
    lbp = lbp.astype(int)
    
    lbpLen = lbp.max()+1
    m, n = labelMap.shape
    maxLabel = labelMap.max()+1
    
    lbpList = []
    for label in range(maxLabel):   
        mask = labelMap == label
        lbpHistogram = np.array([0]*lbpLen)
        for i in lbp[mask]:
            lbpHistogram[i] += 1
        lbpList += [np.array(lbpHistogram).astype(float)/mask.sum()]
    if returnRawLbp:
        return np.array(lbpList),lbp
    return np.array(lbpList)
    
def getDistance(labelMap):
    '''
    return a N*N martrix, the spatial distance(infinite norm) of any two super pixel 
    see paper Eq(2)
    
    '''
    m, n = labelMap.shape
    maxLabel = labelMap.max()+1
    pos = []
    m,n = labelMap.shape
    for label in range(maxLabel):
        mask = labelMap==label
        x = (mask.sum(axis=0)*np.array(range(n))).sum()/float(mask.sum())
        y = (mask.sum(axis=1)*np.array(range(m))).sum()/float(mask.sum())
        pos += [(x,y)]
    def f_distance(_, i, j):
        if i==j:
            return 0.0
#        dis = ((pos[i][0]-pos[j][0])**2+(pos[i][1]-pos[j][1])**2)**0.5
        '''see paper Eq(2)'''
        dis = max([abs(pos[i][0]-pos[j][0])/float(n),abs(pos[i][1]-pos[j][1])/float(m)])
        # infinity norm distance
        return dis
    
    distanceMa = mapp(f_distance, 
                      np.zeros((maxLabel,maxLabel)),
                        need_i_j=True)
    #io.imshow(distanceMa[:,:])
    return distanceMa

#def getAff(img, labelMap, use_vector=False, alpha=0.99, delta=0.1):
#    mr = MR.MR_saliency(alpha, delta)
#    aff = mr._MR_saliency__MR_affinity_matrix(img,labelMap,use_vector)
#    #show([aff, sk.exposure.equalize_hist(aff)])
#    return aff


def getColorVector(img, labelMap):
    '''
    return average value of each super pixel.
    
    when img is gray which means len(img.shape) is 2, 
    should return a vector like [[0.1],[0.2],..[0.5]]
    instead of [0.1,0.2,..0.5]
    '''
    maxLabel = labelMap.max()+1
    vector = np.array(map(lambda label: 
                                np.mean(img[labelMap == label],0) if len(img.shape) != 2 
                                else [np.mean(img[labelMap == label],0)],
                            range(maxLabel)))
    return vector


def getW(vector):
    '''
    Gaussian functions. 
    return W The color similarity between two superpixels 
    see paper Eq.(3)    
    
    '''
    exp = sp.exp
    Dr = lambda vector: sp.spatial.distance.squareform(sp.spatial.distance.pdist(vector))
    
    Drs = Dr(vector)
    '''i.e., σrc = sr · max i, j Dr(ric, rcj)
    see paper IV.A "Experimental Setup" last sentence
    '''
    sigma = Drs.max()*0.5
    _W = exp(-Drs**2/(2*sigma**2))
#    print _W.min(),Drs.max()
    W = np.array(map(lambda row: row/row.sum(), _W))
    
#    io.imshow(sk.exposure.equalize_hist(_W))
    return W
    
    
def getWs(img, labelMap):
    '''
    return list of 8 W of img, and Ws sequence is
        color*[Lab, L, a, b] + texture*[Lab, L, a, b]
    see paper Eq.(3)
    '''
    Ws = map(lambda image: getW(
                                    getColorVector(image,labelMap)), 
               [img,img[...,0],img[...,1],img[...,2]]
               ) 
               
    lbpWs = map(lambda image: getW(
                                    getLbp(image,labelMap))
#                                       ,labelMap,True) 
               ,[img,img[...,0],img[...,1],img[...,2]]
               )
    
    Ws += lbpWs
    return Ws

def getVectors(img,labelMap):
    '''
    color and texture scatter degree 
    see paper Eq.(5), Eq.(6)
    '''
    maxLabel = labelMap.max()+1
    
    dis = getDistance(labelMap)
    
    Ws = getWs(img, labelMap)
#    map(lambda img: show(sk.exposure.equalize_hist(img)),Ws)
    vectors = []
    for i in range(maxLabel):
        summ = np.array([0.]*len(Ws))
        for j in range(maxLabel):
            summ += dis[i,j]*np.array([W[i,j] for W in Ws])
        vectors += [1/summ]
    return np.array(vectors), Ws
    
def getWeightSum(labelMap, vectors, Ws):
    '''
    compactness :weighted sum of scatter degree
    see paper Eq(7)
    '''
    maxLabel = labelMap.max()+1
    weightSum = []
    for label in range(maxLabel):
        colorWeightSums = map(
                             lambda colorIndex :sum(
                                                    map(lambda j: Ws[colorIndex][label,j]
                                                        *vectors[j][colorIndex], range(maxLabel))
                                                    )
                             ,range(4))
        lbpWeightSums = map(
                             lambda lbpIndex :sum(
                                                    map(lambda j: Ws[lbpIndex][label,j]
                                                        *vectors[j][lbpIndex], range(maxLabel))
                                                    )
                             ,range(4,8))
        weightSum += [colorWeightSums + lbpWeightSums]
                        
    return np.array(weightSum)
    


def getCoarseTrain(coarseImg, labelMap):
    '''
    return coarseTrain (n*2 Matrix), collected by super pixel 
    return vectorsTrainTag (1 dimension array of bool), tag which super pixel to train 
    see paper III.D "Refinde Saliency Map" first pragraph
    '''
    omega = np.mean(coarseImg)
    alpha = 0.8
    th = min([0.9,(1+alpha)*omega])
    tl = min([0.1,(1-alpha)*omega])
    maxLabel = labelMap.max()+1
    
    coarseTrain = []
    vectorsTrainTag = [True]*maxLabel
    for label in range(maxLabel):
        mask = label == labelMap
        mean = np.mean(coarseImg[mask])
        if mean >= th:
            coarseTrain += [(1, 0)]
        elif mean <= tl:
            coarseTrain += [(0, 1)]
        else:
            vectorsTrainTag[label]=False
    return np.array(coarseTrain), np.array(vectorsTrainTag)

@performance
def getRefindImgsOneElm(img,
             coarseImgs,
             labelMap
             ):
    '''MY1
    use all coarse imgs to train one elm, than, predict one refinedImg
    '''
    img = sk.color.rgb2lab(img)
    #show([mark_boundaries(img,labelMap),imgGt])
    # 获得4+4维  distance
    degreeVectors, Ws = getVectors(img, labelMap)
    vectors = getWeightSum(labelMap, degreeVectors, Ws)
    
    vectorsTrains = []
    coarseTrains = []
    for coarseImg in coarseImgs:
        coarseTrain, vectorsTrainTag = getCoarseTrain(coarseImg, labelMap)
        vectorsTrains += list(vectors[vectorsTrainTag])
        coarseTrains += list(coarseTrain)
    
    elm = getElm(np.array(vectorsTrains), np.array(coarseTrains))
    refined = elm.predict(vectors)[:,0]
    refinedImg = valueToLabelMap(labelMap,normalizing(refined))
    return refinedImg

@performance
def getRefindImgsManyElm(img,
             coarseImgs,
             labelMap
             ):
    '''MY2
    for each coarse img train a elm, each elm predict a refinedImg
    merge refinedImgs and normalizing to one refinedImg
    '''
    img = sk.color.rgb2lab(img)
    #show([mark_boundaries(img,labelMap),imgGt])
    # 获得4+4维  color and texture scatter degree
    degreeVectors, Ws = getVectors(img, labelMap)
    # 
    vectors = getWeightSum(labelMap, degreeVectors, Ws)
    refinedImgs = []
    for coarseImg in coarseImgs:
        coarseTrain, vectorsTrainTag = getCoarseTrain(coarseImg, labelMap)
        elm = getElm(vectors[vectorsTrainTag], coarseTrain)
        refined = elm.predict(vectors)[:,0]
        refinedImg = valueToLabelMap(labelMap,normalizing(refined))
        refinedImgs += [refinedImg]
    
    # 合并 归一
    refinedImgSum = integratImgsBy3way(refinedImgs)
    return refinedImgSum


@performance
def getRefindImgsOneElmAddLabAndLbp(img,
             coarseImgs,
             labelMap):
    '''MY3
    use all coarse imgs to train one elm, than, predict one refinedImg
    add Lab
    '''
    img = sk.color.rgb2lab(img)
    #show([mark_boundaries(img,labelMap),imgGt])
    # 获得4+4维  distance
    degreeVectors, Ws = getVectors(img, labelMap)
    weightSumVectors = getWeightSum(labelMap, degreeVectors, Ws)
    
    '''add lab lbp'''
    labVectors = getColorVector(img,labelMap)
    lbpVectors = getLbp(img,labelMap)
    vectors = np.c_[weightSumVectors,labVectors,lbpVectors]
#    print 'weightSumVectors labVectors vectors.shape',weightSumVectors.shape,labVectors.shape,vectors.shape
    
    
    vectorsTrains = []
    coarseTrains = []
    for coarseImg in coarseImgs:
        coarseTrain, vectorsTrainTag = getCoarseTrain(coarseImg, labelMap)
        vectorsTrains += list(vectors[vectorsTrainTag])
        coarseTrains += list(coarseTrain)
    
    elm = getElm(np.array(vectorsTrains), np.array(coarseTrains))
    refined = elm.predict(vectors)[:,0]
    refinedImg = valueToLabelMap(labelMap,normalizing(refined))
    return refinedImg


def getLabelWeightSum(label,
                      otherLabel,
                      W,
                      distanceMatrix,):
    '''Paper Eq.(5)'''
   #lrr:remove refluence of distance
   #scatter = 1./sum([distanceMatrix[label][j]*W[label][j] for j in otherLabel])
    #scatter=1./sum([W[label][j] for j in otherLabel])
    scatter=1
    weightSum = sum([W[label][j]*scatter for j in otherLabel])
    return weightSum
def getDiffEdgeAndNeighbor(labelMap,W,edge,neighborMatrix,distanceMatrix): 
    '''
    
    '''
    maxLabel = labelMap.max()+1
    diffEdge=np.array(map(lambda label:getLabelWeightSum(label,edge,W,distanceMatrix),
                          range(maxLabel)))
#    show(valueToLabelMap(labelMap,diffEdge))
    diffNeighbor = []
    for label in range(maxLabel):
        neighbors = getNeighbor(label,labelMap,neighborMatrix,1).keys()
        diffNeighbor += [getLabelWeightSum(label,neighbors,W,distanceMatrix)]
#    show(valueToLabelMap(labelMap,diffNeighbor))
    return diffEdge,diffNeighbor
def getAllDiffEdgeAndNeighbor(labelMap,Ws):
    '''
    diffEdges,diffNeighbors both shape are N*8 , 
    the 8 mean color*[Lab, L, a, b] + texture*[Lab, L, a, b] respectively  
    '''
    edge = getEdge(labelMap)
    neighborMatrix = getNeighborMatrix(labelMap)    
    distanceMatrix = getDistance(labelMap)
    temp = [getDiffEdgeAndNeighbor(labelMap,W,edge,neighborMatrix,distanceMatrix) for W in Ws]
    diffEdges = [diffEdge for diffEdge,diffNeighbor in temp]
    diffNeighbors = [diffNeighbor for diffEdge,diffNeighbor in temp]
    return np.transpose(np.array(diffEdges)),np.transpose(np.array(diffNeighbors))

    
@performance
def my4diffEdge(img,
             coarseImgs,
             labelMap
             ):
    '''MY4
    add different from Edge 
    '''
    img = sk.color.rgb2lab(img)
    # 获得4+4维  distance
    degreeVectors, Ws = getVectors(img, labelMap)
    weightSumVectors = getWeightSum(labelMap, degreeVectors, Ws)
    
    diffEdges,diffNeighbors = getAllDiffEdgeAndNeighbor(labelMap,Ws)

    vectors = np.append(weightSumVectors,diffEdges,1)
#    vectors = np.append(vectors,diffNeighbors,1)

    vectorsTrains = []
    coarseTrains = []
    for coarseImg in coarseImgs:
        coarseTrain, vectorsTrainTag = getCoarseTrain(coarseImg, labelMap)
        vectorsTrains += list(vectors[vectorsTrainTag])
        coarseTrains += list(coarseTrain)
    
    elm = getElm(np.array(vectorsTrains), np.array(coarseTrains))
    refined = elm.predict(vectors)[:,0]
    refinedImg = valueToLabelMap(labelMap,normalizing(refined))
    mask = grabCut(sk.color.lab2rgb(img),refinedImg)
    refinedImg = normalizing(mask*refinedImg)
    return refinedImg
    
@performance
def my5diffEdgeAndNeighbor(img,
             coarseImgs,
             labelMap
             ):
    '''MY5
    add different from Edge And Neighbor
    '''
    img = sk.color.rgb2lab(img)
    # 获得4+4维  distance
    degreeVectors, Ws = getVectors(img, labelMap)
    weightSumVectors = getWeightSum(labelMap, degreeVectors, Ws)
    
    diffEdges,diffNeighbors = getAllDiffEdgeAndNeighbor(labelMap,Ws)

    vectors = np.append(weightSumVectors,diffEdges,1)
    vectors = np.append(vectors,diffNeighbors,1)

    vectorsTrains = []
    coarseTrains = []
    for coarseImg in coarseImgs:
        coarseTrain, vectorsTrainTag = getCoarseTrain(coarseImg, labelMap)
        vectorsTrains += list(vectors[vectorsTrainTag])
        coarseTrains += list(coarseTrain)
    
    elm = getElm(np.array(vectorsTrains), np.array(coarseTrains))
    refined = elm.predict(vectors)[:,0]
    refinedImg = valueToLabelMap(labelMap,normalizing(refined))
    return refinedImg
    

'''
buildMethodDic:
k:方法名称的缩写
v:对应函数
'''
buildMethodDic={
    'MY1':getRefindImgsOneElm,# 全部 coarseImgs 用于训练一个elm 预测一个 refinedImg
    'MY2':getRefindImgsManyElm,# 每个 coarseImgs 都训练一个elm 将所有预测的 refinedImg 合并
    'MY3':getRefindImgsOneElmAddLabAndLbp, # 对getRefindImgsOneElm 增加Lab LBP特征
    'MY4':my4diffEdge, # add different from Edge
    'MY5':my5diffEdgeAndNeighbor, # add different from Edge and neighbor
}

@performance
def buildImgs(  imgName,
            buildMethods,
            coarseMethods,
            segmentList=[200,250,750],
            compactness=20 ):
    '''
    do every funcation in buildMethods
    '''
    if isinstance(imgName, int):
        imgName = IMG_NAME_LIST[imgName]
    print 'img index:%d/%d'%(IMG_NAME_LIST.index(imgName),len(IMG_NAME_LIST))
    img,imgGt = readImg(imgName)
    
    #讨论：只能分两类 更多没用
    coarseDic = getCoarseDic(imgName,coarseMethods)
    #show(coarsesDic)   
    sumCoarseImg = getSumCoarseImg(coarseDic)
    coarseImgs=coarseDic.values()
    coarsePath = COARSE_DIR+('%s_MEAN.png' % imgName[:imgName.rindex('.')])
    io.imsave(coarsePath,sumCoarseImg)
    
    labelMapDic = {}
    for n_segments in segmentList:
        labelMapDic[n_segments] = getSlic(img,n_segments,compactness)
        
    refinedImgs = []
    for buildMethod in buildMethods:
        funcation = buildMethodDic[buildMethod]
        _refinedImgs = []
        for n_segments in segmentList:
            
            _refinedImg= funcation(img=img,
                                  coarseImgs=coarseImgs,
                                  labelMap=labelMapDic[n_segments])
            _refinedImgs += [_refinedImg]
        
        refinedImg = integratImgsBy3way(_refinedImgs)
        refinedImgs += [refinedImg]
    
        methodNameFormat = '%s_'+buildMethod+'.png'
        path = COARSE_DIR+(methodNameFormat % imgName[:imgName.rindex('.')])
        io.imsave(path,refinedImg)
    
    show([img,imgGt,sumCoarseImg],lab=False)
    print 'buildMethods: ',' || '.join(buildMethods)
    show(refinedImgs)
    return refinedImgs

def integratImgsBy3way(refinedImgs):
    def f(refinedImg):    
        h,w = refinedImg.shape[:2]
        a,b,c,d = int(0.25*h),int(0.75*h),int(0.25*w),int(0.75*w)
        center = refinedImg[a:b,c:d].sum()
        ratio = float(center)/(refinedImg.sum()-center)
        
        m,n = refinedImg.shape
        distribut = 1./mapp(lambda x,i,j:float(x)*((i-m/2)**2+(j-n/2)**2) ,refinedImg,True).sum()
        
        var = np.var(refinedImg)
        return ratio,distribut,var
    
    l = np.array(map(f,refinedImgs))
    l = l/l.max(0)
    l = l.sum(1)
    mergeImg = sum(map(lambda x:x[0]*x[1],zip(refinedImgs,l)))
    mergeImg = normalizing(mergeImg)
    return mergeImg
'''
mergeMethodDic:
k:方法名称的缩写
v:None #说明
'''
mergeWayDic={
    'ME1':None, # sum of all mergeMethods
    'ME2':None, # mul of all mergeMethods
    'ME3':integratImgsBy3way,
}

@performance
def mergeImgs(  imgName,
            mergeMethods):
    '''
    生成合并图像
    '''
    if isinstance(imgName, int):
        imgName = IMG_NAME_LIST[imgName]
    print 'img index:%d/%d'%(IMG_NAME_LIST.index(imgName),len(IMG_NAME_LIST))
    img =  io.imread(IMG_DIR+imgName)/255.
    imgGt = io.imread(IMG_DIR+imgName[:-3]+'png')/255.
    
    coarseDic = getCoarseDic(imgName,mergeMethods)
    sumCoarseImg = sum(coarseDic.values())
    sumCoarseImg = sumCoarseImg/sumCoarseImg.max()
    
    mulImg = reduce(lambda x, y: x*y, coarseDic.values())
    mulImg=normalizing(mulImg)
    
    integratImg = integratImgsBy3way(coarseDic.values())
    
    coarsePath = COARSE_DIR+('%s_ME1.png' % imgName[:imgName.rindex('.')])
    io.imsave(coarsePath,sumCoarseImg)
    coarsePath = COARSE_DIR+('%s_ME2.png' % imgName[:imgName.rindex('.')])
    io.imsave(coarsePath,mulImg)
    coarsePath = COARSE_DIR+('%s_ME3.png' % imgName[:imgName.rindex('.')])
    io.imsave(coarsePath,integratImg)
    
    show([img,imgGt],lab=False)
    print 'mergeMethods: ',' || '.join(mergeMethods)
    show(map(lambda k:coarseDic[k],mergeMethods))
    print 'resoult: ',' || '.join(['ME1','ME2','ME3'])
    show([sumCoarseImg,mulImg,integratImg])

    

def grabCut(img, refinedImg=None):
    mask = np.zeros(img.shape[:2],np.uint8)   # img.shape[:2] = (413, 620)
    
    bgdModel = np.zeros((1,65),np.float64)
    fgdModel = np.zeros((1,65),np.float64)
    
    rect = tuple((0,0)+img.shape[:2])
    imgUint8 = (img*255.9999999999).astype(np.uint8)
    if refinedImg != None:
        omega = np.mean(refinedImg)
        alpha = 0.8
        th = min([0.9,(1+alpha)*omega])
        tl = min([0.1,(1-alpha)*omega])
        mask = mask + 4
        mask[np.where(refinedImg<tl,True,False)] = 0
        mask[np.where(refinedImg>th,True,False)] = 1
        mask[np.where((tl<=refinedImg)&(refinedImg<=th),True,False)] = 3
        tmp = deepcopy(mask)
        cv2.grabCut(imgUint8,mask,rect,bgdModel,fgdModel,5,cv2.GC_INIT_WITH_MASK)
#        show([tmp,mask])
    else:
        # this modifies mask 
        cv2.grabCut(imgUint8,mask,rect,bgdModel,fgdModel,5,cv2.GC_INIT_WITH_RECT)
    # If mask==2 or mask== 1, mask2 get 0, other wise it gets 1 as 'uint8' type.
    mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
    return mask2



if __name__ == '__main__':
    from saliency import *
    segmentList = [200,300,400]
    compactness = 20
    buildMethods = ['MY4']
    img = da.astronaut()
    labelMapDic = {}
    for n_segments in segmentList:
        labelMapDic[n_segments] = getSlic(img,n_segments,compactness)
    refinedImgs = []
    for buildMethod in buildMethods:
        funcation = buildMethodDic[buildMethod]
        _refinedImgs = []
        for n_segments in segmentList:
            
            _refinedImg= funcation(img=img,
                                  coarseImgs=coarseImgs,
                                  labelMap=labelMapDic[n_segments])
            _refinedImgs += [_refinedImg]
        
        refinedImg = integratImgsBy3way(_refinedImgs)
        refinedImgs += [refinedImg]
#    return refinedImgs
    show(refinedImgs)    
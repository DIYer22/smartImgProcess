# -*- coding: utf-8 -*-
from __future__ import unicode_literals
'''
函数功能测试
'''
from algorithm import io,np,sk,da
from algorithm import show,random,listdir,mark_boundaries,loga
from algorithm import getSlic,readImg
import algorithm as alg



a=random(3,5)
G = {} # G is a global var to save value for DEBUG in funcation
def setModuleConstant(module):
    module.IMG_DIR=IMG_DIR
    module.COARSE_DIR=COARSE_DIR
    module.IMG_NAME_LIST=IMG_NAME_LIST
    
#IMG_DIR = r'E:\3-experiment\SalBenchmark-master\Data\DataSet1\Imgs/'
#COARSE_DIR =r'E:\3-experiment\SalBenchmark-master\Data\DataSet1\Saliency/'

#IMG_DIR =  '../DataSet1/Imgs/'
#COARSE_DIR ='../DataSet1/Saliency/'   
#   
#IMG_DIR =  r'E:\3-experiment\SalBenchmark-master\Data\MSRA\imgs/'
#COARSE_DIR =r'E:\3-experiment\SalBenchmark-master\Data\MSRA\Saliency/'

#IMG_NAME_LIST = filter(lambda x:x[-3:]=='jpg',listdir(IMG_DIR))

#setModuleConstant(alg)
from saliency import *
imgName = IMG_NAME_LIST[3]
img,imgGt = readImg(imgName)
rgbImg = img

rgbImg = np.zeros((100,100,3))
rgbImg[25:75,25:75,1:]=1.
#show(rgbImg)
allMethods =['DISC2','QCUT','ME1']

def integratImgsBy3wayTest():
    from algorithm import getSlic,getCoarseDic,integratImgsBy3way,buildMethodDic
    labelMap = getSlic(rgbImg,300)
    img = rgbImg
    m, n = labelMap.shape 
    
    coarseMethods = ['MEAN']
    coarseDic = getCoarseDic(imgName,coarseMethods)
#    show(coarseDic)   
    coarseImgs=coarseDic.values()       
    
    refinedImgs=map(lambda f:f(img,coarseImgs,labelMap),buildMethodDic.values())
    img = integratImgsBy3way(refinedImgs)
    show(img)
    

def my4test():
#if 1:
    from algorithm import *
    labelMap = getSlic(rgbImg,200)
    maxLabel = labelMap.max()+1
    m, n = labelMap.shape 
    
    coarseMethods = ['MEAN']
    coarseDic = getCoarseDic(imgName,coarseMethods)
#    show(coarseDic)   
    sumCoarseImg = getSumCoarseImg(coarseDic)
    coarseImgs=coarseDic.values()    

    img = sk.color.rgb2lab(rgbImg)
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
    show(mark_boundaries(sk.color.lab2rgb(img),labelMap))
    print diffEdges.shape
    show(valueToLabelMap(labelMap,diffEdges.sum(1)))
    show(valueToLabelMap(labelMap,diffNeighbors[:,:4].sum(1)))
#    show(valueToLabelMap(labelMap,diffNeighbors[:,4:].sum(1)))
#    show(valueToLabelMap(labelMap,diffNeighbors.sum(1)))
    show(valueToLabelMap(labelMap,vectors[:,4:8].sum(1)))
#my4test()

def getEdgeNeighborTest():
    from algorithm import getEdge,getNeighborMatrix,getNeighbor
    
    labelMap = getSlic(img,200) 
    show(mark_boundaries(img,labelMap),0)
    edge = getEdge(labelMap)
    neighborMatrix = getNeighborMatrix(labelMap)    
    m, n = labelMap.shape 
    dic = getNeighbor(0,labelMap,neighborMatrix,4)
    print dic
    imgg = np.zeros((m,n))
    for k in dic:
        imgg[labelMap==k]=dic[k]
    show(imgg)

    imgg = np.zeros((m,n))
    for k in edge:
        imgg[labelMap==k]=1
    show(imgg)
    edge = getEdge(labelMap,0.06)
    imgg = np.zeros((m,n))
    for k in edge:
        imgg[labelMap==k]=1
    print 'edge width 0.06'
    show(imgg)    

#getEdgeNeighborTest()
def getDistanceTest():
    from algorithm import getDistance
    
    m,n=20,10
    ma = np.zeros((m,n)).astype(int)
    ma[:5,5:]=1
    ma[5:,:5]=2
    ma[5:,5:]=3
    
    dis = getDistance(ma)
    print ma
    print dis[0][2]
    print dis[0][1]


def getWeightSumTest():
    from algorithm import getLbp ,getVectors,getWeightSum
    labelMap = getSlic(img,200) 
    maxLabel = labelMap.max() + 1
    im = sk.color.rgb2lab(img)
    degreeVectors, Ws = getVectors(im, labelMap)
    vectors = getWeightSum(labelMap, degreeVectors, Ws)
    m,n = labelMap.shape
    imgg = np.zeros((m,n))
    imgg2 = np.zeros((m,n))
    order = ['lab','l','a','b','lab-texture','l-texture','a-texture','b-texture']
    labs = [im]+ [im[:,:,i] for i in range(3)]
    lbps = map(lambda c: getLbp(c,labelMap,1)[1],labs)
    labLbp = labs + lbps
    for color in range(vectors.shape[1]):
        for k in range(maxLabel):
            imgg[labelMap==k]=vectors[k][color]
            imgg2[labelMap==k]=degreeVectors[k][color]
        print order[color],'raw | scatter | weight sum'
#        show(sk.exposure.equalize_hist(imgg))
        show([labLbp[color],imgg2,imgg],1)
    loga(degreeVectors)
    loga(vectors)
    





def getRefindImgsTest():
    IMG_DIR = r'E:\3-experiment\SalBenchmark-master\Data\DataSet1\Imgs/'
    COARSE_DIR =r'E:\3-experiment\SalBenchmark-master\Data\DataSet1\Saliency/'
      
    IMG_DIR =  '../DataSet1/Imgs/'
    COARSE_DIR ='../DataSet1/Saliency/'
  
    IMG_NAME_LIST = filter(lambda x:x[-3:]=='jpg',listdir(IMG_DIR))
    coarseMethods = ['QCUT','DRFI']
    imgInd = 1
    n_segments,compactness = 200,10
    
    imgName = IMG_NAME_LIST[imgInd]
    img,imgGt = readImg(imgName)
    coarseDic = getCoarseDic(imgName,coarseMethods)
    #show(coarsesDic)   
    sumCoarseImg = getSumCoarseImg(coarseDic)
    coarseImgs=coarseDic.values()
    labelMap = getSlic(img,n_segments,compactness)
    
    rgb = img
    
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
    
    vectorsImg = valueToLabelMap(labelMap,normalizing(vectors.sum(1)))
    show([rgb,refinedImg])
    show([rgb,vectorsImg])
    show(vectorsImg-refinedImg)
    loga(vectorsImg-refinedImg)
    
def grabCutTest():
    coarseMethods = ['MY4','QCUT','DRFI']
    imgInd = 1
    n_segments,compactness = 200,10
    imgName = IMG_NAME_LIST[imgInd]
    img,imgGt = readImg(imgName)
    coarseDic = getCoarseDic(imgName,coarseMethods)
    refinedImg = coarseDic['DRFI']
    
    mask = grabCut(img,refinedImg)
    imgCut = img*mask[:,:,np.newaxis]
    
    show([img,imgCut])
    show([refinedImg,mask])
    
def findBestPictureWithMethod():
    from analysis import getPrCurve
    from tools import saveData
    d = getPrCurve(allMethods,1000)
    #d[3]['img']['0.jpg']['MEAN'].keys()
    d = d[3]['img']
    aucs = {name:{method:d[name][method]['auc'] for method in d[name]} for name in d}
    # auc = d['13.jpg']['RC']['auc']
    saveData(aucs,'aucs')
    def sortGood(myMethod='ME1',compare='DISC2'):
        l = [(name,aucs[name][myMethod]-aucs[name][compare]) for name in aucs]
        l.sort(key=lambda x:x[1])
        return l
#    [showpr(i) for i,j in l]
    def sortAucByMethod(method='ME1'):
        l = [(name,aucs[name][method]) for name in aucs]
        l.sort(key=lambda x:x[1])
        return l
    return sortGood()

def aucBug():
#    from saliency import *
#    from algorithm import *
#    from analysis import *
    ll=[ ('184.jpg', 0.83881506713016329),
 ('138.jpg', 0.84727243912368833),
 ('122.jpg', 0.88417034903928537),
 ('153.jpg', 0.92387361400695434)]
    name = ll[0][0]
    imgGt = io.imread(IMG_DIR+name[:-3]+'png').astype(np.bool)#.ravel()
    methods= ['DISC2', 'DRFI','GMR','MEAN']
    method = 'DISC2'
    coarseDic = getCoarseDic(name,methods,COARSE_DIR)
    print coarseDic.keys()
#    show(coarseDic.values())
    resoult = coarseDic[method]
    show([resoult])
    show(imgGt)
    ##imgGt.ravel()
    #resoult.ravel()
    p,r,_ = precisionRecallCurve(imgGt, resoult)  
    #l = zip(resoult,imgGt)
    print method,'auc',metrics.roc_auc_score(imgGt.ravel(), resoult.ravel())
    plt.plot(p,r)
    plt.show()

def getEdgeImg(img,labelMap=None,width=0.0):
    '''
    width(float):how width of edge
    return a list of label: edge of labelMap
    '''
    labelMap = getSlic(img,300) if labelMap is None else labelMap
    width = int(min(*labelMap.shape)*width)
    u,d,l,r = (labelMap[0:width+1].ravel(),labelMap[-1-width:].ravel(),
                labelMap[:,0:width+1].ravel(),labelMap[:,-1-width:].ravel())
    edge=np.unique(np.c_[[u],[d],[l],[r]])
    new = np.zeros(img.shape)
    for i in edge:
        new[labelMap==i] = 1
    
    show(new)
    return new
    
if __name__ == "__main__":
    getEdgeImg(img)
    pass
    

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    

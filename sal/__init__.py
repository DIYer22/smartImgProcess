# -*- coding: utf-8 -*-
from __future__ import unicode_literals
# __all__ = ['algorithm.py',
#  'analysis.py',
#  'classifyByCompactness.py',
#  'getColorName.py',
#  'picTool.py',
#  'saliency.py',
#  'test.py',
#  'tools.py',
#  'try.py',]

from saliency import *
from algorithm import *
from tools import *


if __name__ == '__main__':
    1

def getSaliency(img):
    img = normalizing(img)
    
    rgb = img
    lab = sk.color.rgb2lab(img)              
    
    # In[6]:
    
    segmentList=[150,250,450,750]
    labelMaps = [getSlic(rgb,n_segments,compactness=50)
                for n_segments in segmentList ]
    labelMap = labelMaps[0]
    slicImg =lambda img, labelMap:(mark_boundaries(valueToLabelMap(labelMap,getColorVector(img,labelMap)),labelMap))
    
    # In[10]:
    integratImgsBy3way = lambda x:normalizing( sum(x))
    #elm
    coarseImgs =  []
    refinImgs = []
    for labelMap in labelMaps:
        degreeVectors, Ws = getVectors(lab, labelMap)
        vectors = getWeightSum(labelMap, degreeVectors, Ws)
        compactnesses= l = [valueToLabelMap(labelMap,v) for v in vectors.T]
    #    print ('Lab || L\n    a || b')
    #    show(l[:2])
    #    show(l[2:4])
    #    print ('LBP of Lab || L\n')
    #    print ('               a || b')
    #    show(l[4:6])
    #    show(l[6:])
        
        refinedImgs = np.array(compactnesses)

        
        coarseImg = integratImgsBy3way(refinedImgs)
    #    show(coarseImg)
        
            
        tmp = np.zeros(rgb.shape)
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
                tmp[labelMap==label] = [0,1,0]
            elif mean <= tl:
                coarseTrain += [(0, 1)]
                tmp[labelMap==label] = [0,0,1]
            else:
                vectorsTrainTag[label]=False
                
        coarseTrains, vectorsTrainTag = np.array(coarseTrain), np.array(vectorsTrainTag)
        vectorsTrains = vectors[vectorsTrainTag]
    #    print (coarseTrains.shape,vectorsTrains.shape)
        
    #    show([rgb,coarseImg,tmp])
        elm = getElm(np.array(vectorsTrains), np.array(coarseTrains))
        refined = elm.predict(vectors)[:,0]
        refinedImg = valueToLabelMap(labelMap,normalizing(refined))
        refinImgs += [refinedImg]
        coarseImgs += [coarseImg]
    
    # In[14]:
    refinedImg = integratImgsBy3way(refinImgs)
    coarseImg = integratImgsBy3way(coarseImgs)
    output = integratImgsBy3way([refinedImg,coarseImg])
    return (output*255.9999).astype(np.uint8)
#    show(refinImgs)
#    show(coarseImgs)
#    show([refinedImg,coarseImg,output])

# In[ ]:

#getSaliency(img)


    
    
    

 
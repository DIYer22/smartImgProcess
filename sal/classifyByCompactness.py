# -*- coding: utf-8 -*-

from algorithm import *

def saveCompactness(imgName,
                    segmentList=[200,250,750]
                    ):
    '''
    将Compactness 保存在 LABEL_DATA_DIR 下的 .compactness文件中
    '''
    compactness=20 
    if isinstance(imgName, int):
        imgName = IMG_NAME_LIST[imgName]
    print 'img index:%d/%d'%(IMG_NAME_LIST.index(imgName),len(IMG_NAME_LIST))
    img,imgGt = readImg(imgName)
    rgb = img
    img = sk.color.rgb2lab(img)
    compactnessImgs = []
    for n_segments in segmentList:
        labelMap = getSlic(rgb,n_segments,compactness)  
        # 获得4+4维  distance
        degreeVectors, Ws = getVectors(img, labelMap)
        weightSumVectors = getWeightSum(labelMap, degreeVectors, Ws)
        compactness = valueToLabelMap(labelMap,weightSumVectors)
        compactnessImgs += [compactness]
#        show(compactness[...,:].sum(2))
    compactness = (sum(compactnessImgs))/3.
    show([mark_boundaries(rgb,labelMap),mark_boundaries(compactness[...,:].sum(2),imgGt==0)])
    saveData(compactness,LABEL_DATA_DIR+'%s.compactness'%imgName)

def saveAllCompactness(num=None):
    '''
    将所有图片的 Compactness 保存在 LABEL_DATA_DIR 下的 .compactness文件中
    (由于SLIC+campactness运算特别慢 所以我把它保存在了硬盘上，只需运行一次)
    '''
    if not num:
        num = len(IMG_NAME_LIST)
    for name in IMG_NAME_LIST[:num]:
        saveCompactness(name)

def getCompactnessVector(imgName):
    '''
    Divide the img's compactness into w*h equal regions
    return a w*h*8 dim vector
    '''
    w, h = 8, 8 # num of blocks  in row and col
    if isinstance(imgName, int):
        imgName = IMG_NAME_LIST[imgName]
    print 'img index:%d/%d'%(IMG_NAME_LIST.index(imgName),len(IMG_NAME_LIST))
    compactness = loadData(LABEL_DATA_DIR+'%s.compactness'%imgName)
    m, n = compactness.shape[:2] 
    maskM = np.array([[i]*n for i in range(m)])
    maskN = np.array([range(n) for i in range(m)])
    vector = []
    for i in range(h):
        for j in range(w):
            ind = np.where((i*m/h<=maskM)&(maskM<(i+1)*m/h)&(j*n/w<=maskN)&(maskN<(j+1)*n/w),True,False)
#            show(ind)            
            vector += list(np.mean(compactness[ind],0))
    vector = np.array(vector)
    return vector

def getCompactnessTrain(dataDic,aucMethod='ME1'):
    '''
    dataDic:getPrCurve 生成的含有auc详细信息的dic
    return train:list of [name,(1,0)] or  [name,(0,1)] to train
    '''
    data = dataDic['img']
    aucs = np.array([data[name][aucMethod]['auc'] for name in data])
    maxx,minn = aucs.max(),aucs.min()
    d = maxx - minn
    omega = np.mean(aucs) - minn
    alpha = 0.8
    '''有改动 把tl的min换成max'''
    th = min([0.9*d+minn,(1+alpha)*omega+minn])
    tl = max([0.6*d+minn,(1-alpha)*omega+minn])
#    loga(aucs)
#    print th,tl,maxx,minn
    train = []
    for name in data:
        auc = data[name][aucMethod]['auc']
        if auc > th:
            train += [[name,(1,0)]]
        if auc < tl:
            train += [[name,(0,1)]]
    return train
    
def buildAucElm(dataDic,save=False,aucMethod='ME1'):
    '''
    dataDic:getPrCurve 生成的含有auc详细信息的dic
    save:若为字符串 则将elm保存在当前文件夹下
    aucMethod: 用于训练的auc的来源
    '''
    train =getCompactnessTrain(dataDic,aucMethod=aucMethod)
    vectors = []
    ys = []
    for name,y in train:
        print name
        vector = getCompactnessVector(name)
        vectors += [vector]
        ys += [y]
        
    vectors,ys = np.array(vectors).astype(np.float64),np.array(ys)
#    print vectors.shape,np.mean(vectors,0),ys.sum(0)
    aucElm = getElm(vectors,ys)
    if save:
        if not (isinstance(save,unicode) or isinstance(save,str)):
            save = 'aucElm.elm'
        aucElm.save(save)
    return aucElm


def getAucElm(num=None,save=False,aucMethod='ME1'):
    '''
    运行前 请确保在LABEL_DATA_DIR 里面 以经用saveAllCompactness()生成了 .compactness文件
    num 训练样本数
    save:若为字符串 则将elm保存在当前文件夹下
    aucMethod: 用于训练的auc的来源
    '''
    from analysis import getPrCurve
    _,_,_,dataDic = getPrCurve(methods=[aucMethod],num=num)
    aucElm = buildAucElm(dataDic,aucMethod=aucMethod)
    return aucElm

def predictImgAuc(imgName,aucElm):
    vector = getCompactnessVector(imgName)
    rel = aucElm.predict(np.array([vector]))[:,0]
    return rel[0]
    

if __name__ == '__main__':

    aucMethod = 'ME1'
    num = 100
    #saveAllCompactness(num+2)
    aucElm = getAucElm(num,False,aucMethod)
    print predictImgAuc(num+1,aucElm)









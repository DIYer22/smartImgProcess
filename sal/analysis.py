# -*- coding: utf-8 -*-
from os import listdir
import os

import skimage.io as io
import matplotlib.pyplot as plt
import numpy as np

from sklearn.metrics import precision_recall_curve,average_precision_score
from sklearn import metrics

from tools import loadData,saveData
from tools import performance,show,normalizing,random
from algorithm import getCoarseDic

import saliency as sal
IMG_DIR=sal.IMG_DIR
COARSE_DIR=sal.COARSE_DIR
IMG_NAME_LIST=sal.IMG_NAME_LIST 
LABEL_DATA_DIR=sal.LABEL_DATA_DIR

@performance
def precisionRecallCurve(imgGt,refinedImg,maxPoints=256):
    '''
    maxPoints: 返回多少个点
    手写求PR线 
    '''
    _refinedImg = normalizing(refinedImg).ravel()
    _imgGt = imgGt.ravel()
    l = zip(_refinedImg,_imgGt)
    l.sort(key=lambda x:x[0])
    n = len(_refinedImg)
    step = 1./maxPoints
    P = imgGt.sum()
    _l = [0]*n
    summ = 0
    for i in range(n):
        summ += l[-i-1][1]
        _l[-i-1] = summ
    index = 0
    pl,rl = [], []
    for i in range(maxPoints):
        thre = 0.+i*step
        for j in range(index,n):
            if l[j][0] >=thre:
                break
        index = j
        TP = float(_l[j])
        _P = n - index
        p ,r = TP/_P, TP/P
        pl+=[p]
        rl+=[r]
    pl = np.array(pl)
    rl = np.array(rl)
    return pl,rl,None

@performance
def plotPrCurve(precisions, recalls, labels=None, avgPreScore=-1, save=None):
    '''
    recalls, precision: list of PR curves
    labels: each line's name
    save: if is string of path, save plot img and data to path
    '''
    n = len(recalls) if  isinstance(recalls, list) else 1
    if labels == None:
        labels = range(n) if n!= 1 else 'line'
    if n == 1 and not isinstance(recalls, list):
        precisions = [precisions]
        recalls = [recalls]
        labels = [labels]
    
    plt.clf()
    color = ['r','b', 'k', 'g', 'c', 'y', 'm']
    lineStyle = '-','--',':','-.'
    colors = []
    for i in lineStyle:
        colors += map(lambda x:x+i ,color)

    for recall,precision,label,i in zip(recalls,precisions,labels,range(len(recalls))):
        plt.plot(recall, precision, colors[i%len(colors)], 
                 lw=1, label=label)
    plt.xlabel('Recall')
    plt.ylabel('Precision')

#    plt.axis('equal')
    plt.axis([0., 1., 0., 1.])
    plt.title('Avg of {0:d} line\'s AUC={1:0.2f}'.format(n,avgPreScore))
    plt.legend(bbox_to_anchor=(0., 0, .8, 1.1),loc=3,ncol=3, mode="expand",borderaxespad=0.)
    plt.grid()
    if save != None:
        if not isinstance(save,str) and not isinstance(save,unicode):
            save = 'lastResoult'
        name = save
        data = recalls, precisions, labels, avgPreScore
        
        plt.savefig('./figs/'+name+'.png', dpi=300)
        dirr = './figs/'+name+'/'
        if not os.path.isdir(dirr) :
            os.mkdir(dirr)
        
        plt.savefig(dirr+name+'.svg')
        saveData(data,dirr+name+'.pickle')
    plt.show()


@performance
def getPrCurve(methods= ['DRFI','GMR','MEAN'],
                num = None,
                ):
    '''
    返回数据库前num张图片的平均PR曲线
    dataDic 结构
    img->[imgname]->[method]->auc,pr
    method->[method]->auc,pr
    '''
    
    num = num if num else len(IMG_NAME_LIST)
    dataDic = {'img':{}}
    for name in IMG_NAME_LIST[:num]:
        print '%d/%d'%(IMG_NAME_LIST.index(name),num),name
        imgGt = io.imread(IMG_DIR+name[:-3]+'png').astype(np.bool).ravel()
        coarseDic = getCoarseDic(name,methods,COARSE_DIR)
        dataDic['img'][name] = {}
        
        for k,v in coarseDic.items():
            dataDic['img'][name][k] = {}
            _dic = dataDic['img'][name][k]
            coarseImg = v.ravel()
            p, r, _ = precisionRecallCurve(imgGt, coarseImg)
            _dic['pr'] = (p, r)
            _dic['auc'] = metrics.roc_auc_score(imgGt.ravel(), coarseImg)
    
    dataDic['method'] = dict(zip(methods, [{} for i in range(len(methods))]))
    precisions, recalls = [],[]
    for method in methods:
        _dic = dataDic['method'][method]
        p, r = sum([v[method]['pr'][0] for k,v in dataDic['img'].items()])/num,\
                sum([v[method]['pr'][1] for k,v in dataDic['img'].items()])/num
        _dic['pr'] = (p, r)
        _dic['auc'] = metrics.auc(r, p)    
        precisions += [p]
        recalls += [r]
    return precisions,recalls,methods,dataDic

@performance
def getPrCurveMergeToOne(
                methods= ['MY','DRFI','GMR','MEAN'],
                num = None,
                ):
    '''
    过时的方法 全部读入内存
    '''
    imgGt = reduce(lambda x,y:np.r_[x,y],
                   map(lambda name:io.imread(IMG_DIR+name[:-3]+'png').astype(np.bool).ravel(),
                       IMG_NAME_LIST[:num])
                   )
    print 'imgGt OK!'
    resoultDic = dict(zip(methods, [None]*len(methods)))
    print resoultDic.keys()
    for name in IMG_NAME_LIST[:num]:
        dic = getCoarseDic(name,methods,COARSE_DIR)
        for k,v in dic.items():
            resoultDic[k] = np.r_[resoultDic[k], v.ravel()] if resoultDic[k]!=None else  v.ravel()
        print '%d/%d'%(IMG_NAME_LIST.index(name),num),name
    
    precisions, recalls = [],[]
    for method in methods:
#        p, r ,_ = precisionRecallCurve(imgGt,resoultDic[method])
        p, r ,_ = precision_recall_curve(imgGt,resoultDic[method])
        precisions += [p]
        recalls += [r]
        print method
    return precisions,recalls,methods
    


def plotMethods(methods= ['DRFI','GMR','MEAN'],
                save = None,
                num = None,
                ):
    '''
    分析整个数据库的methods 方法
    返还data
    '''
    p,r,methods,dataDic = getPrCurve(methods,num)
    data = dataDic['method']
    aucs = 0
    l = []
    for method in data:
        aucs += data[method]['auc']
        l += [(method,data[method]['auc'])]
    l.sort(key=lambda x:x[1],reverse=1)
    strr = "\n%s\n%s"
    for name,auc in l:
        strr%=(('%-6s|'%name)+' %s',str(auc)[:5]+' | %s') 
    print strr%('','')
    aucs/= len(data)
    plotPrCurve(p, r, methods, avgPreScore=aucs, save=save)
    return dataDic

def plotImgPr(imgGt,
              coarseDic):
    '''
    对一张图片 画出coarseDic里的PR和AUC
    '''
    methods = []
    true = imgGt.ravel()
    ps, rs = [],[]
    l, aucs = [], 0
    for k,v in coarseDic.items():
        v = v.ravel()
        p, r,_ = precisionRecallCurve(true, v)
        ps, rs = ps+[p], rs+[r]
        auc = metrics.auc(r, p)
        aucs += auc
        l += [(k,auc)]
        methods += [k]
    l.sort(key=lambda x:x[1],reverse=1)
    strr = "\n%s\n%s"
    for name,auc in l:
        strr%=(('%-6s|'%name)+' %s',str(auc)[:5]+' | %s') 
    print strr%('','')
    aucs/= len(l)
    plotPrCurve(ps, rs, methods, avgPreScore=aucs)

    
    
def getAucAndPr(imgGt,refindImg,method):
    '''
    get AUC and PR 废弃
    '''
    y_true,probas_pred = imgGt.ravel(),refindImg.ravel()
    auc = average_precision_score(y_true,probas_pred)
    p,r,_ = precisionRecallCurve(y_true,probas_pred)
    return auc,p,r


def saveImgData(imgName,methods):
    '''
    save data to LABEL_DATA_DIR
    return default method`s AUC
    '''
    if isinstance(imgName, int):
        imgName = IMG_NAME_LIST[imgName]
    imgGt = io.imread(IMG_DIR+imgName[:-3]+'png')/255.
    coarseDic = getCoarseDic(imgName,methods,COARSE_DIR)
    data = {}
    data['name'] = imgName
    
    data['method'] = {}
    for method,refindImg in coarseDic.items():
        data['method'][method] = getAucAndPr(imgGt,refindImg,method)
    
    dataName = LABEL_DATA_DIR+imgName[:-3]+'data'
    AUC_METHOD = "DRFI"
    data['aucMethod'] = AUC_METHOD
    
    data['auc'] = data['method'][AUC_METHOD][0]
    saveData(data,dataName)
    return data['auc']

def saveImgsData(methods):
    '''
    save all imgs data
    '''
    aucs=[]
    for imgName in IMG_NAME_LIST[:]:
        aucs += [(imgName,saveImgData(imgName,methods))]
        print aucs[-1]
    aucs.sort(None,key=lambda x:x[1])
    return aucs
    

if __name__ == '__main__':
    from copy import deepcopy
    from algorithm import showpr
    


    showMethods = ["MY3","MY4","ME1", "DRFI", "QCUT","DISC2"]
    num = 4
    data = plotMethods(num=num,
                       save='last'
                       )
    dic = deepcopy(data['img'])
    for name in dic:
        dic[name] = {'method':dic[name]}
        _dic = dic[name]
        _dic['auc'] =  np.mean([v['auc'] for method,v in _dic['method'].items()])
        _dic['aucs'] = [(method, v['auc']) for method,v in _dic['method'].items()]
#        _dic['var'] = np.var([v['auc'] for method,v in _dic['method'].items()])
        _dic['name'] = name
        _dic['max'] = max(_dic['aucs'],key=lambda x:x[1])
        
        _dic['sortKey'] = _dic['max'][1] - _dic['auc']
        
    listt = dic.values()
    listt.sort(key=lambda x:x['sortKey'])
    for x in listt[-20:]:
        showpr(x['name'],showMethods) 
        print x['max'],x['auc'] 
        print x['sortKey'] 
        
def findBestPictureWithMethod():
    d = getPrCurve(allMethods,10)
    #d[3]['img']['0.jpg']['MEAN'].keys()
    d = d[3]['img']
    aucs = {name:{method:d[name][method]['auc'] for method in d[name]} for name in d}
    saveData(aucs,'aucs')
    def sortAucByMethod(method='ME2'):
        l = [(name,aucs[name][method]) for name in aucs]
        l.sort(key=lambda x:x[1])
        return l




# -*- coding: utf-8 -*-
"""
Created on Thu Jul 28 00:36:24 2016

@author: yl
"""
from imgProcessBackEnd import *

def colorTableTest():
    #inds = [0]
    hisRaw = np.zeros((256,256,3),np.uint8)+90
    addColorToHis(hisRaw, inds)
    
    #plt.plot(np.arange(0,1,0.01),np.arange(0,1,0.01)**a)
    contrast = lambda x, a: 128.5 + 128.5*(abs(x-128.5)/128.5)**(1./a)*(-1 if x<128.5 else 1)
    # 0.3 <= a <= 3
    
    f = lambda x:contrast(x,2)
#    f = lambda x: x+50
    colorTables = getColorTables(f,inds)
    print (colorTables)
    applyColorTable(colorTables,img,inds)
    his = changeLineToHis(img, inds, colorTables)
    show([raw,img])
    show([hisRaw,his])



def getDifferentAFrams(): 
    aas = [a for a in (0.2+2.5*arange(0,1,0.9)**0.5)]
    
    frams = []
    
    for a in aas:
        f = funDic['contrast']
        colorTables = getColorTables(f, inds)
        applyColorTable(colorTables,img,inds)
        frams += [img.copy()]
    img = raw.copy()
    
#getDifferentAFrams()
#show(img)


def draw3dSurfaceTest():
    X = np.arange(-5, 5, 0.1)
    Y = np.arange(-5, 5, 0.1)
    X, Y = np.meshgrid(X, Y)
    
    sig = 1
    Z = np.e**(-(X**2+Y**2)/2/sig)/(2*np.pi*sig**2)
    draw3dSurface(X,Y,Z)
#draw3dSurfaceTest()

#%%
def changeTest():
    raw = {"change":{"亮度":25.6,"对比度":1.92,"饱和度":-0.168,"高亮":0,"阴影":-12}}
    data = raw['change']
#    data["饱和度"]=0
    f = funDic[u'亮度']
    return change(data)
#crun('changeTest()')

#%%
d = {
        u'高亮':10,
        u'\u5bf9\u6bd4\u5ea6': 1, u'\u4eae\u5ea6': 0, u'f': 0}

    
#funDic[u'高亮']= Fun(f ,
#u'高亮',128,-128,0)
#changeColorTable({k:d[k] for k in d if k in funDic})
#
#
#show(vi.v)
#show(g['resHis'])


def heightLightShadowTest():
    xs = range(256)
    for a in np.linspace(-50,50,8):
        plt.plot(xs,[heightLight(i,a) for i in xs])
        plt.plot(xs,[shadow(i,a) for i in xs],'--')
    plt.grid()
    plt.margins(0.5,0.1)
    plt.show()
    #run('[heightLight(i,20) for i in range(256)]')

def gaussCoreTest():
    from math import e,pi    
    r = 20
    sig = 1
    axisLen = 2
#    axisLen = 0.5*sig
    thred = 0.05
    maxR = 50
    X = np.linspace(-axisLen, axisLen, maxR*2+1)
    Y = np.linspace(-axisLen, axisLen, maxR*2+1)
    X, Y = np.meshgrid(X, Y)
    
    Z = np.e**(-(X**2+Y**2)/2/sig)/(2*np.pi*sig**2)
    draw3dSurface(X,Y,Z)
    core = Z[maxR-r:maxR+r+1,maxR-r:maxR+r+1]
    core = core/core.sum()
    show([Z,core])
    print (core.shape)
#%%   
def gaussFunTest():
    img = da.astronaut()
#    show(img)
    #Gaussian Blur
    R = 5
    core = gaussCore(R)

    g['fliterImg'] = getFilterImg(img)
    fi = g['fliterImg']
    maxr = MAX_FILTER_R
    m,n = img.shape[:2]
    u,r,d,l = maxr,n+maxr,m+maxr,maxr
    new = np.zeros((m,n,3),np.uint8)
    for i,y in enumerate(range(maxr,maxr+m)):
        for j,x in enumerate(range(maxr,maxr+n)):
            v = np.round((fi[y-R:y+R+1,x-R:x+R+1]*core[:,:,None]).sum(0).sum(0))
            new[i][j] = v
    show(new)
#%%
def bilateralCoreTest(r=None):
    r = r if r else 10
    sd = 100
    sr = sd*0.3
    block = random((2*r+1,2*r+1,3),255)
    block[r-1:,r-1:] = 10
    def f(_,y,x):
        dd = -((y-r)**2+(x-r)**2)/2./sd
        rr = -np.linalg.norm(block[y,x]-block[r,r])/2./sr
        return np.power(np.e,dd+rr)
    core = mapp(f,np.zeros(block.shape[:2]),True)
    core = core/core.sum()
    show([core])
#    loga(core)
    polt3dSurface(core) 

def reduceOneLineTest():
    def getTestMag():
        s = '''2 1 2 1 3 4
3 2 1 2 2 3
2 1 1 2 3 1'''
        return np.array(map(int,s.replace('\n',' ').split(' '))).reshape((3,6))
    mag = getTestMag()
    m, n = mag.shape
    img = mag.repeat(3).reshape(list(mag.shape)+[3])
    #img = normalizing(img)
    expand = lambda row: np.append(np.append(row[:1],row),row[-1:])
    
    accu = [mag[0]]
    for ind in range(1,m):
        tmp = np.zeros((n,3),mage.dtype)
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
    for r in range(m-1)[::-1]:
        d = np.argmin(accui[r][ind:ind+3])-1
        ind = ind+d
        newimg = [pop(ind,img[r])]+newimg
        pass
    newimg = np.array(newimg)
    print mag
    print newimg[...,0]

if __name__ == '__main__':
    '''能量最小算法'''
    bilateralCoreTest()



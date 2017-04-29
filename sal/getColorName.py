from mlab.releases import latest_release as mlab
from matlab import matlabroot
import skimage.io as io
import scipy.io as sio
import matplotlib.pyplot as plt  
print matlabroot()
from matlab import colorname


def show(l):
    if not isinstance(l,list):
        l = [l]
    n = len(l)
    fig, axes = plt.subplots(ncols=n)
    count = 0
    axes = [axes] if n==1 else axes 
    for img in l:
        axes[count].imshow(img,cmap='gray')
        count += 1
        
def getColorName(img):
    io.imsave('test.jpg',img)    
    colorname('test.jpg','result.jpg')
    img1 = io.imread('result.jpg')
    return img1 
    
img = io.imread(r'E:\3-experiment\SalBenchmark-master\Data\DataSet1\Imgs/0013.jpg')
img1=getColorName(img)
show(img)
show(img1)

# -*- coding: utf-8 -*-
"""
Created on Thu Jul 28 00:36:24 2016

@author: yl
"""
from __future__ import unicode_literals
vi = None
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


import sys
#py3
pyv = sys.version_info.major
py3 = pyv == 3
if py3:
    __listRange__ = range
    range = lambda *x:list(__listRange__(*x))
    __rawOpen__ = open
    open = lambda *l:__rawOpen__(l[0],'r',-1,'utf8') if len(l) == 1 else __rawOpen__(l[0],l[1],-1,'utf8')

MAX_FILTER_R = 50
g = {}
if __name__ == '__main__':
  
  
  
    pass















# -*- coding: utf-8 -*-
from __future__ import unicode_literals
dirr = './imgs/hku/'

paths = [dirr+i for i in  os.listdir(dirr) if '.jpg' in i]
print paths
#io.imsave(path[:path.rindex('.')]+'_SS.png',mask)

from imgProcess import *
s='''背景
飞机
自行车
鸟
船
瓶子
公交车
汽车
猫
椅子
牛
桌子
狗
马
摩托车
人
飞机
羊
沙发
火车
屏幕'''
classList = ["背景", "飞机", "自行车", "鸟", "船", "瓶子", "公交车", "汽车", "猫", "椅子", "牛", "桌子", "狗", "马", "摩托车", "人", "飞机", "羊", "沙发", "火车", "屏幕"]
classDic = {i:n for i,n in enumerate(s.split())}
for path in paths:
    print path
    begin(path)

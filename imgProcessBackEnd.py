# -*- coding: utf-8 -*-
"""
Created on Thu Jul 28 00:36:24 2016

@author: yl
"""
from __future__ import unicode_literals

import webbrowser

import os
from flask import Flask, abort,render_template,redirect,url_for,g
from flask_socketio import SocketIO, emit
from flask import request,jsonify,session

from time import sleep
from json import loads,dumps

from imgProcess import base64Img
from imgProcess import *


app = Flask(__name__)
app.debug = True
app.debug = False
app.secret_key = 'hashhash'

@app.errorhandler(404)
def page_not_found(e):
    return '<h1>Url wrong! 404 not found</h1>', 404

@app.errorhandler(403)
def page_not_found2(e):
    return '<h1>Already has two plyer,please wite for a while Ture</h1>', 403

@app.route('/test', methods=['GET', 'POST'])
def test():
    p = request.get_json(silent=False)
    print (p,[p])
    return jsonify((1,2,"222"))

@app.route('/js/<filee>')
def get_static(filee):
    f = open('html/js/' + filee)
    html = f.read()
    f.close()
    return html



@app.route('/')
def get_index_page():
    img = g['rraw']
    g['raw'][:,:,:] = img
    g['img'][:,:,:] = img
    begin(g['img'])
    with open('html/imgProcess.html') as f:
        strr = f.read()
    return strr
  
@app.route('/repetShow')
def repetShow():
    p = request.args.get('ind', 5, type=int)
    print (request.args)
#    sleep(0.1)
    base = base64Img(frams[p%len(frams)])
    return jsonify(base)

def getRaw(v=None):
    base = base64Img(img)
    return base

def getBegin(v=None):
    imgg = cv2.resize(img,SHOW_SHAPE)
    return base64Img(imgg)
def zoomUp(v=None):
    img, raw = vi.zoomUp()
    return [base64Img(img),base64Img(raw)]
def zoomDown(v=None):
    img, raw = vi.zoomDown()
    return [base64Img(img),base64Img(raw)]

def mouse(data):
    if 'up' in data:
        pass
    elif 'down' in data:
        g['mouse'] = np.array(data['down'])
    else:
        xy = np.array(data)
        img, raw = vi.move(xy - g['mouse'])
        g['mouse'] = xy
        return [base64Img(img),base64Img(raw)]
def getRawHis(v=None):
    return base64Img(g['rawHis'])
#    [,g['resHis']]

def getControlBars(v=None):
    funs = [funDic[k] for k in orderList]
    return [{k:v for k,v in fun.dic.items() if k!='f'} for fun in funs]


def change(d):
    g['fg'][:,:,:] = g['raw']
    img = g['fg']
    changeImg(img,{k:d[k] for k in d if k in funDic and k in imgFunList})
    colorTables = changeColorTable(img,{k:d[k] for k in d if k in funDic and k in channelFunList})
    
    bg = g['bg']
    sal = g['sal'][...,None]
    g['img'][:,:,:] = ((np.zeros(sal.shape)+1-sal)*bg+sal*img).astype(np.uint8)
    g['resHis'] = getHisWithLine(g['img'], g['inds'], colorTables)
    return [base64Img(vi.getView(True)),base64Img(g['resHis'])]

def changeBg(d):
    g['bg'][:,:,:] = g['raw']
    img = g['bg']
    changeImg(img,{k:d[k] for k in d if k in funDic and k in imgFunList})
    colorTables = changeColorTable(img,{k:d[k] for k in d if k in funDic and k in channelFunList})
    
    fg = g['fg']
    sal = g['sal'][...,None]
    g['img'][:,:,:] = ((np.zeros(sal.shape)+1-sal)*img+sal*fg).astype(np.uint8)
    g['resHis'] = getHisWithLine(g['img'], g['inds'], colorTables)
    return [base64Img(vi.getView(True)),base64Img(g['resHis'])]
    
def changeChannels(args):
    
    bg = g['raw'].copy()
    d = args[1]
    changeImg(bg,{k:d[k] for k in d if k in funDic and k in imgFunList})
    changeColorTable(bg,{k:d[k] for k in d if k in funDic and k in channelFunList})
    
    d = args[0]
    img = g['raw'].copy()
    changeImg(img,{k:d[k] for k in d if k in funDic and k in imgFunList})
    changeColorTable(img,{k:d[k] for k in d if k in funDic and k in channelFunList})
    
    sal = g['sal'][...,None]
    g['img'][:,:,:] = ((np.zeros(sal.shape)+1-sal)*bg+sal*img).astype(np.uint8)
    return [base64Img(vi.getView(True)),base64Img(g['resHis'])]
def setImgToRaw(d=None):
    img = g['img'][:,:,:]
    rraw = g['rraw'][:,:,:]
    g.clear()
    begin(img)
    g['rraw'] = rraw
            
def delSc(d=None):
    deln,isRow,isSal = d
    if not deln:
        return [base64Img(vi.getView(True)),base64Img(g['resHis'])]
    
    sal = g['sal'] if isSal else None
    img = seamCarveBlack(g['rraw'],deln,row=isRow,mask=sal)
    g['raw'][:,:,:] = img
    g['img'][:,:,:] = img
    begin(g['img'])
    return [base64Img(vi.getView(True)),base64Img(g['resHis'])]

def getKinds(d=None):
    if 'masks' not in g:
        g['masks'] = getMasksByPath(path)
    return [kindToName[i] for i in g['kinds']]
#%%
def delKinds(d=None):
    if 'masks' not in g:
        g['masks'] = getMasksByPath(path)
    print(d)
    dels = [nameToKind[name] for name in d]
    img = smartSc(dels,g['masks'],g['rraw'])
    g['raw'][:,:,:] = img
    g['img'][:,:,:] = img
    begin(g['img'])
    return [base64Img(vi.getView(True)),base64Img(g['resHis'])]
def getHed(d=None):
    if 'masks' not in g:
        g['masks'] = getMasksByPath(path)
    hed = g['masks']
    img = hed.repeat(3).reshape(list(hed.shape)+[3])
    g['raw'][:,:,:] = img
    g['img'][:,:,:] = img
    begin(g['img'])
    return [base64Img(vi.getView(True)),base64Img(g['resHis'])]
def noSal(d=None):
    g['sal'][:,:] = np.ones(g['img'].shape[:2])
    g['img'][:,:,:] = g['fg']
    return [base64Img(vi.getView(True)),base64Img(g['resHis'])]
  
def useSal(d=None):
    g['sal'][:,:] = getSaliencyByPath(g['path'])
    return [base64Img(vi.getView(True)),base64Img(g['resHis'])]
#%%

dealDic={
    'double':lambda x:x*2 ,
    'getRaw':getRaw,
    'getBegin':getBegin,
    'zoomUp':zoomUp,
    'zoomDown':zoomDown,
    'mouse':mouse,

    'getRawHis':getRawHis,
    'getControlBars':getControlBars,
    
    'setImgToRaw':setImgToRaw,
    'change':change,
    'changeBg':changeBg,
    'noSal':noSal,
    'useSal':useSal,
    
    'delSc':delSc,
    'getKinds':getKinds,
    'delKinds':delKinds,
    
    'getHed':getHed,
    
 }
@app.route('/getJson/<json>')
def getJson(json):
    json = loads(json)
#    print (json,json.keys())
    key = list(json.keys())[0]
    if key not in dealDic:
        abort(404)
        return 404
    respone = dealDic[key](json[key])
#    print('\n\n\n',type(respone),str(respone)[:100],'\n\n')
    return jsonify(respone)


if __name__ == "__main__":
    import imgProcessConfig as config
    begin(config.path)
    vi = g['vi']
    zoomDown()
    app.run(host='0.0.0.0', port=80)
    pass

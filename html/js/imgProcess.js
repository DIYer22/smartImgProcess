test={
    controlBars :[
        {
            name:'亮度',
            value:0,
            max:5,
            min:-5,
        },
        {
            name:'亮度2',
            value:0,
            max:5,
            min:-5,
        }
    ]    
}

var view = {
    scroll:(e)=>{
        e.preventDefault()
        var x=e.offsetX
        var y=e.offsetY
        var change = int(e.deltaY)

        if (change<0){
            getJson('zoomUp',null,putResRawImgs,'zoom')
        }
        if (change>=0){
            getJson('zoomDown',null,putResRawImgs,'zoom')
        }
    },
    onMove:(e)=>{
        e.preventDefault()
        var x = e.clientX
        var y = e.clientY
        getJson('mouse',[x,y],putResRawImgs,'onMove')
    },

    onDown:(e)=>{
        e.preventDefault()
        var x = e.clientX
        var y = e.clientY
        getJson('mouse',{'down':[x,y]},null)
        rawImg.onmousemove = view.onMove
        resImg.onmousemove = view.onMove
    },
    onUp:()=>{
        rawImg.onmousemove = null
        resImg.onmousemove = null
        // getJson('mouse',{'up':null},null)
    },
    mouseBegin:()=>{
        rawImg = $('#rawImg')[0]
        resImg = $('#resImg')[0]
        rawImg.onmousewheel=view.scroll
        resImg.onmousewheel=view.scroll

        rawImg.onmousedown = view.onDown
        resImg.onmousedown = view.onDown
        window.onmouseup = view.onUp 
        window.onmouseup = view.onUp 
        onmouseup()   
    }
}
putResRawImgs=(d)=>{ // 将[resImg,rawImg]显示出来
    putImg('rawImg',d[1])
    putImg('resImg',d[0])
}
putImg = (id, base)=>{  //将base64 img 放到#id中
      var img = '<img src="data:image/jpg;base64,'+base+'" class="inline"/> '
      $('#'+id).html(img)  
}



 // ***************


 tryTest = ()=>{$.getJSON('/test',null,(d)=>log(d))}


change = ()=>{
    var para = getAllPara()
    getJson('change',para,putResImgHis,true)
}
var controlBarTemple = $('#controlBarTemple')[0].innerHTML

getAllPara=()=>{
    var r = {}
    for(var bar of window.controlBars){
        r[bar.name] = float($("#"+bar.name+"-amount" ).val())
    }
    return r
}
getControlBars=(controlBars)=>{
    window.controlBars = controlBars
    var tag = true
    for(var bar of controlBars){
        var div = controlBarTemple.format({name:bar.name})

        $(tag?"#colorControlBars1":"#colorControlBars2")[0].append($(div)[0])
        tag = !tag
        // $("#controlBars")[0].append($(div)[0])
        bar.step = (bar.max-bar.min)/50
        var getSlide=(name)=>{
            var slide=(e, v)=>{
                $("#"+name+"-amount").val( v.value );
                window.change()
            }
            return slide
        }
        bar.slide = getSlide(bar.name)
        $("#"+bar.name+"-slider").slider(bar)
        $( "#"+bar.name+"-amount" ).val( $( "#"+bar.name+"-slider" ).slider( "value" ) );
    }
}

changeBg = ()=>{
    var para = getBgPara()
    getJson('changeBg',para,putResImgHis,true)
}

var bgControlBarTemple = $('#bgControlBarTemple')[0].innerHTML


var putResImgHis = (d)=>{
    putImg('resImg',d[0])
    putImg('resHis',d[1])
}


getBgPara=()=>{
    var r = {}
    for(var bar of window.bgControlBars){
        r[bar.name] = float($("#"+bar.name+"-amount-bg" ).val())
    }
    return r
}
getBgControlBars=(bgControlBars)=>{
    window.bgControlBars = bgControlBars
    var tag = true
    for(var bar of bgControlBars){
        var div = bgControlBarTemple.format({name:bar.name})

        $(tag?"#bgColorControlBars1":"#bgColorControlBars2")[0].append($(div)[0])
        tag = !tag
        // $("#bgControlBars")[0].append($(div)[0])
        bar.step = (bar.max-bar.min)/50
        var getSlide=(name)=>{
            var slide=(e, v)=>{
                $("#"+name+"-amount-bg").val( v.value );
                window.changeBg()
            }
            return slide
        }
        bar.slide = getSlide(bar.name)
        $("#"+bar.name+"-slider-bg").slider(bar)
        $( "#"+bar.name+"-amount-bg" ).val( $( "#"+bar.name+"-slider-bg" ).slider( "value" ) );
    }
}


var checkTemple = '<div style="" class="col-sm-offset-1 col-sm-10"><div class="checkbox scChecks"><label><input type="checkbox" id="{id}">{name}</label></div></div>'
setChecks = (cs)=>{
    for(var name of cs){
        var html = checkTemple.format({name:name,id:name})
        $('#scBars')[0].append($(html)[0])
    }
}

delKinds = ()=>{
    _salTag = true
    switchUseSal()
    var l = []
    var checks = $('.scChecks input')
    for (var i = 0; i < checks.length; i++) {
        if(checks[i].checked){
            l.push(checks[i].id)
        }
    };
    log(l)
    if(len(l)>0){
        getJson('delKinds',l,putResImgHis,true)
    }
    
}

var reload = ()=>{
    window.location.reload(true)
}
setImgToRaw = ()=>{
    getJson('setImgToRaw',null,reload)
}

test = ()=>{
    // setImgToRaw()
    reload()
}
var _bgTag = true
switchBg = ()=>{
    window._bgTag = !window._bgTag
    var bg = $('#bgColorControlBars')
    var fg = $('#colorControlBars')
    var bu = $('#switchBg')
    
    
    if(window._bgTag){
        bu.text('切换到显著模式')
        fg.hide()
        bg.show()
    }else{
        bu.text('切换到背景模式')
        bg.hide()
        fg.show()
    }
}

delSc = ()=>{

    _salTag = false
    switchUseSal()
    var l =[
        int($('#deln')[0].value),
        $('#isRow')[0].checked,
        $('#isSal')[0].checked,
    ]
    log(l)
    if(l[0]){
        getJson('delSc',l,putResImgHis,true)
    }

}
var TABS = [$('#bigControlBars'),$('#bigDelScBar'),$('#bigScBars'),$('#bigHedBars')]
function changTag(e){
    // log(e,$(e).attr('data'))
    var ind = int($(e).attr('data'))
    TABS.map((x)=>x.hide()||x.attr('class',' '))
    TABS[ind].show()
    TABS[ind].attr('class','active')

}
getHed = ()=>{
    getJson('getHed',null,putResImgHis,true)
    
}
getBegin = ()=>{
    getJson('zoomDown',null,putResRawImgs)
    getJson('getRawHis',null,(d)=>{putImg('rawHis',d);})
    view.mouseBegin()
    getJson('getControlBars',null,(d)=>getControlBars(d)||change())
    getJson('getControlBars',null,(d)=>getBgControlBars(d)||changeBg())
    // getJson('getKinds',null,setChecks)
    $('#autoClick').click()
    setTimeout(_=>{
        _bgTag = true
        switchBg()
        _salTag = true
        switchUseSal()
    },100)

}
noSal = ()=>{
    _bgTag = true
    switchBg()
    $(".noSalGroup").hide()
    getJson('noSal',null,putResImgHis)
}
useSal = ()=>{
    $(".noSalGroup").show()
    _bgTag = true
    switchBg()
    getJson('useSal',null,putResImgHis)
}
_salTag = true
switchUseSal=()=>{
    window._salTag = !window._salTag
    var bu = $('#switchUseSal')
    
    
    if(window._salTag){
        bu.text('关闭显著性与背景分离')
        useSal()
    }else{
        bu.text('开启显著性与背景分离')
        noSal()
    }
}
getBegin()





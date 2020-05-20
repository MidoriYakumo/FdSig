# -*- coding: utf-8 -*-
"""
Created on Tue May  2 14:15:26 2017

@author: MidoriYakumo
"""

import random
import numpy as np
from PIL import Image,ImageDraw,ImageFont

def normalizedRGB(img):
    if img.dtype == np.uint8:
        return img[:,:,:3] / 255.
    else:
        return img[:,:,:3].astype(np.float64)
        
def mix(a, b, u, keepImag = False):
    if keepImag:
        return (a.real * (1 - u) + b.real * u) + a.imag * 1j
    else:
        return a * (1 - u) + b * u
    
def centralize(img, side = .06, clip = False):
    img = img.real.astype(np.float64)
    thres = img.size * side
    
    l = img.min()
    r = img.max()
    while l + 1 <= r:
        m = (l + r) / 2.
        s = np.sum(img < m)
        if s < thres:
            l = m
        else:
            r = m
    low = l            
            
    l = img.min()
    r = img.max()
    while l + 1 <= r:
        m = (l + r) / 2.
        s = np.sum(img > m)
        if s < thres:
            r = m
        else:
            l = m            
            
    high = max(low + 1, r)          
    img = (img - low) / (high - low)
    
    if clip:
        img = np.clip(img, 0, 1)
    
    return img, low, high

####################################

def shuffleGen(size, secret = None):
    r = np.arange(size)
    if secret: # Not None, ""
        random.seed(secret)
        for i in range(size, 0, -1):
            j = random.randint(0, i)
            r[i-1], r[j] = r[j], r[i-1]
    return r
    
def xmapGen(shape, secret = None):
    xh, xw = shuffleGen(shape[0], secret), shuffleGen(shape[1], secret)
    xh = xh.reshape((-1, 1))
    return xh, xw

def encodeImage(oa, ob, xmap = None, margins = (1, 1), alpha = None):
    na = normalizedRGB(oa)
    nb = normalizedRGB(ob)
    fa = np.fft.fft2(na, None, (0, 1))
    pb = np.zeros((na.shape[0]//2-margins[0]*2, na.shape[1]-margins[1]*2, 3))
    pb[:nb.shape[0], :nb.shape[1]] = nb

    low = 0
    if alpha is None:
        _, low, high = centralize(fa)
        alpha = (high - low)# / 2
        print("encodeImage: alpha = {}".format(alpha))

    if xmap is None:
        xh, xw = xmapGen(pb.shape)
    else:
        xh, xw = xmap[:2]
        
    # add mode
    fa[+margins[0]+xh, +margins[1]+xw] += pb * alpha
    fa[-margins[0]-xh, -margins[1]-xw] += pb * alpha
    # mix mode
#    fa[+margins[0]+xh, +margins[1]+xw] = mix(fa[+margins[0]+xh, +margins[1]+xw], pb, alpha, True)
#    fa[-margins[0]-xh, -margins[1]-xw] = mix(fa[-margins[0]-xh, -margins[1]-xw], pb, alpha, True)
    # multiply mode
#    la = np.abs(fa[+margins[0]+xh, +margins[1]+xw])
#    la[np.where(la<1e-3)] = 1e-3
#    fa[+margins[0]+xh, +margins[1]+xw] *= (la + pb * alpha) / la
#    la = np.abs(fa[-margins[0]-xh, -margins[1]-xw])
#    la[np.where(la<1e-3)] = 1e-3
#    fa[-margins[0]-xh, -margins[1]-xw] *= (la + pb * alpha) / la
    
    xa = np.fft.ifft2(fa, None, (0, 1))
    # use real part
    xa = xa.real
    xa = np.clip(xa, 0, 1)
    # use abs + zoom/shift
#    xa = np.abs(xa)
#    nv = np.average(na)
#    ev = np.average(ea)
#    delta = (nv - ev) / (1 - ev) if 1 > ev else 0
#    xa = xa * (1 - delta) + delta
    return xa, fa
    
def encodeText(oa, text, *args, **kwargs):
    font = ImageFont.truetype("consola.ttf", oa.shape[0] // 7)
    #font = ImageFont.load_default()
    renderSize = font.getsize(text)
    padding = min(renderSize) * 2 // 10
    renderSize = (renderSize[0] + padding * 2, renderSize[1] + padding * 2)
    textImg = Image.new('RGB', renderSize, (0, 0, 0))
    draw = ImageDraw.Draw(textImg)
    draw.text((padding, padding), text, (255, 255, 255), font = font)
    ob = np.asarray(textImg)
    return encodeImage(oa, ob, *args, **kwargs)
    
def decodeImage(xa, xmap = None, margins = (1, 1), oa = None, full = False):
    na = normalizedRGB(xa)
    fa = np.fft.fft2(na, None, (0, 1))
        
    if xmap is None:
        xh = xmapGen((xa.shape[0]//2-margins[0]*2, xa.shape[1]-margins[1]*2))
    else:
        xh, xw = xmap[:2]
        
    if oa is not None:
        noa = normalizedRGB(oa)
        foa = np.fft.fft2(noa, None, (0, 1))
        fa -= foa
        
    if full:
        nb, _, _ = centralize(fa, clip = True)
    else:
        nb, _, _ = centralize(fa[+margins[0]+xh, +margins[1]+xw], clip = True)
    return nb
    
#################################
    
import argparse
from os import path
import matplotlib.pyplot as pyplot   

def imshowEx(img, *args, **kwargs):
    img, _, _ = centralize(img, clip = True)
    
    kwargs["interpolation"] = "nearest"
    if "title" in kwargs:
        pyplot.title(kwargs["title"])
        kwargs.pop("title")
    if len(img.shape) == 1:
        kwargs["cmap"] = "gray"
    pyplot.imshow(img, *args, **kwargs)
    
def imsaveEx(fn, img, *args, **kwargs):
    kwargs["dpi"] = 1
    
    if img.dtype != np.uint8: # and (fn.endswith(".jpg") or fn.endswith(".jpeg"))
        print("Performing clamp and rounding")
        img, _, _ = centralize(img, clip = True)
        img = (img * 255).round().astype(np.uint8)
        
    pyplot.imsave(fn, img, *args, **kwargs)
    
    
if __name__ == "__main__" or True:
    argparser = argparse.ArgumentParser(description = 
        "Frequency domain image steganography/waterprint/signature.")
    argparser.add_argument("input", metavar = "file", 
        type = str, help = "Original image filename.")
    argparser.add_argument("-o", "--output", dest = "output",
        type = str, help = "Output filename.")
    argparser.add_argument("-s", "--secret", dest = "secret",
        type = str, help = "Secret to generate index mapping.")
    # encode
    argparser.add_argument("-i", "--image", dest = "imagesign", 
        type = str, help = "Signature image filename.")
    argparser.add_argument("-t", "--text", dest = "textsign",
        type = str, help = "Signature text.")
    argparser.add_argument("-a", "--alpha", dest = "alpha",
        type = float, help = "Signature blending weight.")
    # decode
    argparser.add_argument("-d", "--decode", dest = "decode", 
        type = str, help = "Image filename to be decoded.")
    # other
    argparser.add_argument("-v", dest = "visual",
        action = "store_true", default = False, help = "Display image.")
    args = argparser.parse_args()
        
    oa = pyplot.imread(args.input)
    margins = (oa.shape[0] // 7, oa.shape[1] // 7)
    margins=(1,1)
    xmap = xmapGen((oa.shape[0]//2-margins[0]*2, oa.shape[1]-margins[1]*2), args.secret)
    
    if args.decode:
        xa = pyplot.imread(args.decode)
        xb = decodeImage(xa, xmap, margins, oa)
        if args.output is None:
            base, ext = path.splitext(args.decode)
            args.output = base + "-" + path.basename(args.input) + ext
        imsaveEx(args.output, xb)
        print("Image saved.")
        if args.visual:
            pyplot.figure()
            imshowEx(xb, title = "deco")
    else:
        if args.imagesign:
            ob = pyplot.imread(args.imagesign)
            ea, fa = encodeImage(oa, ob, xmap, margins, args.alpha)
            if args.output is None:
                base, ext = path.splitext(args.input)
                args.output = base + "+" + path.basename(args.imagesign) + ext
        elif args.textsign:
            ea, fa = encodeText(oa, args.textsign, xmap, margins, args.alpha)
            if args.output is None:
                base, ext = path.splitext(args.input)
                args.output = base + "_" + path.basename(args.textsign) + ext
        else:
            print("Neither image or text signature is not given.")
            exit(2)
        
        imsaveEx(args.output, ea)
        print("Image saved.")
        if args.visual:
            xa = pyplot.imread(args.output)
            xb = decodeImage(xa, xmap, margins, oa)
    
            pyplot.figure()
            pyplot.subplot(221)
            imshowEx(ea, title = "enco")
            pyplot.subplot(222)
            imshowEx(normalizedRGB(xa) - normalizedRGB(oa), title = "delt")
            pyplot.subplot(223)
            imshowEx(fa, title = "freq")
            pyplot.subplot(224)
            imshowEx(xb, title = "deco")
            pyplot.show() #display

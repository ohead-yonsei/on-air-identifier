"""Unchanged colour/scoring functions extracted from the supplied progress document.
Not the complete detector/assignment pipeline; no target-data fitting.
"""
import os
import numpy as np
import cv2

FX = set(os.environ.get("FX","brown,rel47,edge,unknown,small").split(","))      # which fixes are on (for ablation against the labelled frames)
PAL = ["white","black","red","maroon","orange","yellow","green","skyblue","blue","purple","pink","grey","brown"]
CAP = {1:"white",2:"yellow",3:"red",4:"black",5:"blue",6:"green",7:"maroon",8:"pink",9:"purple",10:"skyblue"}

def quant(bgr, ref=False):
    """pixel-wise named colour (calibrated on the 2026-09-06 Seoul broadcast, which is washed out); index into PAL"""
    hsv = cv2.cvtColor(bgr.reshape(-1,1,3), cv2.COLOR_BGR2HSV).reshape(-1,3).astype(int)
    h,s,v = hsv[:,0],hsv[:,1],hsv[:,2]
    out = np.full(len(hsv), PAL.index("grey"))
    def put(n,m): out[m] = PAL.index(n)
    warm = (h>=15)&(h<=45)
    warm = (h>=20)&(h<=45)
    # low-saturation bright pixels: sand is pinkish (hue >=125 or <15), a white cap/silk is cool (45-125), a pale yellow cap is warm (15-45)
    if ref: put("white",(s<30)&(v>185))
    else:   put("white",(s<30)&(v>228)&(h>=45)&(h<125))
    put("yellow",warm&(s>=10)&(v>150))
    put("black",(v<70)&(s<90)); put("blue",(v<110)&(s>=60)&(h>=105)&(h<128))              # navy
    put("red",(s>45)&((h<8)|(h>=168))&(v>110)); put("maroon",(s>45)&((h<8)|(h>=160))&(v>=60)&(v<=110))
    (put("brown",(h>=8)&(h<28)&(s>=45)&(v>=60)&(v<=185)), put("orange",(s>120)&(h>=8)&(h<20)&(v>170))) if "brown" in FX else put("orange",(s>70)&(h>=8)&(h<18)&(v>140))      # tan / brown cap in sunlight; saturated bright = orange
    put("green",(s>35)&(h>=38)&(h<85)&(v>70)); put("skyblue",(s>30)&(h>=85)&(h<108)&(v>140))
    put("blue",(s>70)&(h>=108)&(h<124)&(v>=110)); put("purple",(s>55)&(h>=124)&(h<146)&(v>80)); put("pink",(s>38)&(h>=146)&(h<168)&(v>130))
    return out
def fgmask(patch, thr=20):
    """pixels that differ from the patch border colour (sand / turf / rail behind the rider)"""
    L = cv2.cvtColor(patch, cv2.COLOR_BGR2LAB).astype(np.float32)
    bg = np.median(np.concatenate([L[0,:], L[:,0], L[:,-1]]), 0)
    return (np.linalg.norm(L-bg, axis=2) > thr).reshape(-1)
A = np.eye(len(PAL))
for a,b,v in [("brown","maroon",.4),("brown","orange",.4),("brown","black",.3),("red","maroon",.6),("skyblue","blue",.6),("orange","yellow",.5),("orange","red",.5),("maroon","pink",.25),("purple","blue",.3),("maroon","purple",.25)]:
    A[PAL.index(a),PAL.index(b)] = A[PAL.index(b),PAL.index(a)] = v
G = PAL.index("grey")
def hist(idx, w=None, torso=None):
    hh = np.bincount(idx, weights=w, minlength=len(PAL)).astype(float)
    known = 1 - hh[G]/max(hh.sum(),1e-6); hh[G] = 0
    if torso: hh[PAL.index("white")] *= 0.5          # every jockey wears white breeches; rails are white
    return hh/max(hh.sum(),1e-6), known
def soft(a,b):
    """how much of b's colour mass is explained by a (with near-colour credit)"""
    return float(np.minimum(b, A@a).sum())

def cap_score(ch, n, ref=None):
    """share of the (foreground) cap patch that has this number's cap colour. Calibrated on the user's corrections (2026-11-16):
    brown caps read black in shade, shaded red caps read maroon, sky-blue striped caps read blue+white."""
    I = PAL.index; w, bk, br = ch[I("white")], ch[I("black")], ch[I("brown")]; sky, bl = ch[I("skyblue")], ch[I("blue")]
    if n == 1:
        v = w
        if "white" in FX:
            for c in ("skyblue","yellow","blue"):
                if ref is None or ref[I(c)] < 0.15: v -= 1.5*ch[I(c)]
    elif n == 3 and "brown" in FX: v = ch[I("red")] + 0.8*ch[I("maroon")]
    elif n == 4 and "brown" in FX: v = bk + 0.3*br
    elif n == 7 and "brown" in FX: v = br + 0.45*bk
    elif n == 5 and "sky" in FX: v = bl*max(0.0, 1-1.5*w) + 0.3*sky
    elif n == 10 and "sky" in FX: v = (sky + 0.5*bl)*max(0.0, 1-1.2*w)
    elif n >= 11 and "sky" in FX:
        base = sky + bl; stripe = ch[I(CAP[n-10])] if n != 15 else bl
        v = min(1.0, base + stripe) if (base > 0.25 and stripe > 0.08) else 0.3*base
    elif n >= 11: v = (sky + 0.6*ch[I(CAP[n-10])]) if sky > 0.12 else 0.0
    else: v = ch[I(CAP[n])] + (0.5*ch[I("red")] if CAP[n]=="maroon" else 0)
    return float(np.clip(v*1.5, 0, 1))

def silk_ref(path):
    im = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if im.shape[2]==4: m = im[:,:,3].reshape(-1)>128; px = im[:,:,:3].reshape(-1,3)[m]
    else: px = im.reshape(-1,3)
    return hist(quant(np.ascontiguousarray(px), ref=True))[0]
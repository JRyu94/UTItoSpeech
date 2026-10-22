"""Trajectory conditioning used by the paper, before alignment and common cropping."""
from pathlib import Path
import csv
import numpy as np
from scipy import signal
from .alignment import resolve_lip_offset

TONGUE=['tongueRoot1','tongueRoot2','tongueBody1','tongueBody2','tongueDorsum1','tongueDorsum2','tongueBlade1','tongueBlade2','tongueTip1','tongueTip2','vallecula','hyoid','mandible','shortTendon']
LIP=['leftLip','rightLip','topleftinner','topmidinner','toprightinner','bottomleftinner','bottommidinner','bottomrightinner']

def read_dlc_csv(path,landmarks):
    with Path(path).open(newline='',encoding='utf-8-sig') as f:rows=list(csv.reader(f))
    columns=[next(i for i,(part,axis) in enumerate(zip(rows[1],rows[2])) if part==p and axis==a) for p in landmarks for a in ['x','y','likelihood']]
    values=np.array([[float(row[i]) if row[i] else np.nan for i in columns] for row in rows[3:] if row],dtype=np.float32).reshape(-1,len(landmarks),3)
    return values[:,:,:2],values[:,:,2]

def interpolate(values):
    out=np.array(values,dtype=np.float64,copy=True);time=np.arange(len(out))
    for j in range(out.shape[1]):
        valid=np.isfinite(out[:,j])
        if not valid.any():raise ValueError('No usable samples in a coordinate series')
        out[:,j]=np.interp(time,time[valid],out[valid,j])
    return out

def primary_condition(xy,confidence):
    """Confidence >=0.1, point-wise population-SD outliers, fourth-order zero-phase LPF."""
    xy=np.asarray(xy,dtype=np.float64);confidence=np.asarray(confidence)
    bad=(confidence<.1)|~np.isfinite(xy).all(axis=2)
    for point in range(xy.shape[1]):
        valid=~bad[:,point]
        if valid.sum()>=2:
            mu=xy[valid,point].mean(0);sd=xy[valid,point].std(0)
            bad[:,point]|=(np.abs(xy[:,point]-mu)>3*np.maximum(sd,1e-6)).any(1)
    repaired=np.empty_like(xy);time=np.arange(len(xy))
    for point in range(xy.shape[1]):
        valid=~bad[:,point]
        for axis in range(2):
            repaired[:,point,axis]=np.interp(time,time[valid],xy[valid,point,axis]) if valid.sum()>=2 else np.nanmedian(xy[:,point,axis])
    b,a=signal.butter(4,20./30.)
    if len(xy)>3*max(len(a),len(b)):repaired=signal.filtfilt(b,a,repaired,axis=0)
    if not np.isfinite(repaired).all():raise ValueError('The raw-median fallback did not yield finite coordinates')
    return repaired.astype(np.float32)

def public_condition(xy,confidence):
    """Published-code filter semantics, with the shared paper normalizer applied separately.

    This implementation follows confidence interpolation, strict per-axis sample-SD
    filtering, and the third-order causal Butterworth with a duplicated 15-frame prefix.
    Upstream code is referenced, not bundled; see docs/PROVENANCE.md.
    """
    xy=np.asarray(xy,dtype=np.float64);a=xy.reshape(len(xy),-1).copy()
    a[~np.repeat(np.asarray(confidence)>.1,2,axis=1)]=np.nan
    a=interpolate(a);mean=a.mean(0);sd=a.std(0,ddof=1)
    a[~((a<mean+3*sd)&(a>mean-3*sd))]=np.nan
    a=interpolate(a)
    if len(a)<15:raise ValueError('The reference filter requires at least 15 frames')
    sos=signal.butter(3,20,output='sos',fs=60)
    a=signal.sosfilt(sos,np.concatenate([a[:15],a]),axis=0)[15:]
    return a.reshape(xy.shape).astype(np.float32)

def make_positions(tongue_xy,tongue_conf,lip_xy,lip_conf,explicit_offset=None,metadata_offset=None,conditioning='primary'):
    fn={'primary':primary_condition,'public':public_condition}[conditioning]
    tongue=fn(tongue_xy,tongue_conf);lip=fn(lip_xy,lip_conf)
    offset=resolve_lip_offset(explicit_offset,metadata_offset,len(lip),len(tongue))
    length=min(len(tongue),len(lip)-offset)
    if length<=0:raise ValueError('No overlapping frames after offset correction')
    # Lip precedes tongue in the original cache; changing this order changes the model input.
    flat=np.concatenate([lip[offset:offset+length],tongue[:length]],axis=1).reshape(length,-1)
    positions=((flat-flat.mean(0))/np.maximum(flat.std(0),1e-3)).astype(np.float32)
    assert positions.shape[1]==44
    return positions,{'offset':offset,'frames':length,'dimensions':44,'conditioning':conditioning}

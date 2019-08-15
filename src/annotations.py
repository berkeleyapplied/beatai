from wfdb import rdann
import os

def getAnnotations(file):

    base, ext = os.path.splitext(file)
    ext = ext.replace('.','')
    ann = rdann(base, ext)
    return ann.sample, ann.symbol, ann.subtype

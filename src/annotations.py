from wfdb import rdann
import os

def getAnnotations(file):

    base, ext = os.path.splitext(file)
    ext = ext.replace('.','')
    ann = rdann(base, ext)
    return ann.sample, ann.symbol, ann.subtype


def extractBeatAnnotationLocations(ann):

    symlist = 'NLRBAaJSVrFejnE/fQ'

    loc, sym, sub = ann

    pk = []
    s = []

    for i in range(len(sym)):
        if sym[i] in symlist:
            pk.append(loc[i])
            s.append(sym[i])

    return pk, s




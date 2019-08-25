from network_cluster import *
import pickle


def load_data(dfile):
    root, datan, bd, filter, nstd, depth, beat_range, dset, G, cls, part, pk1 = pickle.load(open(dfile, 'rb'))
    return root, datan, bd, filter, nstd, depth, beat_range, dset, G, cls, part, pk1


def plot_class_symbols(dfile, clsnum=None, sample_range=None):
    root, datan, bd, filter, nstd, depth, beat_range, dset, G, cls, part, pk1 = load_data(dfile)
    plot_community_symbols(root, datan, cls, pk1, nclass=clsnum, srange=sample_range)


def plot_class_envelope(dfile, clsnum, nstd=1.0, clr='k',traces=True,alpha=0.2):
    root, datan, bd, filter, nstd, depth, beat_range, dset, G, cls, part, pk1 = load_data(dfile)
    plot_community_envelope(dset, cls, clsnum, nstd=nstd, clr=clr, traces=traces, alpha=alpha)


import numpy as np
from os import path

mypath = path.dirname(path.abspath(__file__))

def X_trn():
    fn = path.join(mypath, 'X_trn.mm')
    arr = np.memmap(fn, dtype='float32', shape=(50000, 12800), order='C',
                    mode='c')
    return arr

def y_trn():
    fn = path.join(mypath, 'y_trn.mm')
    arr = np.memmap(fn, dtype='int32', shape=(50000,), order='C',
                    mode='c')
    return arr

def K_trn():
    fn = path.join(mypath, 'K_trn.mm')
    arr = np.memmap(fn, dtype='float32', shape=(50000, 50000), order='C',
                    mode='c')
    return arr

def X_tst():
    fn = path.join(mypath, 'X_tst.mm')
    arr = np.memmap(fn, dtype='float32', shape=(10000, 12800), order='C',
                    mode='c')
    return arr

def y_tst():
    fn = path.join(mypath, 'y_tst.mm')
    arr = np.memmap(fn, dtype='int32', shape=(10000,), order='C',
                    mode='c')
    return arr

def K_tst():
    fn = path.join(mypath, 'K_tst.mm')
    arr = np.memmap(fn, dtype='float32', shape=(50000, 10000), order='C',
                    mode='c')
    return arr

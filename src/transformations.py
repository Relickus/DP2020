import torch
import logging
import numpy as np

class Flip(object):
    def __init__(self, axis=None):
        self._transform = flip
        self._axis=axis

    def __call__(self, x, return_log=False):
        res,param = self._transform(x,axis=self._axis)

        if return_log:
            log = {}
            log[str(self)] = param
            return res, log
        else:
            return res


    def __repr__(self):
        return str(self)

    def __str__(self):
        return type(self).__name__

class Rotate(object):
    def __init__(self, k=None):
        self._transform = rotate
        self._k=k

    def __call__(self, x, return_log=False):
        res,param = self._transform(x,k=self._k)
        
        if return_log:
            log = {}
            log[str(self)] = param
            return res, log
        else:
            return res

    def __repr__(self):
        return str(self)

    def __str__(self):
        return type(self).__name__

def patch2tensor(patch):
    """
    Convert numpy patch to tensor.

    Convert numpy image patch of dims W x H x D to a torch tensor of dims C x D x H x W.
    """
    p=patch.transpose((2,1,0))
    tensor = torch.from_numpy(p)
    # add virtual channel dimension
    tensor = tensor.unsqueeze(0).float()
    return tensor

def standardize(x):
    """
    Standardize the whole tensor to mean=0 and std=0.
    """
    res = (x - x.mean()) / x.std()
    return res


def rotate(x,k=None):
    """
    Wrapper for np.rot90 function.
    Rotate numpy ndarray along axial plane by 90 deg. k (or random number of) times.
    """

    if k is None:
        k=np.random.choice([0,1,2,3])

    rotx = np.rot90(x,int(k),axes=(0,1))
    #logging.debug(f"rot: {k}")
    return rotx, k

def flip(x, axis=None):
    """
    Wrapper for np.flip function.
    Flip numpy ndarray along given (otherwise random) axis.
    """
    
    if axis is None:
        axis=np.random.choice([0,1])
    
    #logging.debug(f"flip:{axis}")
    flipx = np.flip(x, axis)
    return flipx, axis

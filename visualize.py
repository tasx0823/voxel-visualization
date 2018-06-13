import os
import os.path
from util import *
from util_vtk import visualization
import argparse
import numpy as np


filename = 'chair_000000182_1.mat'
#filename = 'voxel.mat'

matname = 'instance'
threshold = 0.01
ind = 1 - 1  # matlab use 1 base index
downsample_factor = 1
downsample_method = 'max'
uniform_size = 0.9
use_colormap = "store_true"
connect = 3  #default is 3
assert downsample_method in ('max', 'mean')

def np_read_tensor(filename):
    """ return a 4D matrix, with dimensions point, x, y, z """

    voxels = np.load(filename)

    dims = voxels.shape
    if len(dims) == 5:
        assert dims[1] == 1
        dims = (dims[0],) + tuple(dims[2:])
    elif len(dims) == 3:
        dims = (1,) + dims
    else:
        assert len(dims) == 4
    result = np.reshape(voxels, dims)
    return result

def load_tensor(filename, varname='instance'):
    """ return a 4D matrix, with dimensions point, x, y, z """
    assert(filename[-4:] == '.mat')
    mats = loadmat(filename)
    if varname not in mats:
        print(".mat file only has these matrices:")
        for var in mats:
            print(var)
        assert(False)

    voxels = mats[varname]

    dims = voxels.shape
    #print('dims is : ',dims)
    if len(dims) == 5:
        voxels = np.squeeze(voxels, axis=4)
        dims = dims[:4]
        #assert dims[1] == 1
        #dims = (dims[0],) + tuple(dims[2:])
    elif len(dims) == 3:
        dims = (1,) + dims
    else:
        assert len(dims) == 4
    result = np.reshape(voxels, dims)
    return result

# read file
print("==> Reading input voxel file: "+filename)
voxels_raw = load_tensor(filename)   # shape : batch_size x h x w x c
#voxels_raw = read_tensor(filename)
print('shape 1 : ',np.shape(voxels_raw))
print("Done")

voxels = voxels_raw[ind]   # shape : h x w x c
print('shape2 : ',np.shape(voxels))
#voxels = np.squeeze(voxels,axis=3)

# keep only max connected component
print("Looking for max connected component")

if connect > 0:
    voxels_keep = (voxels >= threshold)
    voxels_keep = max_connected(voxels_keep, connect)
    voxels[np.logical_not(voxels_keep)] = 0
    #voxels[voxels_keep] = 1

# downsample if needed
if downsample_factor > 1:
    print("==> Performing downsample: factor: "+str(downsample_factor)+" method: "+downsample_method)
    voxels = downsample(voxels, downsample_factor, method=downsample_method)
    print("Done")


visualization(voxels, threshold, title=str(ind+1)+'/'+str(voxels_raw.shape[0]), uniform_size=uniform_size, use_colormap=use_colormap)

# # import numpy as np
# import h5py
# import matplotlib.pyplot as plt
# # a = glob.glob('./proc/72/*.h5')
# # print(a)
# with h5py.File('./proc/72/40.h5', 'r') as hdf:
#     base_items = list(hdf.items())
#     print('Items in the base directory:', base_items)
#     print(hdf['X'].shape)

# import os
# #import Tables
# import pandas as pd
#
#
# direc = './proc/72'
# dirs = os.listdir(direc)
#
# for i in dirs:
#     if i.endswith('.h5'):
#         pd.read_hdf(direc , key )
#         # hdf5 = Tables.openFile(os.path.join(direc,idir))
#         # hdf5.close()
#
import glob
from PIL import Image
import numpy as np
X = []
for filename in glob.glob('./proc/72/*.h5'):  # /{}...format(i)
    im = Image.open(filename)
    X.append(np.array(im))  # turn to array and #resize images to (200,200)   , append to the X list above
    print(X)
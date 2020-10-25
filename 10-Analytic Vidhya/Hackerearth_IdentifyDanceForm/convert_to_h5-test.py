import pandas as pd
import os,shutil,math,scipy,cv2
import numpy as np
import matplotlib.pyplot as plt
import random as rn

from time import time
from glob import glob
from tqdm import tqdm
from skimage.io import imread
from IPython.display import SVG

from scipy import misc,ndimage
from scipy.ndimage.interpolation import zoom
# from scipy.ndimage import imread



dict = {'bharatanatyam' : 0,
 'kathak' : 1,
 'kathakali' : 2,
 'kuchipudi' : 3,
 'manipuri' : 4,
 'mohiniyattam' : 5,
 'odissi' : 6,
 'sattriya' : 7}

img_dir = 'test'
csv_file = 'test.csv'
danceNames = pd.read_csv(csv_file)
danceNames = np.array(danceNames)

X = []
Z = []
names = []
imgsize = 256          #256,150,224


for item in danceNames:
    img = item[0]
    for image in tqdm(os.listdir(img_dir)):
        if image == img:
            print(img , image)
            path = os.path.join(img_dir,img) 
            img = cv2.imread(path,cv2.IMREAD_COLOR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            img = cv2.resize(img,(imgsize,imgsize))
            # plt.imshow(img)
            X.append(np.array(img))
            names.append(np.string_(image))
            break

# print(X,names)


import numpy as np
import h5py


archive = h5py.File('test_256.h5', 'w')
archive.create_dataset('/X', data=X)
archive.create_dataset('/names',data=names)
archive.close()



dataset = h5py.File('test_256.h5', "r")
XX = dataset["X"][:]
nnames = dataset["names"][:]

print(XX.shape,nnames.shape)
dataset.close()

# train_150
# train_256
# train_224
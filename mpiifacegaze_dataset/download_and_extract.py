import os
import sys
import cv2
import wget
import h5py
import pickle
import zipfile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ZIP_FILE = 'MPIIFaceGaze_normalized.zip'
DATASET_DIR = 'MPIIFaceGaze_normalizad'

# Download .zip if not present in folder
if not os.path.isfile(ZIP_FILE):
    url = 'http://datasets.d2.mpi-inf.mpg.de/MPIIGaze/MPIIFaceGaze_normalized.zip'
    print('Downloading dataset from {}...'.format(url))
    wget.download(url, '.')

# Extract files if not already done
if not os.path.isdir(DATASET_DIR) and os.path.isfile(ZIP_FILE):
    print('Extracting dataset .zip file...')
    with zipfile.ZipFile(ZIP_FILE, 'r') as zipObj:
        zipObj.extractall()

dest = 'raw/'
if not os.path.isdir(dest):
    os.mkdir(dest)
    
# Parse and convert data from .mat to raw images and pickle objects
print('Converting dataset. Destination folder: {}'.format(os.path.abspath(dest)))
for r, d, f in os.walk(DATASET_DIR):
    for file in f:
        if not file.endswith('.mat'): continue
        sid = file.split('.')[0][1:]
        print('Parsing data for subject {}'.format(int(sid)))
        dest_folder = os.path.join(dest, sid)
        if not os.path.exists(dest_folder):os.mkdir(dest_folder)        
        gt_obj = {'subject_id': sid,
                 'images':[],
                 'labels':[]}
        
        with h5py.File(os.path.join(DATASET_DIR, file), 'r') as mat:
            for k, v in mat.items():
                data = np.array(mat['Data']['data'])
                label = np.array(mat['Data']['label'])
            
        for idx, (img, gt) in enumerate(zip(data, label)):
            img_file =  '{:04}.png'.format(idx)
            img_path = os.path.join(dest_folder, img_file)
            cv2.imwrite(img_path, img.transpose(1,2,0))
            gt_obj['images'].append(img_path)
            gt_obj['labels'].append(gt)
            
        pickle_file = os.path.join(dest_folder, 'labels.dict')
        with open(pickle_file, 'wb') as pf:
            pickle.dump(gt_obj, pf)

        del data
        del label
import numpy as np
import os
from tqdm import trange, tqdm
from skimage import io, transform
import requests
import OpenEXR

fairchild_data = "http://rit-mcsl.org/fairchild/files/HDRPS_Raws.zip"


def download_file(url):
    local_filename = url.split('/')[-1]
    r = requests.get(url, stream=True)
    with open(local_filename, 'wb') as f:
        for chunk in r.iter_content(chunk_size=1024): 
            if chunk:
                f.write(chunk)
    return local_filename


def load_data(size=5):
    files_in = os.listdir(DIRECTORY)
    files = np.random.choice(files_in, size=size)
    images = []
    for f in tqdm(files):
        images.append(transform.resize(io.imread(DIRECTORY + '/' + f), (SQ_IMG_SIZE,SQ_IMG_SIZE,3), mode='constant'))
    result = np.asarray(images)
    return result


def load_exr(filename):
    loaded_exr = OpenEXR.InputFile(filename)
    print(loaded_exr)

#fairchild_zip = download_file(fairchild_data)
load_exr("../data/PeckLake.exr")

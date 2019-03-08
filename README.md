# cs230-proj

## Setup
First, build the virtual environment and install all dependencies with:
```
make install
```

Next, activate the virtual environment with:
```
source cs230_venv/bin/activate
```

You are ready to run the model.

## Preprocessing
To run preprocessing on HDR dataset raws, generate baseline tone-mapped images:
```
./src/data-preprocess.py ../data/hdr_image_dataset ../data/proc_image_dataset/
```


## Data
Unprocessed raw images (Fairchild HDR dataset):
```
data/hdr_image_dataset/
```

8-bit RGB images (all exposures + baseline render):
```
data/proc_image_dataset/
```

Locally tone-mapped HDR renders (ground truth):
```
data/hdr_image_dataset_renders/
```


## Baseline


## Model script
The file `src/model.py` contains the beginnings of our outlined architecture. When complete, it will be the driver of our entire workflow.

Note that as of Feb 21, this code does not yet properly run.


## Misc
#### HDR Example
https://www.sony-semicon.co.jp/products_en/IS/sensor2/technology/dol-hdr.html

#### Data sources
[Fairchild](http://rit-mcsl.org/fairchild/HDR.html "Fairchild data")

#### Repo TODOs
* Add data augmentation capabilities to preprocessing scripts
* Improve IQ for automatic tone-mapping algorithm via tuning for baseline
* Build jupyter notebook.

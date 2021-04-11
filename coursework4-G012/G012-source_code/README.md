## MLP Coursework 4
This README file indicates how to run the code for the MLP Coursework 4 title "Tackling Class Imbalance on Agriculture-Vision Semantic Segmentation".

## File structure of the source code
```
.
├── Dataset
└── DeepLabV3Plus
    ├── checkpoints
    ├── config
    ├── data
    │   └── AgricultureVision
    ├── experiment_scripts
    │   ├── down_sample_scripts
    │   │   ├── augmented
    │   │   └── losses
    │   ├── full_set
    │   ├── over_sample_scripts
    │   │   └── losses
    │   └── test_scripts
    ├── lib
    │   ├── loss
    │   ├── net
    │   └── utils
    ├── metrics
    ├── network
    │   └── backbone
    ├── notebooks
    ├── plots
    └── utils
```
	
## How to download the data
The data should be downloaded by running ``wget https://www.dropbox.com/s/wpwhb517ck4o7vn/Agriculture-Vision.tar.gz?dl=0`` and unzipping it into the ``Dataset`` directory.

## How the program can be run
From the ``DeepLabV3Plus`` directory you can run ``bash PATH\_TO\_SCRIPT``. All the scripts are inside the ``experiments_scripts`` folder.

To create a reduced version of the dataset from the root directory run ``python Dataset/create_reduced_dataset.py``. Then in ``DeepLabV3Plus/data/Agriculture-Vision/pre_process.py`` comment line 13 (``DATASET_ROOT = '../Dataset/Agriculture-Vision'``) and uncomment line 15 (``#DATASET_ROOT = '../Dataset/Reduced_dataset'``).


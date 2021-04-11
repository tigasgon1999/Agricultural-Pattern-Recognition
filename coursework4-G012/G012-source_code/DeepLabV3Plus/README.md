## MLP Coursework 4


## File structure of the source code
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
	
## How to run the code
The data should be downloaded from https://www.dropbox.com/s/wpwhb517ck4o7vn/Agriculture-Vision.tar.gz?dl=0 into the ``Dataset`` directory.
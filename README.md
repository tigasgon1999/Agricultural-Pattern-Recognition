## Tackling Class Imbalance on Agriculture-Vision Semantic Segmentation 
We focus on improving DeepLabV3+’s performance on the Agriculture-Vision semantic segmentation task by addressing the class imbalance present in the data. This is done by experimenting with different loss functions and augmentation and oversampling schemes. We manage to achieve competitive results with an mIoU of 46.40% on the test set by adding an Adaptive Class Weighting mechanism on the loss function of the Baseline Model.

## Dataset and task
We will perform a semantic segmentation task on the Agriculture-Vision data set (Chiu et al., 2020b), which consists of high-quality aerial farmland images with labelled
agricultural patterns. More specifically, we will use the challenge data set1 used for a public competition ran in 2020 (Chiu et al., 2020a), which is a subset of the full data
set. This subset contains 21,061 images of 512 x 512 pixels in four channels: RGB and Near Infra-red (NIR). We follow the split used in the competition and use 12,901 images for
training, 4,431 for validation, and 3,729 for testing. These 512 x 512 are image patches from large farmland images that were cropped around annotated regions in the image. A sample from each class can be seen below.

figure with sample images

## Problem
Almost 70% of the images have an annotation of the most common class,Weed Cluster, whereas, there are 4 classes with an annotation on less than 5% of the images. The number of images by annotation class is shown in Figure 2. In addition to this, different classes have different sizes and shapes and appear at different frequencies. This is reflected on the total number of pixels for each class, with Weed cluster having significantly more pixels than all the other classes. This can be visualized below.

figure with class imbalance

## Methodology and Final Results
To tackle the class imbalance, we couple different loss functions to the architecture, perform data augmentation and oversampling with the hopes of increasing the performance of DeepLabv3+ on the Agriculture-Vision dataset by improving its performance on the sparse and imbalanced classes. This study thus contributes with an extensive exploration into multiple ways in which class imbalance can be addressed on Agriculture aerial imagery. Our final model improves DeepLabv3+ performance by 5 percentage points, and qualitatively performs better, as shown below.

figure with final qualitative results

## How to run the code.
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


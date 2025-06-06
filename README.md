# MDMU_Net
This project is the official implementation of **"MDMU-Net:3D Multi-dimensional Decoupled Multi-scale U-Net for Pancreatic Cancer Segmentation"**.
This is code for MDMU-Net(for segment PC).
![image](https://github.com/SerendipityInTheWorld/MDMU_Net/blob/main/img1.png)

# Dependencies and Installation
- Python >= 3.10
- PyTorch >= 1.12.0
- monai == 0.8.0
- Other dependencies refer to requirements.txt (or pip install requirements.txt)

# Dataset Preparation
we use MSDPT Dataset(https://msd-for-monai.s3-us-west-2.amazonaws.com/Task07_Pancreas.tar) and NIHP Dataset(https://www.cancerimagingarchive.net/collection/pancreas-ct/).

# Training
**run model_train.py**

This is a direct training MDMUNet, where the detailed parameters are all included in it, and only the correct dataset root needs to be adjusted
The preconditioning.py is about data processing, such as datasets and dataloaders.
# Testing
**run model_test.py**

This is a direct testing MDMUNet, where the detailed parameters are all included in it, and only the correct dataset root needs to be adjusted.
However, detailed metric calculations also require running change_ResuliFile.py files
# Result
## Comparison of model parameters, floating-point numbers and Dice results of pancreatic cancer segmentation
![image](https://github.com/SerendipityInTheWorld/MDMU_Net/blob/main/img2.png)
## Ablation experiments
![image](https://github.com/SerendipityInTheWorld/MDMU_Net/blob/main/img3.png)
## Visualize the results
![image](https://github.com/SerendipityInTheWorld/MDMU_Net/blob/main/img4.png)
# Contact
if you have any question, please email lulian_email@163.com

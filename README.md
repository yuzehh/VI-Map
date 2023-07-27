This is the official implementation of the paper **VI-Map: Infrastructure-Assisted Real-Time HD Mapping for Autonomous Driving**. 

# VI-Map

VI-Map is the first system that leverages roadside infrastructure to enhance real-time HD mapping for autonomous driving. In contrast to the single-vehicle online HD map construction, VI-Map empowers vehicles with a significantly more precise and comprehensive HD map from roadside infrastructure. Witness the capabilities of VI-Map in action through our demo video below. 

https://github.com/yuzehh/VI-Map/assets/67583016/388181dd-035f-4e6a-9f59-47d653f24db1

## System Overview

The key idea of VI-Map is to exploit the unique spatial and temporal observations of roadside infrastructure to build and maintain an accurate and up-to-date HD map, which is then fused by the vehicle with its on-vehicle HD map in real-time to boost/update the vehicle’s scene understanding. 

![teaser](https://github.com/yuzehh/VI-Map/blob/master/images/teaser.png)

## Requirements
VI-Map's artifact evaluation relies on some basic hardware and software environment as shown below. The listed environment versions are the ones we have tested, but variations with slight differences should still be compatible.

| Hardware Environment  | Version |
| ------------- | ------------- |
| GPU  | 1 x NVIDIA Geforce RTX 2060 SUPER (~1.2GB) |
| CPU | Intel(R) Core(TM) i7-10700F CPU @ 2.90GHz |
| RAM | suggest more than 10GB |

| Software Environment  | Version |
| ------------- | ------------- |
| OS  | Ubuntu Server 18.04.6 LTS |
| NVIDIA Driver  | 515.57  |
| CUDA Version  | 11.7  |

![nvidia-smi](https://github.com/yuzehh/VI-Map/blob/master/images/nvidia-smi.png)


## Infrastructure-End Code and Evaluation

### Introduction 

At the infrastructure end, the infrastructure leverages its two unique observations: the accumulated 3D LiDAR point cloud and the precise vehicle trajectories, to estimate a precise and comprehensive HD map. Specifically, the infrastructure extracts meticulously designed bird-eye-view (BEV) features from the two pieces of data sources and then employs them for efficient map construction.

![infra](https://github.com/yuzehh/VI-Map/blob/master/images/infra_pipline.png)

### Install conda environment
```
git clone https://github.com/yuzehh/VI-Map.git
cd VI-Map/infrastructure/
conda env create -f infra_env.yml
```
These steps will create a Conda environment named "VI-Map_infra".

### Execute
For a quick demo and evaluation:
```
conda activate VI-Map_infra
cd VI-Map/infrastructure/
python3 vis_pred.py 
```
This process will generate HD maps for different infrastructures' inputs, and you can find the results in the "infrastructure/vis_results" directory. The images labeled as "evalXXX.png" are the visualizations of the generated HD maps.


## Vehicle-End Code and Evaluation

### Introduction 

At the vehicle end, the vehicle receives the HD map from the infrastructure, and then integrates the infrastructure's HD map with its own HD map. A new
three-stage map fusion algorithm is designed to merge the HD map from the infrastructure with the on-vehicle one.

![veh](https://github.com/yuzehh/VI-Map/blob/master/images/veh_pipline.png)

### Install conda environment
```
cd VI-Map/vehicle/
conda env create -f vehicle_env.yml
```
These steps will create a Conda environment named "VI-Map_veh".

### Execute
For a quick demo and evaluation:
```
conda activate VI-Map_veh
cd VI-Map/vehicle/
python3 vehicle_receive.py 
```
This process will generate the fused HD map for several vehicle-infrastructure HD map pairs. As the code runs, it will first visualize the on-vehicle HD map (only for comparison) and then proceed to plot the resulting fused HD map.

## General Dataset for Research on Infrastructure-Assisted Autonomous Driving

This repository currently contains a limited number of data samples for quick demos and artifact evaluation. However, it's essential to note that VI-Map is evaluated on a vast dataset sourced from both the CARLA simulator and real-world scenarios. To contribute to the community, we are delighted to release the entire dataset collected from the CARLA simulator through [this link (Google Drive)](https://drive.google.com/file/d/1xwg10Ueju2GhR2QSmSOABKSzGI0vIuxd/view?usp=sharing). 

This dataset is general as it comprises sensor data(e.g., 3D LiDAR point clouds) and poses from both the roadside infrastructure and vehicle end. Its applicability is not limited to the VI-Map project alone, but to other research related to infrastructure-assisted autonomous driving.

## General Code for Collecting Data in CARLA Simulator for Research on (Infrastructure-Assisted) Autonomous Driving
Cooperative perception between vehicle and infrastructure (V2I) or vehicle and vehicle (V2V) have become emerging research areas, however, acquiring relevant data for research purposes can be challenging. In addition to sharing the dataset mentioned above, we are pleased to release the code used to collect data within the CARLA simulator. You can access the code through this repository: https://github.com/yuzehh/CARLA-Dataset-Creation.git.

This repository contains code that enables the placement of any pose, type, and number of sensors at arbitrary locations (roadside infrastructure or vehicles) in CARLA. Furthermore, it allows for generating varying traffic flows, ranging from light to heavy, while efficiently recording and storing sensor data. We believe this resource can benefit the broader research community and foster advancements in cooperative perception research.

## Citation
If you find the code or dataset useful, please cite our paper.

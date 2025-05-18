# CSCI_4930_ML_Project

Project members: Hossein Mohammadi, Chau Nguyen, Paul Nguyen

This is the code for our 2d to 3d image reconstruction that uses MiDaS and NeRF to evaluate how images can be rendered to 3d from a single 2d image and multiple 2d images


# How to run the code
- clone the git repository
- run "pip install -r requirements.txt" in your terminal
- the main project is in the jupyter notebook called "Machine_Learning_Project_v2"
- to recreate "your_data" the code should be ran

## Directory Structure
nerf-pytorch/

├── your_data/

│ ├── *.jpg / *.png # RGB images from Pix3D (e.g., beds only)

│ ├── transforms_train.json

│ ├── transforms_test.json

│ ├── transforms_val.json

│

├── run_nerf.py # Main training script

├── custom_pix3d.txt # Config file for training

├── logs/

│ └── pix3d_test/ # Output renders, weights, and logs

custom_pix3d

# File/Folder descriptions:
- your_data: contains the extracted bed images from pix3d
- custom_pix3d.txt: the configuration to run the custom dataset on NeRF
- transforms_train.json, transforms_test.json , transforms_val.json : contains information(like camera angle, lighting position, etc.) about the images in /your_data.

**Credits**:
`
@misc{lin2020nerfpytorch,
  title={NeRF-pytorch},
  author={Yen-Chen, Lin},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished={\url{https://github.com/yenchenlin/nerf-pytorch/}},
  year={2020}
}
`

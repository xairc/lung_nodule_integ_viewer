# lung_nodule_integ_viewer
lung nodule integration viewer, integrate of classification, detector, 

# Prerequisites
python 3.5.2      
pytorch 0.2.0
pyqt 5

# Docker Image
docker pull likebullet86/luna16_detector

# Using Viewer
![ex_screenshot](./demo_iomg/lung_integ.png) 
- settings (detector_viewer/xai_viewer.py)
  - self.init_openpath
    - for ct files path 
  - self.rule_data_dir
    - for preprocess data path
  - self.detect_resume
    - for detector.ckpt path
  - self.attribute_resume
    - for attrivute.ckpt path
  - self.gpu
- excute
  - python xai_viewer.py
- for using viewer in docker
  - xhost + (excute in host side)
  - excute docker with following options:
    -v "$HOME/.Xauthority:/root/.Xauthority:rw"
    -v /tmp/.X11-unix:/tmp/.X11-unix
    -e DISPLAY=$DISPLAY 

# Data Download
 - https://luna16.grand-challenge.org/download/
   - download data and candidates_V2.csv
 - https://wiki.cancerimagingarchive.net/display/Public/LIDC-IDRI
   - download Radiologist Annotations/Segmentations (XML)
   
# pretrained weight
 - https://drive.google.com/drive/folders/18UOgQcqVjFrDRbi2eeuBrdheoYbE6vJM?usp=sharing
 - download detector.ckpt, attribute.ckpt to root dir
 
# try flow
 - download data
   - luna, lidc xml, pretrained weight
 - set path
   - config_training.py
   - xai_viewer.py initialize path
 - excute luna16_detector image with docker
 - preprocessing
   - python prepare.py
 - excute viewer
   - python xai_viewer.py
   


# Image Augmentation

This repository contains a single script that allows to perform multiple image augmentation operations based on the imgaug python library. It augment the images along with the annotations (bboxes) provided in PASCAL VOC format annotated with the labelImg annotation tool.

### Instructions

Clone the repository and enter the directory containing the script
```
git clone https://github.com/yieniggu/data-augmentation.git
cd data-augmentation
```
Use the script indicating
+ imagesFolder: folder containing the images to perform the augmentation.
+ annotationsFolder: folder containing the annotations corresponding to the images.
+ imagesOutputFolder: folder to save the augmented images after the operation.
+ labelsOutputFolder: folder to save the augmented annotations after the operation.
+ augmentaion: type of augmentation to apply to the provided images (flip, rotation, gaussian noise, gaussian blur).

For example
```
python3 --imagesFolder /home/user/aug-demo/raw-images/ --annotationsFolder /home/user/aug-demo/raw-labels/ --augmentation flip --imagesOutputFolder /home/user/aug-demo/flipped-images/ --labelsOutputFolder /home/user/aug-demo/flipped-labels/
```
You can even create more complex pipelines to heavily increase your dataset size. For example, applying a rotation operation after flipping the original dataset
```
python3 --imagesFolder /home/user/aug-demo/flipped-images/ --annotationsFolder /home/user/aug-demo/flipped-labels/ --augmentation rotate --imagesOutputFolder /home/user/aug-demo/flipped-rotated-images/ --labelsOutputFolder /home/user/aug-demo/flipped-rotated-labels/
```

### First edit: April 12, 2020

Added four types of augmentation: **flip**, **rotation**, **gaussian noise** and **gaussian blur**.

The main goal of this repo is to provide a simple way to implement a pipeline for multiple computer vision tasks that requires a great amount of examples on the datasets.

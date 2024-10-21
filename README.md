# CMSC 436 Research Paper Extra Credit

## Using Haar Feature-Based Cascade Classifiers for Face Detection on a Raspberry Pi for IoT Applications 

- [Link to Draft on Google Drive](https://docs.google.com/document/d/1H4GUHuNXbkjKFACpi70M0ZhKLtLqq29fNws-9ac8Di0/edit?usp=sharing)
- [Link to Positive Image Set](https://www.kaggle.com/datasets/fareselmenshawii/face-detection-dataset)
- [Link to Negative Image Set](https://www.kaggle.com/datasets/sagarkarar/nonface-and-face-dataset)
- [Link to OpenCV3.4](https://github.com/opencv/opencv/tree/3.4)
- [Link to OpenCV4](https://github.com/opencv/opencv/archive/refs/tags/4.10.0.zip)


## Project Setup

Prequisites: `python-opencv` installed via pip, `opencv` installed via building from source ([link to macos](https://docs.opencv.org/4.x/d0/db2/tutorial_macos_install.html)).

1. Clone the repo.
2. Download the positive image set. Then rename `labels` to `yolo_labels`, `label2` to `labels`, `images` to `pos_images`.
3. Download the negative image set. Rename the image directory to `neg_images` and remove all samples containing human faces with `rm Human*`. Put the `neg_images` directory in the `archive` directory. This is your complete dataset. 

## Usage

**Run the face detection model** with the following command: 

```
> python face_detection.py [-r] [-t]
```

The `-r` flag denotes that the script is being run on a Raspberry Pi, and the script will use the Picamera2 module accordingly. The `-t` flag denotes that we are logging the time it takes to compute a frame.

**Run the model evaluation script** with the following command:

```
> python test_model.py --config <path_to_config>
```

Reference the config files in `config/` to understand how to write a config file for this script. 

**Run the profiling script** with the following command: 

```
python test_utilization.py face_detection.py
```
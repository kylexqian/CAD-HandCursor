# CAD-HandCursor

Gesture controlled mouses have been utilized in a wide variety of fields. Working with a lot CAD software, I felt that the movement scheme for many of these design tools can often be unintuitive— especially when switching between different CAD programs. I believe that using your hands as a controller for design is a potential improvement to this system. This repo is my exploration into this topic.

The goal of this project is to create a cursor controllable with your finger that allows the user to control and manipulate CAD software. This means that users 
should be able to not only rotate and translate pre-existing 3D models, but also sketch and create new CAD models using their hands as the controller.

Here's a brief demo of the final result

<p align="center">
  <img src="https://drive.google.com/uc?export=view&id=10d_RowjbkUoZZQtvqZlFYVlE5YUY7l_b" alt="Demo.gif">
</p>


## Requirements
In order to use this repo make sure you have OpenCV, numpy, and PyAutoGUI. I am not sure of the exact versions of these libraries necessary for this project to work, but I specifically used OpenCV 4.0.1, PyAutoGUI 0.9.52, and numpy 1.18.5.

## Configuration
#### Device Number
You may need to alter what device number you use to choose the camera you desire, it defaults to 0. 
```Python
def __init__(self, camera=0):
    # camera number
    self.camera = camera
```

#### Skin Mask
This project currently uses a fixed range HSV skin mask. For best results, this should be altered to the lighting of the environment you are filming in. I have commented out two different variations I used dependent on the lighting/time of day I was working in.
```Python
# Skin color segmentation mask (HSV)
self.HSV_min = np.array([0,40,50],np.uint8) # 0,40,50 | 0,20,70
self.HSV_max = np.array([50,250,255],np.uint8) # 50,250,255 | 20,255,255
```

#### Video Recording
I set up video recordings to capture each stage of the image processing as an .avi with settings that worked to the device I was working on. I also set up that these videos would record into a folder called Results. This can all be altered to fit whatever device is being recorded on.

```Python
### video writing ###
def run(self, fps=20, result=False, raw=False, HSV=False, contour=False):
```
The parameters to the function above control the recording fps, as well as which image processing stages will be recorded:
* raw is the unprocessed live footage
* HSV is the HSV color space image post skin mask 
* contour is the contour and corresponding convex hull overlayed on the raw footage
* result is the final result, showing the convex hull, contour, and text indicating current action and finger count

```Python
fourcc = cv.VideoWriter_fourcc(*'MJPG')
result_video = cv.VideoWriter('Results/result_video.avi', fourcc, fps, (frame_width,frame_height))
```
The above code is where I create the video fourcc, indicate what file type to save the video as (.avi), as well as the save destination and file name. This setup is the same for all the video types listed above, and can all be configured to ones own desire.

## Videos
More demos and full videos

[Demo 1](https://drive.google.com/open?id=1Xb1DgXRRcnDjX-qZMRL6rDfTSYVFn6n3)\
[Demo 2](https://drive.google.com/open?id=15JjBhbicW6-9F0BUoH22E3jLql0n7wqS)      

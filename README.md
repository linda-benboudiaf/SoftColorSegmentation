# Soft Color Segmentation

This repository was developed as a final project for Computer Vision Master's Degree. 

### Prerequisits
* Python 3.9
* OpenCV
* Numpy
* scipy

### Get Started 
``` 
git clone https://github.com/lindaAcc/SoftColorSegmentation.git 
cd src
Open run.ipynb on GoogleColab or JupyterNotebook and run cells from top to bottom. 
```

### For trainning 
- Run ```train.py``` with arguments.
#### Arguments : 
- --run-name : this name will be used for output folder. 
- --batch-size : Input batch size for trainning, set to 32 by default.
- --epochs : number of epochs to train, set to 10 by default
- --num_primary_color : number of layer, set to 7 by default. 
- --csv_path : path to csv of images path. Exemple 'sample.csv' 


### Run example: 
[run.ipynb](src/run.ipynb) shows a test decomposition of an image test

you can can change image name and platte values on the third cell 

```
#img_name = 'palette1.jpeg'; manual_color_0 = [98, 12, 15]; manual_color_1 = [138, 206, 225]; manual_color_2 = [226, 179, 159]; manual_color_3 = [69, 173, 198]; manual_color_4 = [213, 215, 221]; manual_color_5 = [85,26,20]; manual_color_6 = [160,217,214]; 
```

```
img_name -> name of the image to test
manual_color_x -> RGB values given manually by the user, otherwise Kmeans algorithm give these values. 
```

- make sure to include images paths on the csv file **sample.csv** and matching color values on **palette_7_sample.csv** if you use 7 primary colors or **palette_6_sample.csv** if you 6 colors. 













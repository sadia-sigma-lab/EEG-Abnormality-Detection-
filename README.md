# EEG-Abnormality-Detection-


# EDF Viewer 
Full EDf Viewer Folder contains all files required for our customized online viewer to visualize edf files, annotatinng by drawing bounding boxes and visualizing predictions. This viewer is hosted at the following site <br/> 
https://usamah375.github.io/full-edf-viewer/ <br/>
Languages used for viewer  <br/>
Javascript <br/>
HTML5 <br/>
CSS <br/>
For guide about different features of viewer and how to use you can refer to the following link <br/>
https://docs.google.com/presentation/d/1eQK4VfLCHGVgSrfyeZNp1pB0fQ9kT1GdpKHD3ALYIX8/edit?usp=sharing

Requirements for deep learning based models used for abnormality detection in EEG signals are as follows <br/>




# Chrononet 
Python 3.8 <br/>
tensorflow 2.4.0 <br/>
mne 0.19.2 <br/>
1. datafolder paths for edf and annotated files are written in new_chorono.py <br/>
2. new_chorono.py file for loading data and training model <br/>
3. chrono_pred for loading trained model to predict and then storing predictions in csv format. <br/>



# Deep CNN
Python 3.8 <br/>
torch 1.8.1 <br/>
Braindecode 0.4.85 <br/>
mne 0.19.2 <br/>
1. Path for datafolder is written in config.py file <br/>
2. For training run diagnosis.py where functions from other files are being called <br/>
3. For predictions load the trained model by removing comment sign for model.load... in diagnosis.py and commenting exp.run() from the same file. <br/>





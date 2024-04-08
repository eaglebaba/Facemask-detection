# Face mask detection

## What It does
1. It detects if someone uses facemask or not
2. Takes pictures of people not putting on face mask and saves on file
3. Saves time stamp to database

## Features
1. Viewable on the web with the ability to run on any specified port
2. Can connect to remote cameras

## Dependencies

  * Tensorflow
  * pandas
  * numpy
  * flask
  * open-cv

  
Use [pip](https://pypi.python.org/pypi/pip) to install any missing dependencies (pip install --upgrade ... ) 

### Dependencies on Windows with python3
```
pip3 install pandas
pip3 install flask
pip3 install numpy
pip3 install tensorflow (or compiled)
pip3 install tqdm
pip3 install opencv-python
```

## Basic Usage
To run this project, simply clone this directory and run.
```
pip install -r requirements.txt
python app.py
```

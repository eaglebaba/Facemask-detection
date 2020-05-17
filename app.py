import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
from flask import Response
from flask import Flask
from flask import render_template
import threading
import argparse
import datetime
import imutils
import time
import cv2
import os
import numpy as np
import pyodbc

from datetime import datetime
from PIL import ImageGrab


ap = argparse.ArgumentParser()
ap.add_argument("-i", "--ip", type=str, default="1")
ap.add_argument("-o", "--port", type=int, default=8000)
ap.add_argument("-c", "--camera", type=str, default=0)
args = vars(ap.parse_args())

ip_addr = f"127.0.0.{args['ip']}"


# initialize the output frame and a lock used to ensure thread-safe
# exchanges of the output frames (useful for multiple browsers/tabs
# are viewing tthe stream)
outputFrame = None
lock = threading.Lock()

# initialize a flask object
app = Flask(__name__)

vs = cv2.VideoCapture(args['camera'])
time.sleep(2.0)

prototxtPath = os.path.sep.join(['./models/', "deploy.prototxt"])
weightsPath = os.path.sep.join(['./models/',
	"res10_300x300_ssd_iter_140000.caffemodel"])
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

time_ = datetime.now()
insert_data_sql =   """ INSERT INTO ImageStats (masked, date_updated) VALUES (?, ?)"""

conn = pyodbc.connect(Driver='{SQL Server};',
                      Server='DESKTOP-50CPBLJ\SQLEXPRESS;',
                      Database='facemask_db;',
                      Trusted_connection='yes;')

@app.route("/")
def index():
	return render_template("index.html")


def take_screenshot_and_save_to_db():
    global time_, insert_data_sql, conn
    new_time = datetime.now()
    if new_time.minute - time_.minute != 0:
        # take screenshot and save image
        im = ImageGrab.grab()
        im.save(f"./images/IMG_{ new_time.strftime('%Y-%m-%d_%H-%M-%S') }.jpg")
        # save timestamp to database
        to_insert = [0, new_time]
        cursor = conn.cursor()
        cursor.execute(insert_data_sql, to_insert)
        conn.commit()
        time_ = datetime.now()


def detect():
    global vs, outputFrame, lock, faceNet
    maskNet = load_model('./models/mask_detector_model.h5')

    def detect_and_predict_mask(frame, faceNet, maskNet):

        (h, w) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))

        # detect faces
        faceNet.setInput(blob)
        detections = faceNet.forward()

        # initialize list of faces, locations and predictions
        faces = []
        locs = []
        preds = []

        # loop over detections
        for i in range(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.5:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (start_x, start_y, end_x, end_y) = box.astype('int')

                # ensure the bounding boxes fall within the dimensions of the frame
                (start_x, start_y) = (max(0, start_x), max(0, start_y))
                (end_x, end_y) = (min(w - 1, end_x), min(h - 1, end_y))

                face = frame[start_y:end_y, start_x:end_x]
                face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                face = cv2.resize(face, (224, 224))
                face = img_to_array(face)
                face = preprocess_input(face)
                face = np.expand_dims(face, axis=0)

                faces.append(face)
                locs.append((start_x, start_y, end_x, end_y))

        # only make predictions if atleast one face was detected
        if len(faces) > 0:
            preds = maskNet.predict(faces)

        return (locs, preds)


    while True:
        # grab the frame from the video stream and resize it to a mx width of 400
        ret, frame = vs.read()
        frame = cv2.resize(frame, (500, 500))

        # detect faces, detect and predict mask, return location of face and
        # mask prediction
        (locs, preds) = detect_and_predict_mask(frame, faceNet, maskNet)

        for (box, pred) in zip(locs, preds):
            # unpack the bounding box and predictions
            (start_x, start_y, end_x, end_y) = box
            (mask, withoutMask) = pred

            # for any of the predictions, if without mask, take screenshot
            if withoutMask > mask:
                take_screenshot_and_save_to_db()

            label = "Mask" if mask > withoutMask else "No Mask"
            color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

            # include the probability in the label
            label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)

            # display the label and bounding box rectangle on the output frame
            cv2.putText(frame, label, (start_x, start_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
            cv2.rectangle(frame, (start_x, start_y), (end_x, end_y), color, 2)

        with lock:
            outputFrame = frame.copy()

		
def generate():
	# grab global references to the output frame and lock variables
    # used to feed the web app with the image
	global outputFrame, lock

	# loop over frames from the output stream
	while True:
		# wait until the lock is acquired
		with lock:
			# check if the output frame is available, otherwise skip
			# the iteration of the loop
			if outputFrame is None:
				continue

			# encode the frame in JPEG format
			(flag, encodedImage) = cv2.imencode(".jpg", outputFrame)

			# ensure the frame was successfully encoded
			if not flag:
				continue

		# yield the output frame in the byte format
		yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + 
			bytearray(encodedImage) + b'\r\n')

@app.route("/video_feed")
def video_feed():
	# return the response generated along with the specific media
	# type (mime type)
	return Response(generate(),
		mimetype = "multipart/x-mixed-replace; boundary=frame")


# check to see if this is the main thread of execution
if __name__ == '__main__':
    
    # start a thread to perform face mask detection
    t = threading.Thread(target=detect)
    t.daemon = True
    t.start()

    # start the flask app
    app.run(host=ip_addr, port=args['port'], debug=True,
            threaded=True, use_reloader=False)

# release the video stream pointer
vs.release()


### running: python app.py -i 1 -o 8000
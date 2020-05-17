"""todo:

send data,
connect to url"""

# it works
from PIL import ImageGrab

im = ImageGrab.grab()
im.show()

# for the camera ip:
# capture = cv2.VideoCapture('rtsp://username:password@192.168.1.64/1')
# capture = cv2.VideoCapture('rtsp://192.168.1.64/1')

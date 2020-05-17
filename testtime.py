from datetime import datetime
from PIL import ImageGrab
import sqlite3


masked_status = 1
entry_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

conn = sqlite3.connect("mask.db")

# try:
#     cxn = conn.cursor()
#     cxn.execute("INSERT into Person (masked, time) values (?, ?)", (masked_status, entry_time))
#     conn.commit()
# except err:
#     print(err)

rows = None
try:
    cxn = conn.cursor()
    cxn.execute("SELECT * FROM Person")
    rows.fetchAll()
except:
    print("sorry an error occurred")

print(rows)

"""
#### SOLVES THE SAVE IMAGE PROBLEM

time_ = datetime.now()

while True:
    new_time = datetime.now()
    if new_time.minute - time_.minute != 0:
        im = ImageGrab.grab()
        im.save(f"IMG_{ new_time.strftime('%Y-%m-%d_%H-%M-%S') }.jpg")
        print("saved new image")
        time_ = datetime.now()

"""
import time
import cv2
import json

config = json.load(open("config.json"))

if config["MonitorWindow"]:
    from windowcapture import WindowCapture
    video = WindowCapture(config["WindowPrefix"])
else:
    video = cv2.VideoCapture(0,cv2.CAP_DSHOW)
    video.set(cv2.CAP_PROP_FRAME_WIDTH,1920)
    video.set(cv2.CAP_PROP_FRAME_HEIGHT,1080)
    video.set(cv2.CAP_PROP_BUFFERSIZE,1)
roi_y = 0
roi_x = 0
roi_h = 10
roi_w = 10
UP = 2490368
DOWN = 2621440
LEFT = 2424832
RIGHT = 2555904
print("use up/down/left/right to move box, 8/4/6/2 to expand/shrink box, q to close")
while True:
    _,frame = video.read()
    time_counter = time.perf_counter()

    cv2.rectangle(frame,(roi_x,roi_y), (roi_x+roi_w,roi_y+roi_h), 255, 2)
    cv2.imshow("view", frame)
    keypress = cv2.waitKeyEx(1)
    if keypress == ord('q'):
        video.release()
        cv2.destroyAllWindows()
        exit()
    elif keypress == UP:
        roi_y -= 10
    elif keypress == DOWN:
        roi_y += 10
    elif keypress == LEFT:
        roi_x -= 10
    elif keypress == RIGHT:
        roi_x += 10
    elif keypress == ord('8'):
        roi_h -= 10
    elif keypress == ord('2'):
        roi_h += 10
    elif keypress == ord('4'):
        roi_w -= 10
    elif keypress == ord('6'):
        roi_w += 10
    if keypress != ord('q') and keypress != -1:
        print([roi_x,roi_y,roi_w,roi_h])
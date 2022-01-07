import cv2
import heapq
import json
import os.path
import signal
import sys
import threading
import time
import tkinter as tk
import tkinter.filedialog as fd
import rngtool
from tkinter import ttk
from os import listdir
from os.path import isfile, join
from PIL import Image, ImageTk
from xorshift import Xorshift

class Application(tk.Frame):
    def __init__(self, master=None):
        super().__init__(master)
        self.master = master
        self.previewing = False
        self.monitoring = False
        self.reidentifying = False
        self.timelining = False
        self.config_json = {}
        self.default_config = {
            "MonitorWindow": True,
            "WindowPrefix": "SysDVR-Client [PID ",
            "image": "./images/cave/eye.png",
            "view": [0, 0, 0, 0],
            "thresh": 0.9,
            "white_delay": 0.0,
            "advance_delay": 0,
            "crop": [0,0,0,0],
            "camera": 0
        }
        self.pack()
        self.create_widgets()
        signal.signal(signal.SIGINT, self.signal_handler)

    def update_configs(self,event=None):
        self.config_jsons =  [f for f in listdir("configs") if isfile(join("configs", f))]
        self.config_combobox['values'] = self.config_jsons

    def create_widgets(self):
        self.master.title("Player Blink")

        self.progress = ttk.Label(self,text="0/0")
        self.progress.grid(column=0,row=0)

        self.config_combobox = ttk.Combobox(self, state="readonly", values=[])
        self.config_combobox.grid(column=1,row=0)
        self.config_combobox.bind("<<ComboboxSelected>>", self.config_combobox_onchange)
        self.config_combobox.bind("<Button-1>", self.update_configs)
        self.update_configs()

        self.new_config_button = ttk.Button(self,text="+",command=self.new_config,width=2)
        self.new_config_button.grid(column=2,row=0)

        self.eye_display = ttk.Label(self)
        self.eye_display.grid(column=1,row=1)

        self.prefix_input = ttk.Entry(self)
        self.prefix_input.grid(column=1,row=2)

        self.camera_index = tk.Spinbox(self, from_= 0, to = 99, width = 5)
        self.camera_index.grid(column=2,row=1)

        self.monitor_window_var = tk.IntVar()
        self.monitor_window = ttk.Checkbutton(self,text="Monitor Window",variable=self.monitor_window_var)
        self.monitor_window.grid(column=2,row=2)

        self.monitor_display_buffer = ttk.Label(self)
        self.monitor_display_buffer.grid(column=1,row=3,rowspan=64,columnspan=2)
        self.monitor_display = ttk.Label(self)
        self.monitor_display.grid(column=1,row=3,rowspan=64,columnspan=2)

        self.monitor_blink_button = ttk.Button(self, text="Monitor Blinks", command=self.monitor_blinks)
        self.monitor_blink_button.grid(column=3,row=0)

        self.reidentify_button = ttk.Button(self, text="Reidentify", command=self.reidentify)
        self.reidentify_button.grid(column=3,row=1)

        self.preview_button = ttk.Button(self, text="Preview", command=self.preview)
        self.preview_button.grid(column=3,row=2)

        self.stop_tracking_button = ttk.Button(self, text="Stop Tracking", command=self.stop_tracking)
        self.stop_tracking_button.grid(column=3,row=3)

        self.legendary_timeline_button = ttk.Button(self, text="Legendary Timeline", command=self.legendary_timeline)
        self.legendary_timeline_button.grid(column=3,row=4)

        x = y = w = h = 0
        th = 0.9

        ttk.Label(self,text="X").grid(column=4,row=1)
        ttk.Label(self,text="Y").grid(column=4,row=2)
        ttk.Label(self,text="W").grid(column=4,row=3)
        ttk.Label(self,text="H").grid(column=4,row=4)
        ttk.Label(self,text="Th").grid(column=4,row=5)
        ttk.Label(self,text="whi del").grid(column=4,row=6)
        ttk.Label(self,text="adv del").grid(column=4,row=7)

        self.pos_x = tk.Spinbox(self, from_= 0, to = 99999, width = 5)
        self.pos_x.grid(column=5,row=1)
        self.pos_y = tk.Spinbox(self, from_= 0, to = 99999, width = 5)
        self.pos_y.grid(column=5,row=2)
        self.pos_w = tk.Spinbox(self, from_= 0, to = 99999, width = 5)
        self.pos_w.grid(column=5,row=3)
        self.pos_h = tk.Spinbox(self, from_= 0, to = 99999, width = 5)
        self.pos_h.grid(column=5,row=4)
        self.pos_th = tk.Spinbox(self, from_= 0, to = 1, width = 5, increment=0.1)
        self.pos_th.grid(column=5,row=5)
        self.whi_del = tk.Spinbox(self, from_= 0, to = 5, width = 5, increment=0.1)
        self.whi_del.grid(column=5,row=6)
        self.adv_del = tk.Spinbox(self, from_= 0, to = 5, width = 5, increment=1)
        self.adv_del.grid(column=5,row=7)

        self.save_button = ttk.Button(self, text="Select Eye",command=self.new_eye)
        self.save_button.grid(column=4,row=8,columnspan=2)

        self.new_eye_button = ttk.Button(self, text="Save Config",command=self.save_config)
        self.new_eye_button.grid(column=4,row=9,columnspan=2)

        self.s0_1_2_3 = tk.Text(self, width=10, height=4)
        self.s0_1_2_3.grid(column=0,row=2,rowspan=4)
        
        self.s01_23 = tk.Text(self, width=20, height=2)
        self.s01_23.grid(column=0,row=6,rowspan=4)

        self.advances = 0
        self.adv = ttk.Label(self,text=self.advances)
        self.adv.grid(column=0,row=10)

        self.count_down = 0
        self.cd = ttk.Label(self,text=self.count_down)
        self.cd.grid(column=0,row=11)

        self.pos_x.delete(0, tk.END)
        self.pos_x.insert(0, x)
        self.pos_y.delete(0, tk.END)
        self.pos_y.insert(0, y)
        self.pos_w.delete(0, tk.END)
        self.pos_w.insert(0, w)
        self.pos_h.delete(0, tk.END)
        self.pos_h.insert(0, h)
        self.pos_th.delete(0, tk.END)
        self.pos_th.insert(0, th)
        self.whi_del.delete(0, tk.END)
        self.whi_del.insert(0, 0.0)
        self.adv_del.delete(0, tk.END)
        self.adv_del.insert(0, 0)
        self.camera_index.delete(0, tk.END)
        self.camera_index.insert(0, 0)

        self.after_task()
    
    def new_config(self):
        with fd.asksaveasfile(initialdir="./configs/", filetypes=[("JSON", ".json")]) as f:
            json.dump(self.default_config,f,indent=4)
            self.config_combobox.set(os.path.basename(f.name))
        self.config_combobox_onchange()
    
    def new_eye(self):
        self.config_json["image"] = "./"+os.path.relpath(fd.askopenfilename(initialdir="./images/", filetypes=[("Image", ".png")])).replace("\\","/")
        self.player_eye = cv2.imread(self.config_json["image"], cv2.IMREAD_GRAYSCALE)
        self.player_eye_tk = self.cv_image_to_tk(self.player_eye)
        self.eye_display['image'] = self.player_eye_tk

    def save_config(self):
        json.dump(self.config_json,open(join("configs",self.config_combobox.get()),"w"),indent=4)

    def cv_image_to_tk(self, image):
        split = cv2.split(image)
        if len(split) == 3:
            b,g,r = split
            image = cv2.merge((r,g,b))
        im = Image.fromarray(image)
        return ImageTk.PhotoImage(image=im) 

    def config_combobox_onchange(self, event=None):
        self.config_json = json.load(open(join("configs",self.config_combobox.get())))
        x,y,w,h = self.config_json["view"]
        self.pos_x.delete(0, tk.END)
        self.pos_x.insert(0, x)
        self.pos_y.delete(0, tk.END)
        self.pos_y.insert(0, y)
        self.pos_w.delete(0, tk.END)
        self.pos_w.insert(0, w)
        self.pos_h.delete(0, tk.END)
        self.pos_h.insert(0, h)
        self.pos_th.delete(0, tk.END)
        self.pos_th.insert(0, self.config_json["thresh"])
        self.whi_del.delete(0, tk.END)
        self.whi_del.insert(0, self.config_json["white_delay"])
        self.adv_del.delete(0, tk.END)
        self.adv_del.insert(0, self.config_json["advance_delay"])
        self.camera_index.delete(0, tk.END)
        self.camera_index.insert(0, self.config_json["camera"])
        self.player_eye = cv2.imread(self.config_json["image"], cv2.IMREAD_GRAYSCALE)
        self.player_eye_tk = self.cv_image_to_tk(self.player_eye)
        self.eye_display['image'] = self.player_eye_tk
        self.prefix_input.delete(0, tk.END)
        self.prefix_input.insert(0, self.config_json["WindowPrefix"])
        self.monitor_window_var.set(self.config_json["MonitorWindow"])

    def stop_tracking(self):
        self.tracking = False

    def legendary_timeline(self):
        self.timelining = True

    def monitor_blinks(self):
        if not self.monitoring:
            self.monitor_blink_button['text'] = "Stop Monitoring"
            self.monitoring = True
            self.monitoring_thread=threading.Thread(target=self.monitoring_work)
            self.monitoring_thread.daemon = True
            self.monitoring_thread.start()
        else:
            self.monitor_blink_button['text'] = "Monitor Blinks"
            self.monitoring = False

    def reidentify(self):
        if not self.reidentifying:
            self.reidentify_button['text'] = "Stop Reidentifying"
            self.reidentifying = True
            self.reidentifying_thread=threading.Thread(target=self.reidentifying_work)
            self.reidentifying_thread.daemon = True
            self.reidentifying_thread.start()
        else:
            self.reidentify_button['text'] = "Reidentify"
            self.reidentifying = False
    
    def monitoring_work(self):
        self.tracking = False
        blinks, intervals, offset_time = rngtool.tracking_blink(self.player_eye, *self.config_json["view"], MonitorWindow=self.config_json["MonitorWindow"], WindowPrefix=self.config_json["WindowPrefix"], crop=self.config_json["crop"], camera=self.config_json["camera"], tk_window=self, th=self.config_json["thresh"])
        prng = rngtool.recov(blinks, intervals)

        self.monitor_blink_button['text'] = "Monitor Blinks"
        self.monitoring = False
        self.preview()

        waituntil = time.perf_counter()
        diff = round(waituntil-offset_time)
        prng.getNextRandSequence(diff)

        state = prng.getState()
        s0 = f"{state[0]:08X}"
        s1 = f"{state[1]:08X}"
        s2 = f"{state[2]:08X}"
        s3 = f"{state[3]:08X}"

        s01 = s0+s1
        s23 = s2+s3

        print(s01,s23)
        print(s0,s1,s2,s3)
        self.s0_1_2_3.delete(1.0, tk.END)
        self.s01_23.delete(1.0, tk.END)

        self.s0_1_2_3.insert(1.0,s0+"\n"+s1+"\n"+s2+"\n"+s3)
        self.s01_23.insert(1.0,s01+"\n"+s23)

        self.advances = 0
        self.tracking = True
        self.count_down = None
        while self.tracking:
            if self.count_down is None:
                if self.timelining:
                    self.count_down = 10
            elif self.count_down != 0:
                self.count_down -= 1
                print(self.count_down+1)
            else:
                break
            
            self.advances += 1
            r = prng.next()
            waituntil += 1.018

            print(f"advances:{self.advances}, blinks:{hex(r&0xF)}")        
            
            next_time = waituntil - time.perf_counter() or 0
            time.sleep(next_time)
        if self.timelining:
            prng.next()
            # white screen
            time.sleep(self.config_json["white_delay"])
            waituntil = time.perf_counter()
            prng.advance(self.config_json["advance_delay"])
            self.advances += self.config_json["advance_delay"]
            print("entered the stationary symbol room")
            queue = []
            heapq.heappush(queue, (waituntil+1.017,0))

            blink_int = prng.rangefloat(100.0, 370.0)/30 - 0.048

            heapq.heappush(queue, (waituntil+blink_int,1))
            while queue and self.tracking:
                self.advances += 1
                w, q = heapq.heappop(queue)
                next_time = w - time.perf_counter() or 0
                if next_time>0:
                    time.sleep(next_time)

                if q==0:
                    r = prng.next()
                    print(f"advances:{self.advances}, blink:{hex(r&0xF)}")
                    heapq.heappush(queue, (w+1.017, 0))
                else:
                    blink_int = prng.rangefloat(100.0, 370.0)/30 - 0.048

                    heapq.heappush(queue, (w+blink_int, 1))
                    print(f"advances:{self.advances}, interval:{blink_int}")
            self.timelining = False

    def reidentifying_work(self):
        self.tracking = False
        state = [int(x,16) for x in self.s0_1_2_3.get(1.0,tk.END).split("\n")[:4]]

        s0 = f"{state[0]:08X}"
        s1 = f"{state[1]:08X}"
        s2 = f"{state[2]:08X}"
        s3 = f"{state[3]:08X}"

        s01 = s0+s1
        s23 = s2+s3

        print(s01,s23)
        print(s0,s1,s2,s3)
        self.s0_1_2_3.delete(1.0, tk.END)
        self.s01_23.delete(1.0, tk.END)

        self.s0_1_2_3.insert(1.0,s0+"\n"+s1+"\n"+s2+"\n"+s3)
        self.s01_23.insert(1.0,s01+"\n"+s23)

        print([hex(x) for x in state])
        observed_blinks, _, offset_time = rngtool.tracking_blink(self.player_eye, *self.config_json["view"], MonitorWindow=self.config_json["MonitorWindow"], WindowPrefix=self.config_json["WindowPrefix"], crop=self.config_json["crop"], camera=self.config_json["camera"], tk_window=self, th=self.config_json["thresh"], size=20)
        reidentified_rng, adv = rngtool.reidentifyByBlinks(Xorshift(*state), observed_blinks, return_advance=True)


        self.reidentify_button['text'] = "Reidentify"
        self.reidentifying = False
        self.preview()

        waituntil = time.perf_counter()
        diff = round(waituntil-offset_time)+1
        reidentified_rng.getNextRandSequence(diff)
        state = reidentified_rng.getState()

        self.advances = adv+diff
        self.tracking = True
        self.count_down = None
        while self.tracking:
            if self.count_down is None:
                if self.timelining:
                    self.count_down = 10
            elif self.count_down != 0:
                self.count_down -= 1
                print(self.count_down+1)
            else:
                break
            
            self.advances += 1
            r = reidentified_rng.next()
            waituntil += 1.018

            print(f"advances:{self.advances}, blinks:{hex(r&0xF)}")        
            
            next_time = waituntil - time.perf_counter() or 0
            time.sleep(next_time)
        if self.timelining:
            reidentified_rng.next()
            # white screen
            time.sleep(self.config_json["white_delay"])
            waituntil = time.perf_counter()
            reidentified_rng.advance(self.config_json["advance_delay"])
            self.advances += self.config_json["advance_delay"]
            print("entered the stationary symbol room")
            queue = []
            heapq.heappush(queue, (waituntil+1.017,0))

            blink_int = reidentified_rng.rangefloat(100.0, 370.0)/30 - 0.048

            heapq.heappush(queue, (waituntil+blink_int,1))
            while queue and self.tracking:
                self.advances += 1
                w, q = heapq.heappop(queue)
                next_time = w - time.perf_counter() or 0
                if next_time>0:
                    time.sleep(next_time)

                if q==0:
                    r = reidentified_rng.next()
                    print(f"advances:{self.advances}, blink:{hex(r&0xF)}")
                    heapq.heappush(queue, (w+1.017, 0))
                else:
                    blink_int = reidentified_rng.rangefloat(100.0, 370.0)/30 - 0.048

                    heapq.heappush(queue, (w+blink_int, 1))
                    print(f"advances:{self.advances}, interval:{blink_int}")
            self.timelining = False

    def preview(self):
        if not self.previewing:
            self.preview_button['text'] = "Stop Preview"
            self.previewing = True
            self.previewing_thread=threading.Thread(target=self.previewing_work)
            self.previewing_thread.daemon = True
            self.previewing_thread.start()
        else:
            self.preview_button['text'] = "Preview"
            self.previewing = False
    
    def previewing_work(self):
        last_frame_tk = None

        if self.config_json["MonitorWindow"]:
            from windowcapture import WindowCapture
            video = WindowCapture(self.config_json["WindowPrefix"],self.config_json["crop"])
        else:
            if sys.platform.startswith('linux'): # all Linux
                backend = cv2.CAP_V4L
            elif sys.platform.startswith('win'): # MS Windows
                backend = cv2.CAP_DSHOW
            elif sys.platform.startswith('darwin'): # macOS
                backend = cv2.CAP_QT
            else:
                backend = cv2.CAP_ANY # auto-detect via OpenCV
            video = cv2.VideoCapture(self.config_json["camera"],backend)
            video.set(cv2.CAP_PROP_FRAME_WIDTH,1920)
            video.set(cv2.CAP_PROP_FRAME_HEIGHT,1080)
            video.set(cv2.CAP_PROP_BUFFERSIZE,1)
            print(f"camera {self.config_json['camera']}")


        while self.previewing:
            eye = self.player_eye
            w, h = eye.shape[::-1]
            roi_x, roi_y, roi_w, roi_h = self.config_json["view"]
            _, frame = video.read()
            if not self.config_json["MonitorWindow"]:
                frame = cv2.resize(frame,(960,540))
            roi = cv2.cvtColor(frame[roi_y:roi_y+roi_h,roi_x:roi_x+roi_w],cv2.COLOR_RGB2GRAY)
            res = cv2.matchTemplate(roi,eye,cv2.TM_CCOEFF_NORMED)
            _, match, _, max_loc = cv2.minMaxLoc(res)

            cv2.rectangle(frame,(roi_x,roi_y), (roi_x+roi_w,roi_y+roi_h), (0,0,255), 2)
            if 0.01<match<self.config_json["thresh"]:
                cv2.rectangle(frame,(roi_x,roi_y), (roi_x+roi_w,roi_y+roi_h), 255, 2)
            else:
                max_loc = (max_loc[0] + roi_x,max_loc[1] + roi_y)
                bottom_right = (max_loc[0] + w, max_loc[1] + h)
                cv2.rectangle(frame,max_loc, bottom_right, 255, 2)
            frame_tk = self.cv_image_to_tk(frame)
            self.monitor_tk_buffer = last_frame_tk
            self.monitor_display_buffer['image'] = self.monitor_tk_buffer
            self.monitor_tk = frame_tk
            self.monitor_display['image'] = self.monitor_tk
            last_frame_tk = frame_tk
        self.monitor_tk_buffer = None
        self.monitor_tk = None

    def after_task(self):
        self.config_json["view"] = [int(self.pos_x.get()),int(self.pos_y.get()),int(self.pos_w.get()),int(self.pos_h.get())]
        self.config_json["thresh"] = float(self.pos_th.get())
        self.config_json["WindowPrefix"] = self.prefix_input.get()
        self.config_json["white_delay"] = float(self.whi_del.get())
        self.config_json["advance_delay"] = int(self.adv_del.get())
        self.config_json["MonitorWindow"] = bool(self.monitor_window_var.get())
        self.config_json["camera"] = int(self.camera_index.get())
        self.adv['text'] = self.advances
        self.cd['text'] = self.count_down
        self.after(100,self.after_task)

    def signal_handler(self, signal, frame):
        sys.exit(0)

root = tk.Tk()
app = Application(master=root)
app.mainloop()
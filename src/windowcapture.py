import numpy as np
try:
    import win32gui, win32ui, win32con
except ImportError:
    raise Exception("Could not import win32, if you are not on windows Monitor Window must be unchecked, otherwise make sure packages are installed correctly.")

# Class to monitor a window
class WindowCapture:
    def __init__(self, partial_window_title, crop):
        # set up variables
        self.w = 0
        self.h = 0
        self.hwnd = None
        self.cropped_x = 0
        self.cropped_y = 0
        self.offset_x = 0
        self.offset_y = 0

        # a string contained in the window title, used to find windows who's name is not constant (pid of MonitorWindow changes)
        self.partial_window_title = partial_window_title
        # find the handle for the window we want to capture
        hwnds = []
        win32gui.EnumWindows(self.winEnumHandler, hwnds)
        if len(hwnds) == 0:
            raise Exception('Window not found')
        self.hwnd = hwnds[0]

        # get the window size
        window_rect = win32gui.GetWindowRect(self.hwnd)
        self.w = window_rect[2] - window_rect[0]
        self.h = window_rect[3] - window_rect[1]

        # account for the window border and titlebar and cut them off
        border_pixels = 8
        titlebar_pixels = 31
        self.w = self.w - (border_pixels * 2)
        self.h = self.h - titlebar_pixels - border_pixels
        self.cropped_x = border_pixels
        self.cropped_y = titlebar_pixels
        if crop is not None and crop != [0,0,0,0]:
            self.cropped_x,self.cropped_y,self.w,self.h = crop

        # set the cropped coordinates offset so we can translate screenshot images into actual screen positions
        self.offset_x = window_rect[0] + self.cropped_x
        self.offset_y = window_rect[1] + self.cropped_y

    # handler for finding the target window
    def winEnumHandler(self, hwnd, ctx):
        # check if window is not minimized
        if win32gui.IsWindowVisible(hwnd):
            # check if our partial title is contained in the actual title
            if self.partial_window_title in win32gui.GetWindowText(hwnd):
                # add to list
                ctx.append(hwnd)

    # used to send a screenshot of the window to cv2
    def read(self):
        # get the window image data
        wDC = win32gui.GetWindowDC(self.hwnd)
        dcObj = win32ui.CreateDCFromHandle(wDC)
        cDC = dcObj.CreateCompatibleDC()
        dataBitMap = win32ui.CreateBitmap()
        try:
            dataBitMap.CreateCompatibleBitmap(dcObj, self.w, self.h)
        except win32ui.error:
            raise Exception("Failed to pull input from target window, make sure nothing else is accessing the window (stop preview before monitoring blinks).")
        cDC.SelectObject(dataBitMap)
        cDC.BitBlt((0, 0), (self.w, self.h), dcObj, (self.cropped_x, self.cropped_y), win32con.SRCCOPY)

        # convert the raw data into a format opencv can read
        signedIntsArray = dataBitMap.GetBitmapBits(True)
        img = np.fromstring(signedIntsArray, dtype='uint8')
        img.shape = (self.h, self.w, 4)

        # free resources
        dcObj.DeleteDC()
        cDC.DeleteDC()
        win32gui.ReleaseDC(self.hwnd, wDC)
        win32gui.DeleteObject(dataBitMap.GetHandle())

        # drop the alpha channel, to avoid throwing an error
        img = img[...,:3]

        # make image C_CONTIGUOUS to avoid errors
        # https://github.com/opencv/opencv/issues/14866#issuecomment-580207109
        img = np.ascontiguousarray(img)

        return True,img

    # translate a pixel position on a screenshot image to a pixel position on the screen
    def get_screen_position(self, pos):
        return (pos[0] + self.offset_x, pos[1] + self.offset_y)
    
    def release(self):
        pass
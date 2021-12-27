import rngtool
import cv2
import time
import json

config = json.load(open("config.json"))

def expr():
    player_eye = cv2.imread(config["image"], cv2.IMREAD_GRAYSCALE)
    if player_eye is None:
        print("path is wrong")
        return
    blinks, intervals, offset_time = rngtool.tracking_blink_manual()
    prng = rngtool.recov(blinks, intervals)

    waituntil = time.perf_counter()
    diff = round(waituntil-offset_time)
    prng.getNextRandSequence(diff)

    state = prng.getState()
    print(hex(state[0]<<32|state[1]), hex(state[2]<<32|state[3]))

    advances = 0

    for _ in range(1000):
        advances += 1
        r = prng.next()
        waituntil += 1.018

        print(f"advances:{advances}, blinks:{hex(r&0xF)}")        
        
        next_time = waituntil - time.perf_counter() or 0
        time.sleep(next_time)


if __name__ == "__main__":
    expr()
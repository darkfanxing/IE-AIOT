import threading
import time

stop_screenshot = False

def screenshot(user_id, stop):
  timeframe = 0
  while True:
    grabImgByFrame(timeframe, user_id)
    time.sleep(1)
    if stop():
      break

def main():
    global stop_screenshot

    tmp = threading.Thread(target=screenshot, args=(id, lambda: stop_screenshot))
    tmp.start()   
    time.sleep(3)
    stop_screenshot = True
    print('main: done sleeping; time to stop the threads.')
    
    print('Finis.')

main()
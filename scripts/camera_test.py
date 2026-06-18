import cv2

# DSHOW gives more reliable property control than MSMF for UVC webcams on Windows
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# Order matters: codec -> resolution -> fps -> manual controls
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)   # 720p is plenty for MediaPipe; lower latency than 4K
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_FPS, 30)

# Manual exposure: DSHOW uses 0.25 = manual, 0.75 = auto (yes, really)
cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
cap.set(cv2.CAP_PROP_EXPOSURE, -6)         # tune to your room
cap.set(cv2.CAP_PROP_AUTO_WB, 0)
cap.set(cv2.CAP_PROP_WB_TEMPERATURE, 4600) # tune to your lighting

for name, prop in [
    ("width",  cv2.CAP_PROP_FRAME_WIDTH),
    ("height", cv2.CAP_PROP_FRAME_HEIGHT),
    ("fps",    cv2.CAP_PROP_FPS),
    ("auto_exp", cv2.CAP_PROP_AUTO_EXPOSURE),
    ("exposure", cv2.CAP_PROP_EXPOSURE),
    ("auto_wb",  cv2.CAP_PROP_AUTO_WB),
    ("wb_temp",  cv2.CAP_PROP_WB_TEMPERATURE),
    ("focus",    cv2.CAP_PROP_FOCUS),
]:
    print(f"{name:10s}: {cap.get(prop)}")
import os
import cv2
from ultralytics import YOLO


os.chdir(r"F:\YOLO Projects\IslamicPray")

model = YOLO("IslamicBest.pt")

TheClass = model.model.names

def draw_text_with_background(frame, text, position, font, scale, text_color, background_color, border_color, thickness=2, padding=5):
    """Draw text with background and border on the frame."""
    (text_width, text_height), baseline = cv2.getTextSize(text, font, scale, thickness)
    x, y = position
    # Background rectangle
    cv2.rectangle(frame, 
                  (x - padding, y - text_height - padding), 
                  (x + text_width + padding, y + baseline + padding), 
                  background_color, 
                  cv2.FILLED)
    # Border rectangle
    cv2.rectangle(frame, 
                  (x - padding, y - text_height - padding), 
                  (x + text_width + padding, y + baseline + padding), 
                  border_color, 
                  thickness)
    # Text
    cv2.putText(frame, text, (x, y), font, scale, text_color, thickness, lineType=cv2.LINE_AA)


cap = cv2.VideoCapture("istockphoto-1345393460-640_adpp_is.mp4")

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("The frames have been finished")
        break
    else:
        frame = cv2.resize(frame, (1100, 700))
        results = model.predict(frame, conf=0.5)
        for result in results:
            boxes = result.boxes.xyxy
            classes = result.boxes.cls
            confs = result.boxes.conf
            for box, cls, conf in zip(boxes, classes, confs):
                x, y, w, h = box
                x, y, w, h = int(x), int(y), int(w), int(h)
                cls = int(cls)
                cv2.rectangle(frame, (x, y), (w, h), [0, 255, 0], 2)
                draw_text_with_background(frame, 
                                      f"{TheClass[cls].capitalize()}, Conf:{(conf)*100:0.2f}%", 
                                      (x, y + 20), 
                                      cv2.FONT_HERSHEY_COMPLEX, 
                                      0.6, 
                                      (255, 255, 255),  # White text
                                      (0, 0, 0),  # Black background
                                      (0, 0, 255))  # Red border
        cv2.imshow("IslamicPray", frame)
        if cv2.waitKey(1) == 27:
            break
cap.release()
cv2.destroyAllWindows()
                
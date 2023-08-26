import torch
from ultralytics import YOLO
import cv2
import easyocr
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from deep_sort_realtime.deepsort_tracker import DeepSort

Video_FILE = '/home/ict4025/content/dataset/testimg/testvideo.mp4'
model = torch.hub.load('ultralytics/yolov5', 'custom', '/home/ict4025/content/yolov5/runs/train/test_yolov5s_evplate_epoch1002/weights/best.pt')
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)
output_name = 'output9.mp4'
video_size = (1280, 720)

reader = easyocr.Reader(['ko'])

def score_frame(frame):
    frame = [frame]
    results = model(frame)
    results = results.pred[0]
    labels = results[:, -1]
    cord = results[:, :4]
    conf = results[:, 4]
    return labels, cord, conf

def plot_boxes(results, frame):
    detections = []
    labels, cord, conf = results
    n = len(labels)
    for i in range(n):
        row = cord[i]
        if conf[i] >= 0.2:
            x1, y1, x2, y2 = int(row[0]), int(row[1]), int(row[2]), int(row[3])
            if labels[i] == 0:
                cv2.rectangle(
                    img=frame,
                    pt1=(x1, y1),
                    pt2=(x2, y2),
                    color=(0, 0, 255),
                    thickness=2
                )
                cv2.putText(
                    img=frame,
                    text="Car",
                    org=(x1, y1 - 10),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.9,
                    color=(0, 255, 0),
                    thickness=2
                )
            elif labels[i] in [1, 2]:
                try:
                    if labels[i - 1] == 0:
                        plate_region = frame[y1:y2, x1:x2]
                        plate_image = Image.fromarray(cv2.cvtColor(plate_region, cv2.COLOR_BGR2RGB))
                        plate_array = np.array(plate_image)
                        plate_number = reader.readtext(plate_array)
                        concat_number = ' '.join([number[1] for number in plate_number])
                        number_conf = np.mean([number[2] for number in plate_number])
                        detections.append(([x1, y1, int(x2 - x1), int(y2 - y1)], conf[i].item(), concat_number))
                        cv2.rectangle(
                            img=frame,
                            pt1=(x1, y1),
                            pt2=(x2, y2),
                            color=(0, 0, 255),
                            thickness=2
                        )
                        cv2.putText(
                            img=frame,
                            text="electric" if labels[i] == 1 else "Illegal",
                            org=(x1, y1 - 10),
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale=0.9,
                            color=(0, 255, 0),
                            thickness=2
                        )
                        cv2.putText(
                            img=frame,
                            text=concat_number,
                            org=(x1, y2 + 25),
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale=0.8,
                            color=(0, 255, 0),
                            thickness=2
                        )
                except IndexError:
                    pass
    return detections

def main():
    cap = cv2.VideoCapture(Video_FILE)
    if not cap.isOpened():
        print("Error opening video file")
        return
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    codec = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_name, codec, fps, video_size)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv2.resize(frame, video_size)
        results = score_frame(frame)
        detections = plot_boxes(results, frame)
        
        for detection in detections:
            box, confidence, number = detection
            print(f"Detected plate: {number}, Confidence: {confidence:.2f}")
        
        out.write(frame)
        
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()

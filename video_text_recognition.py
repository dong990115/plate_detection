import  torch
from    ultralytics import YOLO
import  cv2
import  easyocr
from PIL import Image
import  numpy                               as np
import  matplotlib.pyplot                   as plt
from    deep_sort_realtime.deepsort_tracker import DeepSort


Video_FILE  = '/home/ict4025/content/dataset/testimg/testvideo2.mp4'                                                                 # With ultralytics, everytime you train the model
# model       = YOLO(r"/home/ict4025/content/yolov5/runs/train/test_yolov5s_evplate_epoch1002/weights/best.pt")      # Load YOLO model we trained in the previous demo video~      
model = torch.hub.load('ultralytics/yolov5', 'custom', path='/home/ict4025/content/yolov5/runs/train/test_yolov5s_evplate/weights/best.pt')
device      = 'cuda' if torch.cuda.is_available() else 'cpu'                                #    the related information will be saved in the runs/detect file
model.to(device)                                                                            #    that's why I think it's a wonderful tool~ let me show you the results
output_name = 'output9.mp4'                                                                 # Because the source video resolution is 2560x1440
video_size  = (1280,720)                                               


reader      = easyocr.Reader(['ko'])

def score_frame(frame):                                                                     # score_frame is the same as the previous                            
    frame   = [frame]                                                                       
    results = model(frame)                                                                  
    # results = results[0].boxes.
    # labels  = results.cls                                                                  
    # cord    = results.xyxyn
    # conf    = results.conf
    results = results.pred[0]  # Detections의 pred 속성을 사용하여 결과에 접근
    labels  = results[:, -1]  # 마지막 열은 클래스 레이블
    cord    = results[:, :4]   # 앞부분은 좌표 정보
    conf    = results[:, 4]    # 마지막 열은 신뢰도
    return labels, cord, conf                                               


def plot_boxes(results, frame):
    detections = []
    # global id
    labels, cord, conf = results
    n = len(labels)
    for i in range(n):
        row = cord[i]
      #  cls_number = int(labels[i])
      #  print(f'Class number: {cls_number}')
        if conf[i] >= 0.4:
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

            elif labels[i] == 1:  # 수정된 부분: 클래스 1번에 대한 처리
                try:
                    # 클래스 1번 바운딩 박스가 클래스 0번 바운딩 박스 안에 있는지 확인
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
                            color=(255, 0, 0),
                            thickness=2
                        )
                        cv2.putText(
                            img=frame,
                            text="electric",
                            org=(x1, y1 - 10),
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale=0.9,
                            color=(255, 0, 0),
                            thickness=2
                        )
                except:
                    pass        

            elif labels[i] == 2:
                try:
                    # 클래스 2번 바운딩 박스가 클래스 0번 바운딩 박스 안에 있는지 확인
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
                            text="Illegal",
                            org=(x1, y1 - 10),
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale=0.9,
                            color=(0, 255, 0),
                            thickness=2
                        )
                        cv2.putText(
                            img=frame,
                            text=concat_number,
                            org=(x1, y1 - 35),
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale=0.9,
                            color=(0, 255, 0),
                            thickness=2
                        )
                        print(f'Plate Numbers: {concat_number}')
                except:
                    pass
    return frame, detections




object_tracker = DeepSort(                                                                  # Here is the key function/method with the object tracking technique~~~!!
    max_iou_distance        = 0.7,                                                          # I think there are some parameter you should pay attention:
    max_age                 = 30,                                                           #  Maximum number of missed misses before a track is deleted. 
    n_init                  = 3,                                                            #  Number of frames that a track remains in initialization phase
    nms_max_overlap         = 1,
    max_cosine_distance     = 0.2,
    nn_budget               = None,
    gating_only_position    = False,
    override_track_class    = None,
    embedder                = "mobilenet",
    half                    = True,
    bgr                     = True,
    embedder_gpu            = True,
    embedder_model_name     = None,
    embedder_wts            = None,
    polygon                 = False,
    today                   = None
)


def get_frames(filename):                                                                   # fourcc = cv2.VideoWriter_fourcc(*'MP4V') 
    global fourcc, video_out, fps                                                           # Save Video → Using cv2.VideoWriter()    ;   Save Image → Using cv2.imwriter()
    fps         = 60                                                                        # UP2U
    fourcc      = cv2.VideoWriter_fourcc(*'mp4v')                                           # RECOMMENDED                           
    video_out   = cv2.VideoWriter(output_name, fourcc, fps, video_size)                     # So, you need to use the correct video size to get our frame here!!!!
    video       = cv2.VideoCapture(filename)                            
    while video.isOpened():
        rete,frame=video.read()
        if rete:
            yield frame
        else:
            video.release()
            yield None

# main : the following part with get_frames() will output the final new video.
counter = 0
id      = 0
for frame in get_frames(Video_FILE):                                    # f = get_frames(Video_FILE) ; frame = next(f)
    print(counter)                                                      # cv2.imshow("img", frame) ; cv2.waitKey(0) ; plt.show()
    if frame is None:                                                   # cv2.putText(frame, f'FPS: {fps}', (20,70), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,255,0), 2)
        break
    results         = score_frame(frame)
    img, detections = plot_boxes(results, frame)
    tracks          = object_tracker.update_tracks(detections, frame=img)       # !!!
    for track in tracks:                                                # track = tracks[0]
        if not track.is_confirmed():
            continue
        track_id    = track.track_id
        ltrb        = track.to_ltrb()
        bbox        = ltrb
        cv2.rectangle(
            img         = img, 
            pt1         = (int(bbox[0]), int(bbox[1])),                 # part1 Vertex of the rectangle
            pt2         = (int(bbox[2]), int(bbox[3])),                 # part2 Vertex of the rectangle opposite to pt1
            color       = (0, 0, 255), 
            thickness   = 2
        )
        try: 
            ####################### for EASYOCR ###############################
            r0              = int(bbox[0]) if int(bbox[0])>0 else 0
            r1              = int(bbox[1]) if int(bbox[1])>0 else 0
            r2              = int(bbox[2]) if int(bbox[2])>0 else 0
            r3              = int(bbox[3]) if int(bbox[3])>0 else 0
            plate_region    = frame[r1:r3, r0:r2]
            plate_image     = Image.fromarray(cv2.cvtColor(plate_region, cv2.COLOR_BGR2RGB))
            plate_array     = np.array(plate_image)
            plate_number    = reader.readtext(plate_array)
            concat_number   = ' '.join([number[1] for number in plate_number])
            number_conf     = np.mean([number[2] for number in plate_number])
            #####################################################################
        except:
            pass
       
        # print(f'Plate Numbers: labels')
    cv2.imshow("Tracking", img)  # Show the frame with tracked objects
    cv2.waitKey(1)  # Pause for a little while (1ms)
    video_out.write(img)
    counter +=1


video_out.release()                                                     # This is important too, don't forget!!!
cv2.destroyAllWindows()                           

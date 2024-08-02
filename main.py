from ultralytics import YOLO
import cv2
import numpy as np
import util
from sort.sort import *
from util import get_car, read_license_plate, write_csv
from readplate import read_plate

results = {}

mot_tracker = Sort()


# car model
coco_model = YOLO('yolov8n.pt')

# license plate model
license_plate_detector = YOLO('best.pt')

# load video
cap = cv2.VideoCapture('demo.mp4')

# indexes corresponding to list 
# 2 is car, 3 is motorbike, 5 is bus, 7 is truck
vehicles = [2, 3, 5, 7]

# read frames
frame_nmr = -1 # frame counter
ret = True
while ret:
    frame_nmr += 1
    ret, frame = cap.read()
    if ret:
        results[frame_nmr] = {}
        

        # detect vehicles
        detections = coco_model(frame)[0]
        detections_ = [] # same as detections but no class_id
        for detection in detections.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = detection 
            if int(class_id) in vehicles: 
                detections_.append([x1, y1, x2, y2, score])

        # track vehicles
        track_ids = mot_tracker.update(np.asarray(detections_))

        # detect license plates
        license_plates = license_plate_detector(frame)[0]
        
        license_plate_detections = license_plates.boxes.data.tolist()

        # Draw detected license plates on the frame for visualization
        for license_plate in license_plate_detections:
            x1, y1, x2, y2, score, class_id = license_plate
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        
        # Path for the frame with license plate detections
        detection_frame_filename = f"./imagetest/detection_frame_{frame_nmr}.jpg"

        for license_plate in license_plate_detections:
            x1, y1, x2, y2, score, class_id = license_plate

            # assign license plate to car
            xcar1, ycar1, xcar2, ycar2, car_id = get_car(license_plate, track_ids)

            if car_id != -1:
                # crop license plate
                license_plate_crop = frame[int(y1):int(y2), int(x1): int(x2), :]

                # process license plate
                license_plate_crop_gray = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY)

                _, license_plate_crop_thresh = cv2.threshold(license_plate_crop_gray, 85, 255, cv2.THRESH_BINARY_INV)
                cv2.imwrite(detection_frame_filename, license_plate_crop_thresh)

                # read license plate number
                license_plate_text, license_plate_text_score = read_license_plate(license_plate_crop_thresh)

            print(frame_nmr)
            if license_plate_text is not None:
                results[frame_nmr][car_id] = {
                            'car': {'bbox': [xcar1, ycar1, xcar2, ycar2]},
                            'license_plate': {
                                'bbox': [x1, y1, x2, y2],
                                'text': license_plate_text,
                                'bbox_score': score,
                                'text_score': license_plate_text_score
                            }
                        }

# write results
write_csv(results, './test.csv')

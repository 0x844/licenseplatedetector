import ast

import cv2
import numpy as np
import pandas as pd


def draw_border(img, top_left, bottom_right, color=(0, 255, 0), thickness=3, line_length_x=50, line_length_y=50):
    x1, y1 = top_left
    x2, y2 = bottom_right

    cv2.line(img, (x1, y1), (x1, y1 + line_length_y), color, thickness)  #-- top-left
    cv2.line(img, (x1, y1), (x1 + line_length_x, y1), color, thickness)

    cv2.line(img, (x1, y2), (x1, y2 - line_length_y), color, thickness)  #-- bottom-left
    cv2.line(img, (x1, y2), (x1 + line_length_x, y2), color, thickness)

    cv2.line(img, (x2, y1), (x2 - line_length_x, y1), color, thickness)  #-- top-right
    cv2.line(img, (x2, y1), (x2, y1 + line_length_y), color, thickness)

    cv2.line(img, (x2, y2), (x2, y2 - line_length_y), color, thickness)  #-- bottom-right
    cv2.line(img, (x2, y2), (x2 - line_length_x, y2), color, thickness)

    return img


results = pd.read_csv('./test_interpolated.csv')

# load video
video_path = 'demo.mp4'
cap = cv2.VideoCapture(video_path)

fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Specify the codec
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter('./out.mp4', fourcc, fps, (width, height))

license_plate = {}
for car_id in np.unique(results['car_id']):
    max_ = np.amax(results[results['car_id'] == car_id]['license_number_score'])
    license_plate[car_id] = {'license_crop': None,
                             'license_plate_number': results[(results['car_id'] == car_id) &
                                                             (results['license_number_score'] == max_)]['license_number'].iloc[0]}
    cap.set(cv2.CAP_PROP_POS_FRAMES, results[(results['car_id'] == car_id) &
                                             (results['license_number_score'] == max_)]['frame_nmr'].iloc[0])
    ret, frame = cap.read()

    x1, y1, x2, y2 = ast.literal_eval(results[(results['car_id'] == car_id) &
                                              (results['license_number_score'] == max_)]['license_plate_bbox'].iloc[0].replace('[ ', '[').replace('   ', ' ').replace('  ', ' ').replace(' ', ','))

    license_crop = frame[int(y1):int(y2), int(x1):int(x2), :]
    license_crop = cv2.resize(license_crop, (int((x2 - x1) * 400 / (y2 - y1)), 400))

    license_plate[car_id]['license_crop'] = license_crop


frame_nmr = -1

cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

# read frames
ret = True
while ret:
    ret, frame = cap.read()
    frame_nmr += 1
    if ret:
        df_ = results[results['frame_nmr'] == frame_nmr]
        for row_indx in range(len(df_)):
            # draw car
            car_x1, car_y1, car_x2, car_y2 = ast.literal_eval(df_.iloc[row_indx]['car_bbox'].replace('[ ', '[').replace('   ', ' ').replace('  ', ' ').replace(' ', ','))
            draw_border(frame, (int(car_x1), int(car_y1)), (int(car_x2), int(car_y2)), (0, 255, 0), 3,
                        line_length_x=100, line_length_y=100)

            # draw license plate
            x1, y1, x2, y2 = ast.literal_eval(df_.iloc[row_indx]['license_plate_bbox'].replace('[ ', '[').replace('   ', ' ').replace('  ', ' ').replace(' ', ','))
            #bounding box around license plate
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)

            # crop license plate
            license_crop = license_plate[df_.iloc[row_indx]['car_id']]['license_crop']

            H, W, _ = license_crop.shape

            try:
                print(f"Processing car ID: {df_.iloc[row_indx]['car_id']}")

                # Draw car bounding box
                car_x1, car_y1, car_x2, car_y2 = ast.literal_eval(df_.iloc[row_indx]['car_bbox'].replace('[ ', '[').replace('   ', ' ').replace('  ', ' ').replace(' ', ','))
                cv2.rectangle(frame, (int(car_x1), int(car_y1)), (int(car_x2), int(car_y2)), (0, 255, 0), 2)
                
                # Draw license plate bounding box
                x1, y1, x2, y2 = ast.literal_eval(df_.iloc[row_indx]['license_plate_bbox'].replace('[ ', '[').replace('   ', ' ').replace('  ', ' ').replace(' ', ','))
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)

                # Get the license plate crop
                license_crop = license_plate[df_.iloc[row_indx]['car_id']]['license_crop']

                # Resize the license plate crop to fit in the top-right corner
                max_height = 100  # Maximum height for the resized license plate image
                aspect_ratio = license_crop.shape[1] / license_crop.shape[0]
                new_width = int(max_height * aspect_ratio)
                resized_license_crop = cv2.resize(license_crop, (new_width, max_height))

                H, W, _ = resized_license_crop.shape

                print(f"Resized license crop shape: {resized_license_crop.shape}")
                print(f"Car bounding box: ({car_x1}, {car_y1}), ({car_x2}, {car_y2})")

                # Calculate fixed overlay position (top-right corner)
                start_y = 0
                end_y = start_y + H
                start_x = frame.shape[1] - W
                end_x = start_x + W

                # Ensure the overlay positions are within the frame bounds
                if end_y <= frame.shape[0] and end_x <= frame.shape[1]:
                    # Overlay the resized license plate crop on the frame
                    frame[start_y:end_y, start_x:end_x, :] = resized_license_crop

                    # Overlay a white background for the license plate number text
                    background_start_y = end_y
                    background_end_y = background_start_y + 30
                    frame[background_start_y:background_end_y, start_x:end_x, :] = (255, 255, 255)

                    # Get text size and draw the license plate number
                    license_plate_number = license_plate[df_.iloc[row_indx]['car_id']]['license_plate_number']
                    (text_width, text_height), _ = cv2.getTextSize(license_plate_number, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)
                    print(f"License plate number: {license_plate_number}, Text size: {text_width}, {text_height}")

                    text_x = start_x + max((W - text_width) // 2, 0)
                    text_y = background_start_y + (30 - text_height) // 2 + text_height
                    if text_y - text_height >= 0:  # Ensure text position is within bounds
                        cv2.putText(frame, license_plate_number, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2, cv2.LINE_AA)
                else:
                    print("Skipping overlay due to size mismatch or out-of-bounds error.")

            except Exception as e:
                print(f"Exception occurred: {e}")






        out.write(frame)
        frame = cv2.resize(frame, (1280, 720))

        cv2.imshow('frame', frame)
        

out.release()
cap.release()
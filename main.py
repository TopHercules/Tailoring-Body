import cv2
import numpy as np
import math
from flask import Flask, flash, request, jsonify;
import os
from image_process import segment_body as segment, get_landmark
from image_measure import measure_height as get_height_in_image
from skimage.draw import line
import PredictHeight.predict_height as height

def delete_images(id):
    os.remove(f'front{id}.png')
    os.remove(f'side{id}.png')

def find_index(): 
    cnt = 0
    while os.path.exists(f'front{cnt}.png'):
        cnt = cnt + 1
    return cnt

def get_white_points(mask, start, end):
    _, width, _ = mask.shape
    a = (start[1] - end[1]) / (start[0] - end[0])
    start = [0, int(start[1] - start[0] * a)]
    end = [width - 1, int(end[1] + a * (width - end[0]))]
    
    rr, cc = line(start[1], start[0], end[1], end[0])
    white_points = sum([1 for r, c in zip(rr, cc) if np.array_equal(mask[r, c], [255, 255, 255])])
    return white_points

def get_white_points_horizontal_line(mask, y_pos):
    white_points = np.sum(np.all(mask[int(y_pos), :] == [255, 255, 255], axis=1))
    return white_points

def get_white_points_from_id(mask, spt, y_pos):
    _, width, _ = mask.shape
    id = 0
    current = 0
    prev = 0
    cnt = 0
    spt = 1
    for i in range(width):
        if np.array_equal(mask[int(y_pos), i], [255, 255, 255]):
            current = 1
        else: current = 0
        if current != prev and prev == 0:
            id += 1
        if id == spt and current == 1:
            cnt += 1
        if id > spt:
            break
        prev = current
    return cnt

def convert_to_binary_mask(mask_img, threshold):
    # Convert the mask image to grayscale
    gray_mask = cv2.cvtColor(mask_img, cv2.COLOR_BGR2GRAY)
    
    # Apply thresholding to create a binary mask
    _, binary_mask = cv2.threshold(gray_mask, threshold, 255, cv2.THRESH_BINARY)
    
    result = cv2.cvtColor(binary_mask, cv2.COLOR_GRAY2BGR)
    return result

def get_distance_between_points(start, end):
    distance = math.sqrt((end[0] - start[0])**2 + (end[1] - start[1])**2)
    return distance

app = Flask(__name__)
app.secret_key = 'Tailoring App'
@app.route('/measure', methods=['GET', 'POST'])

def index():
    if request.method == 'POST':
        if 'gender' in request.form:
            gender = request.form.get('gender')
        else:
            flash("Input the gender of the person")
        
        if 'front' not in request.files:
            flash("You didn't upload the front image. Please upload the front image")
        if 'side' not in request.files:
            flash("You didn't upload the side image. Please upload the side image")
        
        id = find_index()
        
        img_front = request.files['front']
        img_side = request.files['side']
        
        img_front.save(f'front{id}.png')
        img_side.save(f'side{id}.png')
        
        if img_front.name == '':
            delete_images(id)
            flash("Can't find the front image. Please upload the front image again")
        if img_side.name == '':
            delete_images(id)
            flash("Can't find the side image. Please upload the side image again")
    else:
        flash("Should use POST method")
        return
    
    front_path = f'front{id}.png'
    side_path = f'side{id}.png'
    
    mask_front = segment(front_path)
    mask_side = segment(side_path)
    
    mask_front = convert_to_binary_mask(mask_front, 100)
    mask_side = convert_to_binary_mask(mask_side, 100)
    
    front_img_height = get_height_in_image(mask_front)
    side_img_height = get_height_in_image(mask_side)

    if front_img_height == 0 or side_img_height == 0:
        delete_images(id)
        flash("Can't process the image. Please capture again")
    
    marked_front, landmark_front = get_landmark(front_path)
    marked_side, landmark_side = get_landmark(side_path)
    
    if len(marked_front) == 0:
        delete_images(id)
        flash("Can't find the human in front image. Please capture again")
    if len(marked_side) == 0:
        delete_images(id)
        flash("Can't find the human in side image. Please capture again")
    
    if len(landmark_front) != 33 or len(landmark_side) != 33:
        delete_images(id)
        flash("Please capture the full body image")
    
    # Estimate the height
    try:
        realHeight = height.predict(f'front{id}.png')
        print(realHeight)
    except:
        delete_images(id)
        flash("Error occured during the height estimation")
    
    scale = realHeight / front_img_height
    mark = landmark_front
    
    result = {}
    result["height"] = int(realHeight)
    
    if gender == "female":
        # Shoulder (Shoulder to shoulder)
        result["shoulder"] = int(get_white_points(mask_front, mark[11], mark[12]) * scale)
        
        # Bust (Armpit to armpit across the breasts)
        result["bust"] = int(get_distance_between_points(mark[12], mark[11]) * 1.2 * scale)
        
        # Chest (Round the chest area to the back - 360 degrees)
        sFront = (get_distance_between_points(mark[12], mark[11]) * 2 + get_distance_between_points(mark[24], mark[23])) / 3
        sSide = get_white_points_horizontal_line(mask_side, (mark[12][1] + mark[11][1] + (mark[23][1] + mark[24][1]) * 2) / 6)
        result["chest"] = int((sFront + sSide) * 2 * 0.9 * scale)
        
        # Shoulder to Under Bust (Vertical measurement)
        result["shoulder2under_bust"] = int(get_white_points(mask_front, mark[11], mark[12]) * scale)
        
        # High Waist (abdomen to waist)  
        result["high_waist"] = int(((mark[23][1] + mark[24][1]) / 2 - (mark[11][1] + mark[12][1]) / 2) / 2 * scale)
        
        # Waist (Above bottom)
        sFront = (get_distance_between_points(mark[12], mark[11]) + get_distance_between_points(mark[24], mark[23]) * 2) / 3
        sSide = get_white_points_horizontal_line(mask_side, ((mark[12][1] + mark[11][1]) * 2 + mark[23][1] + mark[24][1]) / 6)
        
        result["waist"] = int((sFront + sSide) * 2 * 0.8 * scale)
        
        # Hips (round the bottom)
        sFront = get_white_points_from_id(mask_front, 2, (mark[24][1] + mark[23][1]) / 2)
        sSide = get_white_points_horizontal_line(mask_front, (mark[24][1] + mark[23][1]) / 2)
        result["hips"] = int((sFront + sSide) * 2 * 0.9 * scale)
        
        # Thigh (of one leg)
        avg = ((mark[24][1] * 2 + mark[26][1]) / 3 + (mark[24][1] * 2 + mark[26][1]) / 3) / 2
        avg = (get_white_points_horizontal_line(mask_front, avg) + get_white_points_horizontal_line(mask_side, avg) * 2) * 0.9
        result["thigh"] = int(avg * scale)
        
        # Half body Length (Neckline to waist line)
        avg = ((mark[12][1] + mark[11][1]) / 2 - (mark[10][1] + mark[9][1]) / 2) / 2 + (mark[23][1] + mark[24][1]) / 2 - (mark[11][1] + mark[12][1]) / 2
        result["neck2waist"] = int(avg * scale)
        
        # Half body Length (Neckline to below buttock line)
        avg = (mark[24][1] + mark[23][1]) / 2 - (mark[10][1] + mark[9][1]) / 2
        result["neck2buttock"] = int(avg * scale)
        
        # Body Length (Neckline to the back of the knee)
        avg = (mark[26][1] + mark[25][1]) / 2 - (mark[10][1] + mark[9][1]) / 2
        result["neck2knee"] = int(avg * scale)
        
        # Full Body Length (Neckline to the ankle)
        avg = (mark[28][1] + mark[27][1]) / 2 - (mark[10][1] + mark[9][1]) / 2
        result["neck2ankle"] = int(avg * scale)
        
        # Bust Point (From the armpit to top of the breast)
        avg = get_distance_between_points(mark[12], mark[11]) / 4 * 1.2
        result["bust_point"] = int(avg * scale)
        
        # Arm Hole (The socket where arm connects to the shoulder) 
        avg = (get_white_points_from_id(mask_front, 1, mark[14][1]) + get_white_points_from_id(mask_front, 3, mark[13][1]))
        result["arm_hole"] = int(avg * scale)
        
        # Round Sleeves (the mouth or end of the sleeve - Short and long i.e elbow area or wrist area of the hand)
        avg = (get_white_points_from_id(mask_front, 1, mark[16][1]) + get_white_points_from_id(mask_front, 3, mark[15][1]))
        result["round_sleeve"] = int(avg * scale)
        
        # Tummy Height (Under the breast to the waistline)
        sFront = (get_distance_between_points(mark[12], mark[11]) + get_distance_between_points(mark[24], mark[23]) * 2) / 3
        sSide = get_white_points_horizontal_line(mask_side, ((mark[12][1] + mark[11][1]) * 2 + mark[23][1] + mark[24][1]) / 6)
        result["tummy"] = int((sFront + sSide) * 2 * 0.9 * scale)
        
        # Crotch without Band - (this is the measurement of the distance between the front waist and back wait through under the virginia to the back)
        sFront = (get_distance_between_points(mark[12], mark[11]) + get_distance_between_points(mark[24], mark[23]) * 2) / 3
        sSide = get_white_points_horizontal_line(mask_side, ((mark[12][1] + mark[11][1]) * 2 + mark[23][1] + mark[24][1]) / 6)
        result["crotch"] = int((sFront + sSide) * 2 * 0.9 * scale)

    elif gender == "male":
            # Head
            avg = (mark[4][1] + (mark[4][1] - mark[8][1]) + mark[1][1] + (mark[1][1] - mark[7][1])) / 2
            result["head"] = int(get_white_points_horizontal_line(mask_front, avg) * math.pi * scale)
            
            # Neck (round)
                # First, measure your height (Hin) in inches.
                # Next, measure your weight (Wlb) in pounds.
                # Use the formula NC = (Hin / 39.3701 + Wlb / 2.20462) / 2 to estimate the neck circumference.
                # predict1 = (realHeight / 100 + weight) / 2 / 0.393701
            avg = sum(mark[i][1] for i in range(9, 13)) / 4
            result["neck"] = int(get_white_points_horizontal_line(mask_front, avg) * scale)
            
            # Upper body length (Neckline to below buttock line)
            avg = (mark[24][1] + mark[23][1]) / 2 - (mark[10][1] + mark[9][1]) / 2
            result["neck2buttock"] = int(avg * scale)
            
            # Upper body length (Neckline to the knee)
            avg = (mark[26][1] + mark[25][1]) / 2 - (mark[10][1] + mark[9][1]) / 2
            result["neck2knee"] = int(avg * scale)
            
            # Full body length (Neckline to the ankle)
            avg = (mark[28][1] + mark[27][1]) / 2 - (mark[10][1] + mark[9][1]) / 2
            result["neck2ankle"] = int(avg * scale)
            
            # Shoulder (shoulder to shoulder)
            result["shoulder"] = int(get_white_points(mask_front, mark[11], mark[12]) * scale)
            
            # Long Sleeve (from the shoulder socket/Arm hole to the wrist)
            avg = get_distance_between_points(mark[12], mark[14]) + get_distance_between_points(mark[14], mark[16]) + get_distance_between_points(mark[11], mark[13]) + get_distance_between_points(mark[13], mark[15])
            result["long_sleeve"] = int(avg / 2 * scale)
            
            # Short sleeve (from the shoulder socket/Arm hole to the elbow)
            avg = get_distance_between_points(mark[12], mark[14]) + get_distance_between_points(mark[11], mark[13])
            result["short_sleeve"] = int(avg / 2 * scale)
            
            # Cuflink area (same as wrist hole as explained below)
            avg = (get_white_points_from_id(mask_front, 1, mark[16][1]) + get_white_points_from_id(mask_front, 3, mark[15][1]))
            result["cuflink"] = int(avg * scale)
            
            # Arm Hole (The socket where arm connects to the shoulder) 
            avg = (get_white_points_from_id(mask_front, 1, mark[14][1]) + get_white_points_from_id(mask_front, 3, mark[13][1]))
            result["arm_hole"] = int(avg * scale)
            
            # Round Sleeves (the mouth or end of the sleeve - Short and long i.e elbow area or wrist area of the arm)
            avg = (get_white_points_from_id(mask_front, 1, mark[16][1]) + get_white_points_from_id(mask_front, 3, mark[15][1]))
            result["round_sleeve"] = int(avg * scale)
            
            # Chest (round the chest area to the back)
            # sFront = get_white_points_from_id(mask_front, 2, (mark[12][1] + mark[11][1] + (mark[23][1] + mark[24][1]) * 2) / 6)
            sFront = (get_distance_between_points(mark[12], mark[11]) * 2 + get_distance_between_points(mark[24], mark[23])) / 3
            sSide = get_white_points_horizontal_line(mask_side, (mark[12][1] + mark[11][1] + (mark[23][1] + mark[24][1]) * 2) / 6)
            result["chest"] = int((sFront + sSide) * 2 * 0.9 * scale)
            
            # Round Body (Tummy)
            sFront = (get_distance_between_points(mark[12], mark[11]) + get_distance_between_points(mark[24], mark[23]) * 2) / 3
            sSide = get_white_points_horizontal_line(mask_side, ((mark[12][1] + mark[11][1]) * 2 + mark[23][1] + mark[24][1]) / 6)
            result["tummy"] = int((sFront + sSide) * 2 * 0.9 * scale)
            
            # Trouser Length (to the ankle or knee for knickers)
            avg = get_distance_between_points(mark[24], mark[26]) + get_distance_between_points(mark[23], mark[25]) + get_distance_between_points(mark[25], mark[27]) +get_distance_between_points(mark[26], mark[28])
            result["trouser"] = int(avg / 2 * 1.1 * scale)
            
            # Waist 
            sFront = (get_distance_between_points(mark[12], mark[11]) + get_distance_between_points(mark[24], mark[23]) * 2) / 3
            sSide = get_white_points_horizontal_line(mask_side, ((mark[12][1] + mark[11][1]) * 2 + mark[23][1] + mark[24][1]) / 6)
            
            result["waist"] = int((sFront + sSide) * 2 * 0.8 * scale)
            
            # Thigh
            avg = ((mark[24][1] * 2 + mark[26][1]) / 3 + (mark[24][1] * 2 + mark[26][1]) / 3) / 2
            avg = (get_white_points_horizontal_line(mask_front, avg) + get_white_points_horizontal_line(mask_side, avg) * 2) * 0.9
            result["thigh"] = int(avg * scale)
            
            # Hip
            sFront = get_white_points_from_id(mask_front, 2, (mark[24][1] + mark[23][1]) / 2)
            sSide = get_white_points_horizontal_line(mask_front, (mark[24][1] + mark[23][1]) / 2)
            result["hip"] = int((sFront + sSide) * 2 * 0.9 * scale)
            
            # Leg hole (Round the ankle - Trouser opening)
            avg = get_white_points_from_id(mask_front, 1, mark[28][1]) + get_white_points_from_id(mask_front, 2, mark[27][1])
            result["leg_hole"] = int(avg / 2 * 3.14 * 0.8 * scale)
    # except:
    #     delete_images(id)
    #     flash("Calculation is not correct.")
    delete_images(id)
    
    return jsonify(result)
    
if __name__ == "__main__":
    height.load_model()
    app.run(host = "0.0.0.0", port = 8000)
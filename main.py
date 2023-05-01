import json
import os

import cv2
import numpy as np
import torch

from utils import BLUE_PLATE, GREEN_PLATE
from utils import angle2radian, radian2angle
from utils import imgResize, edgesDetection, morphology, imgRectify, get_peaks, get_predict, euqualizeHist
from model import PNet

def plateDetection(img, config):
    # only leave blue and green color
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    blue_bound = np.array(config["blue bound"])
    blue_mask = cv2.inRange(img_hsv, blue_bound[0], blue_bound[1])
    
    green_bound = np.array(config["green bound"])
    green_mask = cv2.inRange(img_hsv, green_bound[0], green_bound[1])
    
    mask = cv2.bitwise_or(blue_mask, green_mask)
    img_filtered = cv2.bitwise_and(img, img, mask=mask)
    # transform to gray and noise reduction
    img_gray = cv2.cvtColor(img_filtered, cv2.COLOR_BGR2GRAY)
    blur = config["blur_ksize"]
    img_blur = cv2.GaussianBlur(img_gray, (blur, blur), 0)
    
    # edges detection
    edges = edgesDetection(img_blur, config["edge detection"])

    # morphology
    morph = morphology(edges, config["morphology"])
    contours, hierarchy = cv2.findContours(morph, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    
    blocks = []
    EPS = config["cut EPS"]
    for cnt in contours:
        rect = cv2.boundingRect(cnt)
        x, y, w, h = rect
        ratio = w / h
        
        if ratio < config["aspect range"][0] or ratio > config["aspect range"][1]: continue
        
        area = cv2.contourArea(cnt)
        if w * h < config["minArea"]: continue
        
        rectangularity = area / (w * h)
        if rectangularity < config["rect thresh"]: continue
    
        blue = blue_mask[y:y+h, x:x+w]
        blue_weight = blue.sum() / 255

        green = green_mask[y:y+h, x:x+w]
        green_weight = green.sum() / 255

        if blue_weight > config["color thresh"] * h * w and blue_weight > green_weight:
            blocks.append((BLUE_PLATE, img[y:y+h, x+EPS:x+w-EPS]))
        elif green_weight > config["color thresh"] * h * w:
            blocks.append((GREEN_PLATE, img[y:y+h, x+EPS:x+w-EPS])) 
        
    return blocks

def plateSegamentation(plates, config):
    res = []
    for t, plate in plates:
        plate_rect = imgRectify(plate, t, config["rectify"])
        
        k = cv2.getStructuringElement(cv2.MORPH_RECT, config["erode ksize"])
        erode = cv2.erode(plate_rect, k, iterations = 1)
        
        y_hist = erode.sum(axis=0) / 255
        y_average = np.sum(y_hist) / y_hist.shape[0]
        y_thresh =  y_average / 3
        segs = get_peaks(y_hist, y_thresh, x_thresh=config["peaks min interval"])
        N_chars = 7 if t == BLUE_PLATE else 8
        if len(segs) < N_chars: continue
        
        length = segs[-1] - segs[0]
        max_interval = length / N_chars
        pairs1 = []
        pair = [segs[0], segs[1]]
        for i in range(1, len(segs), 2):
            if segs[i] - pair[0] < max_interval:
                pair[-1] = segs[i]
            else:
                pairs1.append(pair)
                if len(pairs1) == 2: break
                pair = [segs[i-1], segs[i]]
        
        segs_inv = segs[::-1]
        pairs2 = []
        pair = [segs_inv[1], segs_inv[0]]
        for i in range(1, len(segs_inv), 2):
            if pair[1] - segs_inv[i] < max_interval:
                pair[0] = segs_inv[i]
            else:
                pairs2.append(pair)
                if len(pairs2) == N_chars - 2:
                    if pairs2[0][1] - pairs2[0][0] > 7:
                        break
                    else:
                        pairs2.pop(0)
                pair = [segs_inv[i], segs_inv[i-1]]
        pairs = pairs1 + pairs2[::-1]
    
        chars = []
        for x1, x2 in pairs:
            y_hist = erode[:, x1:x2].sum(axis=1)
            y_zero = np.arange(len(y_hist))[y_hist == 0]
            k = np.argmax(y_zero[1:] - y_zero[:-1])
            y1, y2 = y_zero[k], y_zero[k+1]
            char = plate_rect[y1:y2, x1-5:x2+5]
            char = cv2.resize(char, (20, 20))
            chars.append(char)
        res.append(chars)
        
    return res

def plateRecognition(chars, chinese_net, chars_net, chinese_id2labels, chars_id2labels):
    chinese = torch.as_tensor(np.expand_dims(np.array([chars[0]]) / 255, axis=1)).float()
    chars = torch.as_tensor(np.expand_dims(np.array(chars[1:]) / 255, axis=1)).float()

    char_chinese = get_predict(chinese_net, chinese, chinese_id2labels, is_word=True)
    char_word = get_predict(chars_net, chars, chars_id2labels, is_word=False)
    detect = char_chinese + char_word
    return detect

def easyCheck(directory, config, chinese_net, chars_net, chinese_id2labels, chars_id2labels):
    print(f"#### easy ####")
    for img_name in os.listdir(directory):
        img_path = os.path.join(directory, img_name)
        img = cv2.imread(img_path)
        img = imgResize(img, 700)
        h, w = img.shape[:2]
        
        balance_img = euqualizeHist(img)
        
        img_hsv = cv2.cvtColor(balance_img, cv2.COLOR_BGR2HSV)
        blue_bound = np.array(config["blue bound"])
        blue_mask = cv2.inRange(img_hsv, blue_bound[0], blue_bound[1])
        
        green_bound = np.array(config["green bound"])
        green_mask = cv2.inRange(img_hsv, green_bound[0], green_bound[1])
        
        blue_weight = np.sum(blue_mask / 255)
        green_weight = np.sum(green_mask / 255)

        t = blue_weight < green_weight
        
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_gray = imgResize(img_gray, 256)
        h, w = img_gray.shape[:2]
        img_gray = cv2.GaussianBlur(img_gray, (3, 3), 0)
        _, img_gray = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        if t == 1:
            img_gray = 255 - img_gray
        
        toler = 30
        min_theta = angle2radian(90 - toler)
        max_theta = angle2radian(90 + toler) 
        edges = cv2.Canny(img_gray, 100, 200)
        lines = cv2.HoughLines(edges, 1, np.pi/180, 50, min_theta=min_theta, max_theta=max_theta)
        
        left, right = [], []
        for line in lines:
            theta = line[0][1]
            if theta > np.pi / 2:
                left.append(theta)
            else:
                right.append(theta)
                
        if len(left) > len(right):
            mean = np.mean(left)
        else:
            mean = np.mean(right)
            
        rot_theta = radian2angle(mean) - 90
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, rot_theta, 0.8)

        mask_rot = cv2.warpAffine(img_gray, M, (w, h))
        _, mask_rot = cv2.threshold(mask_rot, 100, 255, cv2.THRESH_BINARY)
        
        # remove rivets and redundant lines
        for y in range(h):
            s = e = -1
            for i in range(w):
                if mask_rot[y, i]:
                    s = i
                    break
            if s == -1: continue
            
            for j in range(w):
                if mask_rot[y, w - j - 1]:
                    e = w - j - 1
                    break
            
            max_part = 0
            last = s
            for k in range(s, e):
                if mask_rot[y, k] != mask_rot[y, k+1]:
                    max_part = max(max_part, k - last + 1)
                    last = k + 1
            max_part = max(max_part, e - last + 1)
            valid = e - s + 1
            if max_part > valid / 3 or mask_rot[y].sum() / 255 > valid * 0.8:
                mask_rot[y-1:y+1, :] = 0         

        # imgs_show.append(mask_rot)
        y_hist = mask_rot.sum(axis=0) / 255
        y_average = np.sum(y_hist) / y_hist.shape[0]
        y_thresh =  y_average / 3
        segs = get_peaks(y_hist, y_thresh, x_thresh=config["peaks min interval"])
        N_chars = 7 if t == 0 else 8
        
        length = segs[-1] - segs[0]
        max_interval = length / N_chars
        pairs1 = []
        pair = [segs[0], segs[1]]
        for i in range(1, len(segs), 2):
            if segs[i] - pair[0] < max_interval:
                pair[-1] = segs[i]
            else:
                pairs1.append(pair)
                if len(pairs1) == 2: break
                pair = [segs[i-1], segs[i]]
        
        segs_inv = segs[::-1]
        pairs2 = []
        pair = [segs_inv[1], segs_inv[0]]
        for i in range(1, len(segs_inv), 2):
            if pair[1] - segs_inv[i] < max_interval:
                pair[0] = segs_inv[i]
            else:
                pairs2.append(pair)
                if len(pairs2) == N_chars - 2:
                    if pairs2[0][1] - pairs2[0][0] > 7:
                        break
                    else:
                        pairs2.pop(0)
                pair = [segs_inv[i], segs_inv[i-1]]
        pairs = pairs1 + pairs2[::-1]

        chars = []
        for x1, x2 in pairs:
            y_hist = mask_rot[:, x1:x2].sum(axis=1)
            y1 = 0
            while y1 < len(y_hist):
                if y_hist[y1] > 0:
                    break
                y1 += 1
            y2 = len(y_hist) - 1
            while y2 > y1:
                if y_hist[y2] > 0:
                    break
                y2 -= 1
            char = mask_rot[y1-2:y2+2, x1-3:x2+3]
            char = cv2.resize(char, (20, 20))
            chars.append(char)
            # cv2.imshow("test", char)
            # cv2.waitKey(0)
        res = plateRecognition(chars, chinese_net, chars_net, chinese_id2labels, chars_id2labels)
        print(f"img name: {img_name} Licence plate: " + "".join(res[:2]) + "·"  + "".join(res[2:]))
        

if __name__ == "__main__":
    config_path = "config.json"
    with open(config_path, 'r') as f:
        config = json.load(f)
            
    # load model
    cp_dir = config["checkpoint directory"]
    cp_chinese = os.path.join(cp_dir, "chinese_Pnet.pt")
    cp_chars = os.path.join(cp_dir, "chars_Pnet.pt")
    
    with open("data\VehicleLicense\chinese_match.json", 'r', encoding="utf-8") as f:
        chinese_match = json.load(f)
    with open("data\VehicleLicense\chars_match.json", 'r', encoding="utf-8") as f:
        chars_match = json.load(f)    
    chinese_id2labels = {i: v for i, (k, v) in enumerate(chinese_match.items())}
    chars_id2labels = {i: v for i, (k, v) in enumerate(chars_match.items())}
    
    chinese_net = PNet(len(chinese_id2labels))
    chars_net = PNet(len(chars_id2labels))
    
    chinese_net.load_state_dict(torch.load(cp_chinese))
    chars_net.load_state_dict(torch.load(cp_chars))
    
    
    easyCheck("images\easy", config, chinese_net, chars_net, chinese_id2labels, chars_id2labels)
    
    degrees = [ "medium", "difficult"]
    for degree in degrees:
        print(f"#### {degree} ####")
        directory = os.path.join("images", degree)
        for img_name in os.listdir(directory):
            img_path = os.path.join(directory, img_name)
            # img_path = r"images\medium\2-2.jpg"
            img = cv2.imread(img_path)
            img = imgResize(img, config["MAX_WIDTH"])
            # img = euqualizeHist(img)
            plates = plateDetection(img, config)
            plates_char = plateSegamentation(plates, config)
            
            # print(f"# of licence plates detected: {len(plates_char)}")
            for chars in plates_char:
                res = plateRecognition(chars, chinese_net, chars_net, chinese_id2labels, chars_id2labels)
                print(f"img name: {img_name} Licence plate: " + "".join(res[:2]) + "·"  + "".join(res[2:]))
    
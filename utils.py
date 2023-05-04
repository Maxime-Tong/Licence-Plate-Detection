import cv2
import torch
import numpy as np

BLUE_PLATE = 0
GREEN_PLATE = 1

angle2radian = lambda x: x * np.pi / 180
radian2angle = lambda x: x * 180 / np.pi

def imgResize(img, MAX_WIDTH=700):
    h, w = img.shape[:2]
    r = MAX_WIDTH / w
    if w > MAX_WIDTH:
        return cv2.resize(img, (MAX_WIDTH, int(h * r)), interpolation=cv2.INTER_AREA)
    else:
        return cv2.resize(img, (MAX_WIDTH, int(h * r)), interpolation=cv2.INTER_LINEAR)
    
def edgesDetection(img, config):
    sobelx64f = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=config["Sobel ksize"])
    abs_sobel64f = np.absolute(sobelx64f)
    sobel_8u = np.uint8(abs_sobel64f)
    _, binary = cv2.threshold(sobel_8u, config["thresh"], config["maxVal"], cv2.THRESH_BINARY)
    return binary

def morphology(img, config):
    ksize = config["ksize"]
    iters = config["iters"]
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (ksize[0], ksize[1]))      
    flow = cv2.dilate(img, kernel, iterations = iters[0])
    flow = cv2.erode(flow, kernel, iterations = iters[1])
    output = cv2.dilate(flow, kernel, iterations = iters[2])
    return output

def imgRectify(img, t, config):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_gray = imgResize(img_gray, 256)
    img_gray = cv2.GaussianBlur(img_gray, (3, 3), 0)
    
    h, w = img_gray.shape[:2]
    
    # inverse the green plate
    if t == GREEN_PLATE:
        img_gray = 255 - img_gray
    _, img_gray = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    
    toler = config["toler theta"]
    min_theta = angle2radian(90 - toler)
    max_theta = angle2radian(90 + toler) 
    edges = cv2.Canny(img_gray, config["canny"][0], config["canny"][1])
    lines = cv2.HoughLines(edges, 1, np.pi/180, config["hough thresh"], min_theta=min_theta, max_theta=max_theta)
    
    if lines is None: return img_gray
    
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
    # _, binary = cv2.threshold(img_gray, 100, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
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
        if max_part > valid / 3 or mask_rot[y].sum() / 255 > valid * config["valid ratio"]:
            mask_rot[y-1:y+1, :] = 0         
    
    M_inv = cv2.getRotationMatrix2D(center, -rot_theta, 1.1)
    mask = cv2.warpAffine(mask_rot, M_inv, (w, h))
    
    _, mask = cv2.threshold(mask, 150, 255, cv2.THRESH_BINARY)
    return mask

def get_peaks(hist, y_thresh, x_thresh):
    peaks = []
    in_peak = False
    hist[-1] = hist[0] = 0
    for i, v in enumerate(hist):
        if not in_peak and v > y_thresh:
            peaks.append(i-1)
            in_peak = True
        elif in_peak and v < y_thresh:
            if i - peaks[-1] < x_thresh:
                peaks.pop(-1)
            else:
                peaks.append(i)
            in_peak = False
    return peaks

def get_predict(net, data, id2labels, is_word):
    net.eval()
    with torch.no_grad():
        out = net(data)
        
    if is_word:
        # chinese
        pred = torch.argmax(out, dim=-1)
        res = [id2labels[int(i)] for i in pred]
    else:
        # words
        first = torch.argmax(out[0][:-9])
        pred = torch.argmax(out[1:], dim=-1)
        res = [id2labels[int(first)]] + [id2labels[int(i)] for i in pred]
    return res

def euqualizeHist(img):
    r, g, b = cv2.split(img)
    r1 = cv2.equalizeHist(r)
    g1 = cv2.equalizeHist(g)
    b1 = cv2.equalizeHist(b)
    return cv2.merge([r1, g1, b1])
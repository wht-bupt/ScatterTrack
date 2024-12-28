import cv2
import numpy as np
from tracking_toolbox import *


def gray_and_blur(img, blur_kernel):
    if img.shape[-1] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.GaussianBlur(img, blur_kernel, 0, 0)
    return img


def center_dist(bbox1, bbox2):
    x1, y1, w1, h1 = bbox1
    cx1 = x1 + w1 // 2
    cy1 = y1 + h1 // 2
    x2, y2, w2, h2 = bbox2
    cx2 = x2 + w2 // 2
    cy2 = y2 + h2 // 2
    center_distance = np.sqrt((cx1 - cx2) ** 2 + (cy1 - cy2) ** 2)
    return center_distance


def merge_bbox(fragments, id1, id2):
    result = []
    for i, fragment in enumerate(fragments):
        if i != id1 and i != id2:
            result.append(fragment)
        elif i == id1:
            x11, y11, x12, y12 = xywh_to_xyxy(fragments[id1])
            x21, y21, x22, y22 = xywh_to_xyxy(fragments[id2])
            frag = [min(x11, x21), min(y11, y21), max(x12, x22), max(y12, y22)]
            result.append(xyxy_to_xywh(frag))
    return result


def area_merge(fragments, cfg):
    is_near = True
    a, b = 0, 0
    while is_near:
        is_merge = False
        for i, fragment_base in enumerate(fragments):
            if is_merge:
                break
            for j, fragment in enumerate(fragments):
                if i == j:
                    continue
                center_distance = center_dist(fragment_base, fragment)
                if center_distance < cfg.DET.MIN_DIST:
                    is_merge = True
                    a, b = i, j
                    break

        if is_merge:
            fragments = merge_bbox(fragments, a, b)
        else:
            is_near = False

    return fragments


def get_fragments(frame_list, cfg):
    # img2与img1作差，img3与img2作差，两者进行与运算
    # 不满足t±i的帧保持检测区域与最近一个满足的帧相同
    # 高斯平滑滤波做前处理，平滑滤波核blur_size默认为3，膨胀morph_size默认为9

    morph_kernel = (cfg.DET.MORPH, cfg.DET.MORPH)
    blur_kernel = (cfg.DET.BLUR, cfg.DET.BLUR)
    dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, morph_kernel)

    img1, img2, img3 = frame_list[-(1+2*cfg.DET.INTERVAL)], frame_list[-(1+cfg.DET.INTERVAL)], frame_list[-1]

    img1 = gray_and_blur(img1, blur_kernel)
    img2 = gray_and_blur(img2, blur_kernel)
    img3 = gray_and_blur(img3, blur_kernel)
    img_diff1 = cv2.absdiff(img1, img2)
    img_diff1 = cv2.dilate(img_diff1, dilate_kernel)
    img_diff2 = cv2.absdiff(img2, img3)
    img_diff2 = cv2.dilate(img_diff2, dilate_kernel)
    window = cv2.bitwise_and(img_diff1, img_diff2)
    _, window = cv2.threshold(window, cfg.DET.BINARY_THR, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(window, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    fragments = []
    for contour in contours:
        add_flag = 1
        [x, y, w, h] = cv2.boundingRect(contour)
        if w * h >= cfg.DET.AREA_THR:
            fragments.append([x, y, w, h])

    return fragments


def get_fragments_from_black(img, cfg):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    erode_kernel = np.ones((3, 3), np.uint8)
    img = cv2.erode(img, erode_kernel)
    _, regions = cv2.threshold(img, 1, 255, cv2.THRESH_BINARY)
    fragments = []
    contours, _ = cv2.findContours(regions, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        [x, y, w, h] = cv2.boundingRect(contour)
        if w * h >= cfg.DET.AREA_THR:
            fragments.append([x, y, w, h])

    fragments = area_merge(fragments, cfg)
    return fragments


def get_fragments_from_air(frame_list, cfg):
    morph_kernel = (cfg.DET.MORPH, cfg.DET.MORPH)
    blur_kernel = (cfg.DET.BLUR, cfg.DET.BLUR)
    dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, morph_kernel)

    img1, img2, img3 = frame_list[-(1 + 2 * cfg.DET.INTERVAL)], frame_list[-(1 + cfg.DET.INTERVAL)], frame_list[-1]

    img1 = gray_and_blur(img1, blur_kernel)
    img2 = gray_and_blur(img2, blur_kernel)
    img3 = gray_and_blur(img3, blur_kernel)
    img_diff1 = cv2.absdiff(img1, img2)
    img_diff1 = cv2.dilate(img_diff1, dilate_kernel)
    img_diff2 = cv2.absdiff(img2, img3)
    img_diff2 = cv2.dilate(img_diff2, dilate_kernel)
    window = cv2.bitwise_and(img_diff1, img_diff2)
    _, window = cv2.threshold(window, cfg.DET.BINARY_THR, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(window, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    fragments = []
    for contour in contours:
        add_flag = 1
        [x, y, w, h] = cv2.boundingRect(contour)
        if w * h >= cfg.DET.AREA_THR:
            if len(fragments) == 0:
                fragments.append([x, y, w, h])
            else:
                for fragment in fragments:
                    x_org, y_org, w_org, h_org = fragment
                    iou_box = iou([x_org, y_org, w_org, h_org], [x, y, w, h])
                    if iou_box > 0.001:
                        add_flag = 0
                        break
                if add_flag == 1:
                    fragments.append([x, y, w, h])

    return fragments


def get_fragments_from_universe(img, cfg):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, regions = cv2.threshold(img, cfg.DET.BINARY_THR, 255, cv2.THRESH_BINARY)
    fragments = []
    contours, _ = cv2.findContours(regions, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        [x, y, w, h] = cv2.boundingRect(contour)
        if w * h >= cfg.DET.AREA_THR:
            fragments.append([x, y, w, h])

    return fragments






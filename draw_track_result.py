import cv2
import os
import argparse
import pandas as pd
import numpy as np
from data_toolbox import *


def cv2_draw_box(img, tracks):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    color_map = [[255, 255, 0], [255, 127, 127], [0, 0, 255],
                 [255, 0, 255], [0, 255, 255], [0, 255, 0],
                 [255, 255, 255], [0, 127, 127], [127, 127, 0], [127, 0, 127]]
    font = cv2.FONT_HERSHEY_PLAIN
    font_scale = 1
    thickness = 1
    for track in tracks:
        track_id, frag_id, x, y = track[1:5]
        id_str = '{}_{}'.format(track_id, frag_id) if frag_id != 0 else str(track_id)
        color = color_map[(track_id-1) % 10]
        cv2.rectangle(img, (x - 2, y - 2), (x + 2, y + 2), color, 1)
        text_size, _ = cv2.getTextSize(id_str, font, font_scale, thickness)
        cv2.putText(img, id_str, (x, y), font, font_scale, (0, 255, 0), thickness)
    return img


def main(args):
    tracking_results_root = os.path.join(args.track_results, 'tracks')
    data_root = 'data/imgs'
    video_output_root = os.path.join(args.track_results, 'videos')
    img_output_root = os.path.join(args.track_results, 'track_imgs')
    del_path(video_output_root)
    del_path(img_output_root)
    mkdirs(video_output_root)
    mkdirs(img_output_root)

    tracking_results = sorted(os.listdir(tracking_results_root))
    for tracking_result in tracking_results:
        tracking_results_path = os.path.join(tracking_results_root, tracking_result)
        frames = sorted(os.listdir(os.path.join(data_root, tracking_result[:3])))
        frame_init = cv2.imread(os.path.join(data_root, tracking_result[:3], frames[0]))
        width, height, _ = frame_init.shape
        fps = 25
        video_output_path = os.path.join(video_output_root, '{:03d}.avi'.format(int(tracking_result[:3])))
        out = cv2.VideoWriter(video_output_path, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), fps,
                               (height, width))

        raw_trackings = np.array(pd.read_csv(tracking_results_path).values.tolist(), dtype=int)
        trackings = raw_trackings[np.lexsort((raw_trackings[:, 1], raw_trackings[:, 0]))]
        print('start making video {}'.format(tracking_result[:3]))

        prev_pos = 0
        det_pos = 0
        length = len(trackings)
        img_output_path = os.path.join(img_output_root, tracking_result[:3])
        mkdirs(img_output_path)
        for frame_id, frame_name in enumerate(frames, start=1):
            frame_path = os.path.join(data_root, tracking_result[:3], frames[frame_id-1])
            frame = cv2.imread(frame_path)
            img_path = os.path.join(img_output_path, '{:04d}.jpg'.format(frame_id))

            for i in range(prev_pos, length):
                if trackings[i][0] != frame_id:
                    det_pos = i
                    break
            track_results = trackings[prev_pos:det_pos]
            prev_pos = det_pos
            frame_with_obj = cv2_draw_box(frame, track_results)
            out.write(frame_with_obj)
            cv2.imwrite(img_path, frame_with_obj)

            if frame_id % 50 == 0:
                print('processing frame {}'.format(frame_id))
        print('finish making video {}'.format(tracking_result[:3]))



if __name__ == "__main__":
    parser = argparse.ArgumentParser('generate visualization results')
    parser.add_argument('--track_results', '--tr', type=str, help='path of the tracking results', required=True)
    args = parser.parse_args()
    main(args)
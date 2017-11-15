#!/usr/bin/env python
# -*- coding: utf-8 -*-

#parser = argparse.ArgumentParser(description='Single Shot MultiBox Detection')
#parser.add_argument('--weights', default='weights/ssd_300_VOC0712.pth',
#                    type=str, help='Trained state_dict file path')
#parser.add_argument('--cuda', default=True, type=bool,
#                    help='Use cuda to train model')
#parser.add_argument('--video', required=True, type=str, help='video file')
#args = parser.parse_args()
#


#
#save_dir = os.path.join(os.path.dirname(args.video),\
#        os.path.split(args.video)[-1][:-4])
#if not os.path.exists(save_dir):
#    os.makedirs(save_dir)
#
#
#def cv2_demo(net, transform):
#    def predict(frame):
#        height, width = frame.shape[:2]
#        x = torch.from_numpy(transform(frame)[0]).permute(2, 0, 1)
#        x = Variable(x.unsqueeze(0))
#        y = net(x)  # forward pass
#        detections = y.data
#        # scale each detection back up to the image
#        scale = torch.Tensor([width, height, width, height])
#        for i in range(detections.size(1)):
#            j = 0
#            while detections[0, i, j, 0] >= 0.6:
#                pt = (detections[0, i, j, 1:] * scale).cpu().numpy()
#                cv2.rectangle(frame, (int(pt[0]), int(pt[1])), (int(pt[2]),
#                                                                int(pt[3])), COLORS[i % 3], 2)
#                cv2.putText(frame, labelmap[i - 1], (int(pt[0]), int(pt[1])), FONT,
#                            2, (255, 255, 255), 2, cv2.LINE_AA)
#                j += 1
#        return frame
#
#    # start video stream thread, allow buffer to fill
#    cap = cv2.VideoCapture(args.video)
#    num_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
#    cnt = 0
#    while True:
#        ok, frame = cap.read()
#        if ok:    
#            frame = predict(frame)
#            cv2.imwrite(os.path.join(save_dir, '{:0>6}.jpg'.format(cnt)), frame)
#        print(cnt)
#        cnt += 1
#        if cnt > num_frames:
#            break
#
#    cv2.destroyAllWindows()
#
#
#if __name__ == '__main__':
#    import sys
#    from os import path
#    sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
#
#    from data import BaseTransform, VOC_CLASSES as labelmap
#    from ssd import SSD300
#
#    num_classes = len(labelmap) + 1
#    net = SSD300(num_classes, 'test')
#    net.load_state_dict(torch.load(args.weights))
#    transform = BaseTransform(300, (104/256.0, 117/256.0, 123/256.0))
#
#    cv2_demo(net.eval(), transform)


from __future__ import print_function
import argparse
import os, sys
import numpy as np
import cv2
import torch
from torch.autograd import Variable

COLORS = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
FONT = cv2.FONT_HERSHEY_SIMPLEX

def autolabel(args):
    # init model
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from data import BaseTransform, VOC_CLASSES as labelmap
    from ssd import SSD300
    num_classes = len(labelmap) + 1
    net = SSD300(num_classes, 'test').eval()
    net.load_state_dict(torch.load(args.weights))
    transform = BaseTransform(300, (104/256.0, 117/256.0, 123/256.0))

    def predict(frame):
        height, width = frame.shape[:2]
        x = torch.from_numpy(transform(frame)[0]).permute(2, 0, 1)
        x = Variable(x.unsqueeze(0))
        y = net(x)  # forward pass
        detections = y.data
        # scale each detection back up to the image
        scale = torch.Tensor([width, height, width, height])
        pts = []
        for i in range(detections.size(1)): # 1--head, 2--body, 3--ass
            j = 0
            while detections[0, i, j, 0] >= 0.6:
                pt = (detections[0, i, j, 1:] * scale).cpu().numpy().astype(int)
                pts.append([i] + pt.tolist())
                j += 1
        return pts

    def predict2(frame):
        height, width = frame.shape[:2]
        x = torch.from_numpy(transform(frame)[0]).permute(2, 0, 1)
        x = Variable(x.unsqueeze(0))
        y = net(x)  # forward pass
        detections = y.data
        # scale each detection back up to the image
        scale = torch.Tensor([width, height, width, height])
        for i in range(detections.size(1)):
            j = 0
            while detections[0, i, j, 0] >= 0.6:
                pt = (detections[0, i, j, 1:] * scale).cpu().numpy()
                cv2.rectangle(frame, (int(pt[0]), int(pt[1])), (int(pt[2]),
                                                                int(pt[3])), COLORS[i % 3], 2)
                cv2.putText(frame, labelmap[i - 1], (int(pt[0]), int(pt[1])), FONT,
                            2, (255, 255, 255), 2, cv2.LINE_AA)
                j += 1
        return frame

    fid = open('anno.txt', 'w')
    
    def write2file(imgname, bboxes):
        imgname = imgname.split('/')[-2] + '_' + imgname.split('/')[-1]
        fid.write(imgname)
        for bbox in bboxes:
            fid.write(' ' + str(bbox) + ' ')
        fid.write('\n')

    classes = os.listdir(args.image_dir)
    
    import tqdm
    t = tqdm.tqdm()
    t.total = len(classes)

    for class_id in classes:
        t.update()
        imgfiles = sorted([os.path.join(os.path.join(args.image_dir, class_id), x) for x in sorted(os.listdir(os.path.join(args.image_dir, class_id))) if x.endswith('.jpg')])
        for imgfile in imgfiles:
            img = cv2.imread(imgfile)
            bboxes = predict(img)
            write2file(imgfile, bboxes)

            #img = predict2(img)
            #cv2.imshow('img', img)
            #ch = cv2.waitKey(0) & 0xff
            #if ch == 27: #ord('q')
            #    break
        #if ch == 27: #ord('q')
        #    break

    fid.close()


def main(args):
    autolabel(args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Label images using SSD300')
    parser.add_argument('--image_dir', required=True, type=str, help='image directory, containing "1,2,3,4,...30"')
    parser.add_argument('--anno_dir', default='annos', type=str, help='annotation directory')
    parser.add_argument('--weights', default='', type=str, help='ssd300_PIG2017 weight file')
    args = parser.parse_args()
    print(args)
    main(args)

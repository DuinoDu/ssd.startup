from __future__ import print_function
import torch
from torch.autograd import Variable
import cv2
import time
import argparse
import os

parser = argparse.ArgumentParser(description='Single Shot MultiBox Detection')
parser.add_argument('--weights', default='weights/ssd_300_VOC0712.pth',
                    type=str, help='Trained state_dict file path')
parser.add_argument('--cuda', default=True, type=bool,
                    help='Use cuda to train model')
parser.add_argument('--video', required=True, type=str, help='video file')
args = parser.parse_args()

COLORS = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
FONT = cv2.FONT_HERSHEY_SIMPLEX
save_dir = os.path.join(os.path.dirname(args.video),\
        os.path.split(args.video)[-1][:-4])
if not os.path.exists(save_dir):
    os.makedirs(save_dir)


def cv2_demo(net, transform):
    def predict(frame):
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

    # start video stream thread, allow buffer to fill
    cap = cv2.VideoCapture(args.video)
    num_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    cnt = 0
    while True:
        ok, frame = cap.read()
        if ok:    
            frame = predict(frame)
            cv2.imwrite(os.path.join(save_dir, '{:0>6}.jpg'.format(cnt)), frame)
        print(cnt)
        cnt += 1
        if cnt > num_frames:
            break

    cv2.destroyAllWindows()


if __name__ == '__main__':
    import sys
    from os import path
    sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

    from data import BaseTransform, VOC_CLASSES as labelmap
    from ssd import SSD300

    num_classes = len(labelmap) + 1
    net = SSD300(num_classes, 'test')
    net.load_state_dict(torch.load(args.weights))
    transform = BaseTransform(300, (104/256.0, 117/256.0, 123/256.0))

    cv2_demo(net.eval(), transform)

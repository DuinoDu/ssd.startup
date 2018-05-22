# config.py
import os.path

# gets home dir cross platform
home = os.path.expanduser("~")

# note: if you used our download scripts, this should be right
VOCroot = os.path.join(home, "data/VOCdevkit/")  # path to VOCdevkit root dir
COCOroot = os.path.join(home, "data/coco/")      # path to COCO root dir

# default batch size
BATCHES = 32
# data reshuffled at every epoch
SHUFFLE = True
# number of subprocesses to use for data loading
WORKERS = 4

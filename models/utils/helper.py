"""
@author: Jianxiong Ye
"""

import os
import cv2
import math
import time
import glob
import torch
import inspect
import logging
import numpy as np
from pathlib import Path
from typing import Optional
from threading import Thread

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]


def set_logging(verbose):
    # sets up logging for the given name
    rank = int(os.getenv('RANK', -1))  # rank in world for Multi-GPU trainings
    level = logging.INFO if verbose and rank in {-1, 0} else logging.ERROR
    logging.basicConfig(level=level, format="%(message)s")

set_logging(True)
logger = logging.getLogger('TasselELANet')

def get_nearest_divisible(x, divisor):
    # Returns nearest x divisible by divisor
    if isinstance(divisor, torch.Tensor):
        divisor = int(divisor.max())  # to int
    return math.ceil(x / divisor) * divisor

def verify_img_size(imgsz, s=32, floor=0):
    # Verify image size is a multiple of stride s in each dimension
    if isinstance(imgsz, int):  # integer
        new_size = max(get_nearest_divisible(imgsz, s), floor)
    else:
        imgsz = list(imgsz)  # convert to list if tuple
        new_size = [max(get_nearest_divisible(x, s), floor) for x in imgsz]
    if new_size != imgsz:
        logger.warning(f'WARNING ⚠️ --img-size {imgsz} must be multiple of max stride {s}, updating to {new_size}')
    return new_size

class Time_record:
    def __init__(self, t=0.0):
        self.t = t
        self.cuda = torch.cuda.is_available()

    def __enter__(self):
        self.start = self.time()
        return self

    def __exit__(self, type, value, traceback):
        self.IT = self.time() - self.start  # delta-time
        self.t += self.IT  # accumulate IT

    def time(self):
        if self.cuda:
            torch.cuda.synchronize()
        return time.time()

def colorstr(*input):
    *args, string = input if len(input) > 1 else ('blue', 'bold', input[0])  # color arguments, string
    colors = {
        'blue': '\033[34m',
        'end': '\033[0m',
        'bold': '\033[1m',
    }
    return ''.join(colors[x] for x in args) + f'{string}' + colors['end']

def infer_mode(fn):
    return torch.inference_mode()(fn) if torch.__version__ >= '1.9.0' else torch.no_grad()(fn)

def crement_path(path, sep=''):
    for n in range(0, 9999):
        p = f'{path}{sep}{n}'
        if not os.path.exists(p):
            return p

def print_info(args: Optional[dict] = None, show_file=True, show_func=False):
    # Print function arguments (optional args dict)
    x = inspect.currentframe().f_back  # previous frame
    file, _, func, _, _ = inspect.getframeinfo(x)
    if args is None:  # get args automatically
        args, _, _, frm = inspect.getargvalues(x)
        args = {k: v for k, v in frm.items() if k in args}
    try:
        file = Path(file).resolve().relative_to(ROOT).with_suffix('')
    except ValueError:
        file = Path(file).stem
    s = (f'{file}: ' if show_file else '') + (f'{func}: ' if show_func else '')
    logger.info(colorstr(s) + ', '.join(f'{k}={v}' for k, v in args.items()))

def letterbox(im, new_shape=(608, 608), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return im, ratio, (dw, dh)

class LoadImages:
    def __init__(self, path, img_size, stride):
        files = []
        p = str(Path(path))
        if '*' in p:
            files.extend(sorted(glob.glob(p, recursive=True)))  # glob
        elif os.path.isdir(p):
            files.extend(sorted(glob.glob(os.path.join(p, '*.*'))))  # dir
        elif os.path.isfile(p):
            files.append(p)  # files
        else:
            raise FileNotFoundError(f'{p} does not exist')

        images = [x for x in files if x.split('.')[-1].lower() in img_formats]

        self.img_size = img_size
        self.files = images
        self.nf = len(images)  # number of files
        self.stride = stride

    def __iter__(self):
        self.count = 0
        return self

    def __next__(self):
        if self.count == self.nf:
            raise StopIteration
        path = self.files[self.count]

        self.count += 1
        im0 = cv2.imread(path)  # BGR
        assert im0 is not None, f'Image Not Found {path}'

        info = f'image {self.count}/{self.nf}: '
        im = letterbox(im0, self.img_size, stride=self.stride, auto=True)[0]  # padded resize
        im = im.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        im = np.ascontiguousarray(im)  # contiguous

        return path, im, im0, info

class LoadWebcam:
    def __init__(self, sources, img_size, stride, auto=True, vid_stride=1):
        self.img_size = img_size
        self.stride = stride
        self.vid_stride = vid_stride
        self.sources = [str(x) for x in sources] if isinstance(sources, (list, tuple)) else [str(sources)]
        n = len(self.sources)
        self.imgs, self.fps, self.frames, self.threads = [None] * n, [0] * n, [0] * n, [None] * n
        for i, s in enumerate(self.sources):
            if s == '0':
                cap = cv2.VideoCapture(int(s))
            else:
                cap = cv2.VideoCapture(s)
            fps = cap.get(cv2.CAP_PROP_FPS)
            self.frames[i] = max(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), 0) or float('inf')
            self.fps[i] = max((fps if math.isfinite(fps) else 0) % 100, 0) or 30
            _, self.imgs[i] = cap.read()
            self.threads[i] = Thread(target=self.update, args=(i, cap, s), daemon=True)
            self.threads[i].start()
        s = np.stack([letterbox(x, img_size, stride=stride, auto=auto)[0].shape for x in self.imgs])
        self.rect = np.unique(s, axis=0).shape[0] == 1
        self.auto = auto and self.rect

    def update(self, i, cap, stream):
        n, f = 0, self.frames[i]
        while cap.isOpened() and n < f:
            n += 1
            cap.grab()
            if n % self.vid_stride == 0:
                success, im = cap.retrieve()
                if success:
                    self.imgs[i] = im
                else:
                    self.imgs[i] = np.zeros_like(self.imgs[i])
                    cap.open(stream)
            time.sleep(0.0)

    def __iter__(self):
        self.count = -1
        return self

    def __next__(self):
        self.count += 1
        if cv2.waitKey(1) == 32:
            cv2.destroyAllWindows()
            raise StopIteration
        im0 = self.imgs.copy()
        im = np.stack([letterbox(x, self.img_size, stride=self.stride, auto=self.auto)[0] for x in im0])
        im = im[..., ::-1].transpose((0, 3, 1, 2))
        im = np.ascontiguousarray(im)
        return self.sources, im, im0, None

    def __len__(self):
        return len(self.sources)

img_formats = ['bmp', 'dng', 'jpeg', 'jpg', 'mpo', 'png', 'tif', 'tiff', 'webp', 'pfm']  # acceptable image suffixes

# -----------------------------------------------Redefining cv2----------------------------------------------- #
def imread(path, flags=cv2.IMREAD_COLOR):
    return cv2.imdecode(np.fromfile(path, np.uint8), flags)

cv2.imread = imread
# -----------------------------------------------Redefining cv2----------------------------------------------- #

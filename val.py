"""
@author: Jianxiong Ye
"""

import os
import torch
import argparse
import numpy as np
from tqdm import tqdm
from pathlib import Path

from models.utils.model import MltDetectionModel, NMS, xywh2xyxy, scale_boxes
from models.utils.helper import verify_img_size, crement_path, Time_record, colorstr, logger, print_info, infer_mode
from models.utils.traineval import check_dataset, calc_correct_preds
from models.utils.dataset import create_dataloader
from models.utils.plot import ap_eval


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', default='detect', help='detect, speed')
    parser.add_argument('--data', type=str, default='config/dataset/MTDC.yaml', help='config.yaml path')
    parser.add_argument('--weights', nargs='+', type=str, default=r'data/weights/last.pt', help='model path')
    parser.add_argument('--imgsz', type=int, default=608, help='val image size (pixels)')
    parser.add_argument('--batch-size', type=int, default=4, help='image batch size')
    parser.add_argument('--workers', type=int, default=2, help='max dataloader workers')
    opt = parser.parse_args()
    opt.batch_size = 1 if opt.task == 'speed' else opt.batch_size
    print_info(vars(opt))
    return opt

@infer_mode
def run(
        data,
        weights=None,
        batch_size=16,
        imgsz=608,
        conf_thres=0.001,
        iou_thres=0.5,
        max_det=1000,
        task='val',
        workers=8,
        half=True,
        model=None,
        dataloader=None,
        save_dir=Path(''),
        plots=True,
        compute_loss=None,
):
    # Initialize/load model and set device
    training = model is not None
    if training:  # called by train.py
        device, pt, jit, engine = next(model.parameters()).device, True, False, False  # get model device, PyTorch model
        half &= device.type != 'cpu'  # half precision only supported on CUDA
        model.half() if half else model.float()
    else:  # called directly
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Directories
        save_dir = crement_path('runs/val/exp')  # increment run
        os.makedirs(save_dir, exist_ok=True)

        # Load model
        model = MltDetectionModel(weights, device=device)
        stride = model.stride()
        imgsz = verify_img_size(imgsz, s=stride)  # check image size
        device = model.device

        data = check_dataset(data)  # data check

    # Configure
    model.eval()
    cuda = device.type != 'cpu'
    nc = len(data['names'])  # number of classes
    iouv = torch.linspace(0.5, 0.95, 10, device=device)  # iou vector for mAP@0.5:0.95
    # Dataloader
    if not training:
        model.warmup(imgsz=(1, 3, imgsz, imgsz))  # warmup
        dataloader = create_dataloader(data['test'],
                                       imgsz,
                                       batch_size,
                                       stride,
                                       workers=workers,
                                       prefix=colorstr(f'{task}: '))[0]

    seen = 0
    names = data['names']
    if not training:
        s = ('%22s' + '%11s' * 6) % ('Class', 'Images', 'DT(F1', 'P', 'R', 'mAP50', 'mAP50-95)')
    else:
        s = ('%22s' + '%11s' * 6) % ('Class', 'Images', 'Instances', 'P', 'R', 'mAP50', 'mAP50-95')
    tp, fp, p, r, f1, mp, mr, map50, ap50, map = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    IT =  Time_record(),  Time_record(),  Time_record()
    loss = torch.zeros(3, device=device)
    jdict, stats, ap, ap_class = [], [], [], []
    pbar = tqdm(dataloader, desc=s, bar_format='{l_bar}{bar:10}| {n_fmt}/{total_fmt} {elapsed}')  # progress bar
    for _, (im, targets, paths, shapes) in enumerate(pbar):
        with IT[0]:
            if cuda:
                im = im.to(device, non_blocking=True)
                targets = targets.to(device)
            model, im = (obj.half() if half else obj.float() for obj in (model, im))  # FP16 supported on limited backends with CUDA
            im /= 255  # 0 - 255 to 0.0 - 1.0
            nb, _, height, width = im.shape  # batch size, channels, height, width

        # Inference
        with IT[1]:
            preds, train_out = model(im) if compute_loss else (model(im), None)

        # Loss
        if compute_loss:
            loss += compute_loss(train_out, targets)[1]  # box, obj, cls

        # NMS
        targets[:, 2:] *= torch.tensor((width, height, width, height), device=device)  # to pixels
        with IT[2]:
            preds = NMS(preds, conf_thres, iou_thres, multi_label=True, max_det=max_det)

        # Metrics
        for si, pred in enumerate(preds):
            labels = targets[targets[:, 0] == si, 1:]
            nl, npr = labels.shape[0], pred.shape[0]  # number of labels, predictions
            path, shape = Path(paths[si]), shapes[si][0]
            correct = torch.zeros(npr, len(iouv), dtype=torch.bool, device=device)  # init
            seen += 1

            if npr == 0:
                if nl:
                    stats.append((correct, *torch.zeros((2, 0), device=device), labels[:, 0]))
                continue

            predn = pred.clone()
            scale_boxes(im[si].shape[1:], predn[:, :4], shape, shapes[si][1])  # native-space pred

            # Evaluate
            if nl:
                tbox = xywh2xyxy(labels[:, 1:5])  # target boxes
                scale_boxes(im[si].shape[1:], tbox, shape, shapes[si][1])  # native-space labels
                labelsn = torch.cat((labels[:, 0:1], tbox), 1)  # native-space labels
                correct = calc_correct_preds(predn, labelsn, iouv)
            stats.append((correct, pred[:, 4], pred[:, 5], labels[:, 0]))  # (correct, conf, pcls, tcls)

    t = tuple(x.t / seen * 1E3 for x in IT)  # speeds per image

    # -------------------------------------------------detect eval------------------------------------------------- #
    # Compute metrics
    stats = [torch.cat(x, 0).cpu().numpy() for x in zip(*stats)]  # to numpy
    if len(stats) and stats[0].any():
        f1, p, r, f1, ap, ap_class = ap_eval(*stats, plot=plots, save_dir=save_dir, names=names)
        ap50, ap = ap[:, 0], ap.mean(1)  # AP@0.5, AP@0.5:0.95
        f1, mp, mr, map50, map = f1.mean(), p.mean(), r.mean(), ap50.mean(), ap.mean()
    nt = np.bincount(stats[3].astype(int), minlength=nc)  # number of targets per class

    # Print results
    if not training:
        pf = '%22s' + '%11i' + '%11.3g' * 5  # print format
        logger.info(pf % ('all', seen, f1, mp, mr, map50, map))
    else:
        pf = '%22s' + '%11i' * 2 + '%11.3g' * 4  # print format
        logger.info(pf % ('all', seen, nt.sum(), mp, mr, map50, map))

    # Print results per class
    if (nc < 50 and not training) and nc > 1 and len(stats):
        for i, c in enumerate(ap_class):
            logger.info(pf % (names[c], seen, nt[c], p[i], r[i], ap50[i], ap[i]))

    # Print speeds
    if not training:
        shape = (batch_size, 3, imgsz, imgsz)
        logger.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {shape}' % t)

    # Return results
    model.float()  # for training
    if not training:
        logger.info(f"Results saved to {colorstr('bold', save_dir)}")
    maps = np.zeros(nc) + map
    for i, c in enumerate(ap_class):
        maps[c] = ap[i]
    return (mp, mr, map50, map, *(loss.cpu() / len(dataloader)).tolist()), maps
    # -------------------------------------------------detect eval------------------------------------------------- #

def main(opt):
    if opt.task == 'detect':
        run(**vars(opt), half=False)

    elif opt.task == 'speed':
        weights = opt.weights if isinstance(opt.weights, list) else [opt.weights]
        for opt.weights in weights:
            run(**vars(opt), conf_thres=0.25, iou_thres=0.5, plots=False, half=True)

    else:
        logger.warning(('WARNING ⚠️ --Unrecognized task command, please check the task information.'))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)



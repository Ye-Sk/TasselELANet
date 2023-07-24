"""
@author: Jianxiong Ye
"""

import os
import asyncio
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def compute_ap(recall, precision):
    """ Compute the average precision, given the recall and precision curves
    # Arguments
        recall:    The recall curve (list)
        precision: The precision curve (list)
    # Returns
        Average precision, precision curve, recall curve
    """
    # Append sentinel values to beginning and end
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0]))

    # Compute the precision envelope
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))

    # Integrate area under curve
    method = 'interp'  # methods: 'continuous', 'interp'
    if method == 'interp':
        x = np.linspace(0, 1, 101)  # 101-point interp (COCO)
        ap = np.trapz(np.interp(x, mrec, mpre), x)  # integrate
    else:  # 'continuous'
        i = np.where(mrec[1:] != mrec[:-1])[0]  # points where x axis (recall) changes
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])  # area under curve

    return ap, mpre, mrec

def smooth(y, f=0.05):
    # Box filter of fraction f
    nf = round(len(y) * f * 2) // 2 + 1  # number of filter elements (must be odd)
    p = np.ones(nf // 2)  # ones padding
    yp = np.concatenate((p * y[0], y, p * y[-1]), 0)  # y padded
    return np.convolve(yp, np.ones(nf) / nf, mode='valid')  # y-smoothed

async def plot_pr_curve(px, py, ap, save_dir=Path('pr_curve.png'), names=()):
    # Precision-recall curve
    fig, ax = plt.subplots(1, 1, figsize=(7, 4), tight_layout=True)
    py = np.stack(py, axis=1)

    if 0 < len(names) < 21:  # display per-class legend if < 21 classes
        for i, y in enumerate(py.T):
            ax.plot(px, y, label=f'{names[i]} {ap[i, 0]:.3f}')  # plot(recall, precision)
    else:
        ax.plot(px, py, color='grey')  # plot(recall, precision)

    ax.plot(px, py.mean(1), linewidth=3, color='blue', label='all classes %.3f mAP@0.5' % ap[:, 0].mean())
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    ax.set_title('Precision-Recall Curve')
    fig.savefig(save_dir, dpi=200)
    plt.close(fig)

async def plot_mc_curve(px, py, save_dir=Path('mc_curve.png'), names=(), xlabel='Confidence', ylabel='Metric'):
    # Metric-confidence curve
    fig, ax = plt.subplots(1, 1, figsize=(7, 4), tight_layout=True)

    for i, y in enumerate(py):
        ax.plot(px, y, label=f'{names[i]}')  # plot(confidence, metric)

    y = smooth(py.mean(0), 0.05)
    y = y[:len(px)]  # Ensure that the length of y matches px

    value, index = y.max(), y.argmax()

    ax.plot(px, y, linewidth=3, color='blue', label=f'all classes {value:.2f} at {px[index]:.3f}')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    ax.set_title(f'{ylabel}-Confidence Curve')
    fig.savefig(save_dir, dpi=200)
    plt.close(fig)

async def plot_tp_fp_curve(tp, fp, save_dir, width=0.35):
    fig, ax = plt.subplots()
    x = np.arange(len(tp))
    ax.bar(x - width / 2, tp, width, label='TP')
    ax.bar(x + width / 2, fp, width, label='FP')
    ax.set_ylabel('Value')
    ax.set_title('True Positive and False Positive')

    for i, (tp_val, fp_val) in enumerate(zip(tp, fp)):
        ax.text(i - width / 2, tp_val, str(tp_val), ha='center', va='bottom')
        ax.text(i + width / 2, fp_val, str(fp_val), ha='center', va='bottom')

    x_ticks = [i - width / 2 for i in x] + [i + width / 2 for i in x]
    x_labels = ['TP'] * len(x) + ['FP'] * len(x)
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_labels)
    plt.savefig(save_dir, dpi=150)
    plt.close(fig)

def plot_results(path):
    result_file = os.path.join(path, 'results.txt')
    with open(result_file, 'r') as f:
        lines = f.readlines()

    data = {}
    metrics = lines[0].strip().split()[2:]
    num_metrics = len(metrics)

    for i in range(num_metrics):
        data[metrics[i]] = []

    for line in lines[1:]:
        line = line.strip().split()
        for i in range(num_metrics):
            data[metrics[i]].append(float(line[i+2]))

    epochs = range(1, len(data[metrics[0]]) + 1)

    num_rows = (num_metrics + 5) // 5
    num_cols = min(num_metrics, 4)

    plt.figure(figsize=(12, 3*num_rows))

    plot_counter = 1
    for i, metric in enumerate(metrics):
        if metric not in ['Instances', 'Size']:
            plt.subplot(num_rows, num_cols, plot_counter)
            plt.plot(epochs, data[metric], marker='o')
            plt.xlabel('Epoch')
            plt.ylabel(metric)
            plt.title(metric)
            plot_counter += 1

    precision = data['P']
    recall = data['R']
    f1_scores = [2 * (p * r) / (p + r) if p != 0 and r != 0 else 0 for p, r in zip(precision, recall)]

    plt.subplot(num_rows, num_cols, plot_counter)
    plt.plot(epochs, f1_scores, marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('F1')
    plt.title('F1')

    plt.tight_layout()
    plt.savefig(os.path.join(path, 'results.png'), dpi=120)

def ap_eval(tp, conf, pred_cls, target_cls, plot=False, save_dir='.', names=(), eps=1e-16):
    # Sort by objectness
    i = np.argsort(-conf)
    tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]

    # Find unique classes
    unique_classes, nt = np.unique(target_cls, return_counts=True)
    nc = unique_classes.shape[0]  # number of classes, number of detections

    # Create Precision-Recall curve and compute AP
    px, py = np.linspace(0, 1, 1000), []  # for plotting (1000 point)
    ap, p, r = np.zeros((nc, tp.shape[1])), np.zeros((nc, 1000)), np.zeros((nc, 1000))

    for ci, c in enumerate(unique_classes):
        i = pred_cls == c
        n_l = nt[ci]  # number of labels
        n_p = i.sum()  # number of predictions
        if n_p == 0 or n_l == 0:
            continue

        # Accumulate FPs and TPs
        fpc = (1 - tp[i]).cumsum(0)
        tpc = tp[i].cumsum(0)

        # Recall
        recall = tpc / (n_l + eps)  # recall curve
        r[ci] = np.interp(-px, -conf[i], recall[:, 0], left=0)  # negative x, xp because xp decreases

        # Precision
        precision = tpc / (tpc + fpc)  # precision curve
        p[ci] = np.interp(-px, -conf[i], precision[:, 0], left=1)  # p at pr_score

        # AP from recall-precision curve
        for j in range(tp.shape[1]):
            ap[ci, j], mpre, mrec = compute_ap(recall[:, j], precision[:, j])
            if plot and j == 0:
                py.append(np.interp(px, mrec, mpre))  # precision at mAP@0.5

    # Compute F1 (harmonic mean of precision and recall)
    f1 = 2 * p * r / (p + r + eps)
    names = [v for k, v in names.items() if k in unique_classes]  # list: only classes that have data
    names = dict(enumerate(names))  # to dict
    if plot:
        async def plot_all_curves():
            """
                Asynchronous function for plotting multiple metric curves.
            """
            tasks = [
                asyncio.create_task(plot_pr_curve(px, py, ap, Path(save_dir) / 'PR_curve.png', names)),
                asyncio.create_task(plot_mc_curve(px, f1, Path(save_dir) / 'F1_curve.png', names, ylabel='F1')),
                asyncio.create_task(plot_mc_curve(px, p, Path(save_dir) / 'P_curve.png', names, ylabel='Precision')),
                asyncio.create_task(plot_mc_curve(px, r, Path(save_dir) / 'R_curve.png', names, ylabel='Recall'))
            ]
            await asyncio.gather(*tasks)
        asyncio.run(plot_all_curves())

    i = smooth(f1.mean(0), 0.1).argmax()  # max F1 index
    p, r, f1 = p[:, i], r[:, i], f1[:, i]
    tp = (r * nt).round()  # true positives
    fp = (tp / (p + eps) - tp).round()  # false positives

    if plot:
        asyncio.run(plot_tp_fp_curve(tp, fp, Path(save_dir) / f'TP_FP_curve.png'))

    return f1, p, r, f1, ap, unique_classes.astype(int)


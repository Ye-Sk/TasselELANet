"""
@author: Jianxiong Ye
"""

import os
import numpy as np
import matplotlib.pyplot as plt

try:
    from utils.helper import logger, colorstr
except:
    from models.utils.helper import logger, colorstr


# ------------------------------------------------------count eval------------------------------------------------------ #
class CountMetrics:
    @staticmethod
    def mae(gt, pd):
        return np.mean(np.abs(np.array(gt) - np.array(pd)))

    @staticmethod
    def rmse(gt, pd):
        return np.sqrt(np.mean((np.array(gt) - np.array(pd)) ** 2))

    @staticmethod
    def rmae(gt, pd):
        return np.mean(np.abs(np.array(pd)[np.array(gt) > 0] - np.array(gt)[np.array(gt) > 0]) / np.array(gt)[np.array(gt) > 0])

    @staticmethod
    def rrmse(gt, pd):
        return np.sqrt(np.mean((np.array(pd)[np.array(gt) > 0] - np.array(gt)[np.array(gt) > 0])**2 / np.array(gt)[np.array(gt) > 0]**2))

    @staticmethod
    def rsquared(gt, pd, epsilon=1e-16):
        return np.corrcoef(np.array(gt) + epsilon, np.array(pd) + epsilon)[0, 1] ** 2
# ------------------------------------------------------count eval------------------------------------------------------ #

metrics = [('MAE', CountMetrics.mae), ('RMSE', CountMetrics.rmse), ('rMAE', CountMetrics.rmae), ('rRMSE', CountMetrics.rrmse), ('R-squared', CountMetrics.rsquared)]

# ----------------------------------beneficial experiment for counting task---------------------------------- #
def spilt(truth_list, pre_list):
    return [item[1] for item in truth_list], [dict(pre_list).get(item[0]) for item in truth_list]

def get_label_count(Root_dir):
    gt_data = [(entry.name.split('.txt')[0], len(open(entry.path, 'r+', encoding='utf-8').readlines())) for entry in os.scandir(Root_dir) if entry.name != "classes.txt" and entry.is_file()]
    return gt_data

def check_name_matching(gt_data, content):
    """
    Check if the names in gt_data have a matching name in content.

    Args:
        gt_data: list of (name, count) tuples representing ground truth data.
        content: list of (name, count) tuples representing prediction data.

    Raises:
        Exception: If the number of labels is not equal to the number of detection images
                   or if some names in gt_data do not have a matching name in content.
    """
    # Check if the number of labels is equal to the number of detection images
    assert len(gt_data) == len(content), "The number of labels is not equal to the number of detection images"

    # Get the names from gt_data and content
    gt_names = {item[0] for item in gt_data}
    pd_names = {item[0] for item in content}

    # Find the non-matching names in gt_data
    non_matching_names = gt_names - pd_names

    assert not non_matching_names, "Some names in gt_data do not have a matching name in content: {}".format(non_matching_names)

def count_visualization(gt, pd, metric_results):
    cv1 = np.array([54, 141, 178]) / 255.
    cv2 = np.array([108, 227, 191]) / 255.

    plt.figure(dpi=100)

    max_val = max(max(gt), max(pd))
    plt.xlim(0, 1.1 * max_val)
    plt.ylim(0, 1.1 * max_val)

    plt.scatter(gt, pd, color='r', marker='.', alpha=0.5, edgecolors='none', s=40)
    plt.plot([0, 1.1 * max_val], [0, 1.1 * max_val], color=cv2, linestyle='-')

    sorted_metrics = sorted(metric_results, key=lambda x: ['MAE', 'RMSE', 'rMAE', 'rRMSE', 'R-squared'].index(x[0]))
    for i, (metric_name, metric_result) in enumerate(sorted_metrics):
        y_position = 0.99 * max_val - (0.09 * max_val * i)
        if metric_name in ['rMAE', 'rRMSE']:
            metric_result = "{:.2f}%".format(metric_result)
        elif metric_name == 'R-squared':
            metric_name = 'R²'
        plt.annotate(f"{metric_name} = {metric_result}", (0.2 * max_val, y_position),
                     xytext=(0.1 * max_val, y_position), fontsize=12, color=cv1)

    plt.title('Counting Task', fontsize=15, y=1.01)
    plt.ylabel('Prediction', fontsize=12)
    plt.xlabel('Ground Truth', fontsize=12)

    plt.show()
    plt.close()

def run(pred_file_path, label_root_dir):
    gt_data = get_label_count(label_root_dir)

    with open(pred_file_path, 'r') as file:
        content = file.read()

    # check labels
    check_name_matching(gt_data, eval(content))

    gt, pd = spilt(gt_data, eval(content))

    # Calculate the metric results
    metric_results = []
    for metric_name, metric_func in metrics:
        if metric_name.startswith('r'):
            metric_result = round(metric_func(gt, pd) * 100, 1)
        elif metric_name == 'R-squared':
            metric_result = round(metric_func(gt, pd), 4)
        else:
            metric_result = round(metric_func(gt, pd), 2)

        metric_results.append((metric_name, metric_result))

    logger.info(f"{colorstr('blue', metric_results)}")
    # Visualize the beneficial experiments by plotting the ground truth vs predictions and annotating the metric results
    count_visualization(gt, pd, metric_results)


if __name__ == '__main__':
    """
    Perform evaluation and visualization of beneficial experiments using ground truth and prediction data.

    This code reads ground truth data from label files and prediction data from a result.txt file.
    It calculates various metric results and visualizes the beneficial experiments by plotting the ground truth versus predictions.

    Steps:
    1. Set paths for the result.txt file and corresponding label files.
    2. Extract ground truth data from the label files.
    3. Read content from the result.txt file.
    4. Split ground truth data and prediction data.
    5. Calculate metric results for various metrics.
    6. Visualize beneficial experiments by plotting ground truth versus predictions.
    """

    """
    pred_file_path: result.txt
    label_root_dir: folder----labels1.txt
                          ----labels2.txt
                          ----  .......
                          ----labelsN.txt 
    """
    # Path to the 'result.txt' generated by running 'python infer.py --count'
    pred_file_path = r'..\runs\infer\exp0\count\results.txt'
    # Directory path where the corresponding label files for 'result.txt' are stored
    label_root_dir = r'data/MTDC/train/labels'

    run(pred_file_path, label_root_dir)
# ----------------------------------beneficial experiment for counting task---------------------------------- #
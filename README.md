# TasselELANet
<p align="center">
  <img src="https://github.com/Ye-Sk/TasselELANet/blob/master/data/infer.jpg"/>
</p>  

**The resources in this repository are implemented in this paper：**  
[___TasselELANet: A Vision Foundation Model for Plant Detection and Counting with Efficient Layer Aggregation Network___](https://v.qq.com/x/cover/mpqzavrt4qvdstw/d00148c52qt.html?ptag=360kan.cartoon.free)

## Quantitative results
|Dataset|AP<sub>50</sub>|AP<sub>50-95</sub>|MAE|RMSE|R<sup>2</sup>|
| :----: | :----: | :----: | :----: | :----: | :----: |
|MTDC|0.865|0.463|3.99|6.39|0.9420|
|WEDU|0.931|0.547|7.03|9.03|0.9357|  
|DRPD|0.848|0.497|18.69|25.69|0.7229|

## Installation
1. The code we implement is based on PyTorch 1.12 and Python 3.8, please refer to the file `requirements.txt` to configure the required environment.      
2. To convenient install the required environment dependencies, you can also use the following command look like this：    
~~~
$ pip install -r requirements.txt 
~~~

## Training and Data Preparation
* I have already reorganized three plant datasets, you just need to move them to the specified path.
#### You can download the Maize Tassels Detection and Counting (MTDC), Wheat Ears Detection Update (WEDU), and Diverse Rice Panicle Detection (DRPD) datasets from：
|Dataset|Baidu|Google|Source|
| :----: | :----: | :----: | :----: |
|MTDC|[Baidu](https://pan.baidu.com/s/1uoh9EhC3COEt7TqC5pmA0w?pwd=plat)|[Google](https://drive.google.com/file/d/19cRDCZ4sOSv_DAyecLyOTDAegPXiIMIT/view?usp=sharing)|[Source](https://github.com/Ye-Sk/MrMT)|
|WEDU|[Baidu](https://pan.baidu.com/s/1pMQB-YNViPwRfdWtryyrFw?pwd=plat)|[Google](https://drive.google.com/file/d/1HRWXaR_Gid7-yEQbG_6wAigQ_m93bqHh/view?usp=sharing)|[Source](https://github.com/simonMadec/Wheat-Ears-Detection-Dataset)|
|DRPD|[Baidu](https://pan.baidu.com/s/1pMQB-YNViPwRfdWtryyrFw?pwd=plat)|[Google](https://drive.google.com/file/d/1duBg8yLWAs-LRtTAEFkSi3La3kBQe85_/view?usp=sharing)|[Source](https://github.com/changcaiyang/Panicle-AI)|
* Move the dataset directly into the `data` folder, the correct data format looks like this：
~~~
$./data/MTDC (or WEDU, or DRPD)
├──── train
│    ├──── images
│    └──── labels
├──── test
│    ├──── images
│    └──── labels
~~~
* Run the following command to start training on the MTDC/WEDU/DRPD dataset：
~~~
$ python train.py --data config/dataset/MTDC.yaml    # train MTDC
                         config/dataset/WEDU.yaml    # train WEDU
                         config/dataset/DRPD.yaml    # train DRPD
~~~
# Evaluation
You need to specify your training model's path for the '--weights' parameter.
* Run the following command to evaluate the results： 
~~~
$ python val.py --weights (your training weights (last.pt)) --data config/dataset/MTDC.yaml    # eval MTDC
                                                                   config/dataset/WEDU.yaml    # eval WEDU
                                                                   config/dataset/DRPD.yaml    # eval DRPD
~~~
# Inference
You need to specify your training model's path for the '--weights' parameter.  
Also, specify the source path you want to infer for the '--source' parameter.
* Run the following command on a variety of sources：
~~~
$ python infer.py --weights (your training weights (last.pt)) --source (your source path (file/dir/URL/0(webcam)))
~~~

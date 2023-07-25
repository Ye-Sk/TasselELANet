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
#### I have already reorganized three plant datasets, you just need to move them to the specified path. 
* You can download the Maize Tassels Detection and Counting (MTDC), Wheat Ears Detection Update (WEDU), and Diverse Rice Panicle Detection (DRPD) datasets from：

|Dataset|Baidu|Google|Source|
| :----: | :----: | :----: | :----: |
|MTDC|[Baidu](https://pan.baidu.com/s/16ADem84bvIkqLas-wg4kvQ?pwd=zrf6)|[Google](https://drive.google.com/file/d/1Pf7_sNJztEcMNFU5pHW5q3sEafB0po1p/view?usp=sharing)|[Source](https://github.com/poppinace/mtdc)|
|WEDU|[Baidu](https://pan.baidu.com/s/14y6cV2ukmm4nYq56lPG-Ww?pwd=jtb0)|[Google](https://drive.google.com/file/d/1jsvLSJJzsVUq2anZE0aznaKkACv5lcwi/view?usp=sharing)|[Source](https://github.com/simonMadec/Wheat-Ears-Detection-Dataset)|
|DRPD|[Baidu](https://pan.baidu.com/s/1bngkwmA-ghPJCKL5ZcrjyA?pwd=a3st)|[Google](https://drive.google.com/file/d/13BV3OivDCMpCpjcsIPs0lOoV0f0Lw80g/view?usp=sharing)|[Source](https://github.com/changcaiyang/Panicle-AI)|
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
* Run the following command to start training on the `MTDC/WEDU/DRPD` dataset：
~~~
$ python train.py --data config/dataset/MTDC.yaml    # train MTDC
                         config/dataset/WEDU.yaml    # train WEDU
                         config/dataset/DRPD.yaml    # train DRPD
~~~
## Evaluation and Inference
* Move your trained `last.pt` model to the `data/weights` directory, the correct data format looks like this：
~~~
$./data/weights
├──── last.pt
~~~
* Run the following command to evaluate the results on the `MTDC/WEDU/DRPD` dataset： 
~~~
$ python val.py --data config/dataset/MTDC.yaml    # eval MTDC
                       config/dataset/WEDU.yaml    # eval WEDU
                       config/dataset/DRPD.yaml    # eval DRPD
~~~
* Run the following command on a variety of sources：
~~~
$ python infer.py --save-img --source (your source path (file/dir/URL/0(webcam))) --data config/dataset/MTDC.yaml    # detect maize tassels
                                                                                         config/dataset/WEDU.yaml    # detect wheat ears
                                                                                         config/dataset/DRPD.yaml    # detect rice panicle
~~~
* Run the following command to evaluate the counting performance：
~~~
$ python infer.py --count --data config/dataset/MTDC.yaml    # count maize tassels
                                 config/dataset/WEDU.yaml    # count wheat ears
                                 config/dataset/DRPD.yaml    # count rice panicle
~~~

## Build your own dataset
**To train your own datasets on this framework, we recommend that :**  
1. Annotate your data with the image annotation tool [LabelIMG](https://github.com/heartexlabs/labelImg) to generate `.txt` labels in YOLO format.   
2. Refer to the `config/dataset/MTDC.yaml` example to configure your own hyperparameters file. 
3. Based on the `train.py` code example configure your own training parameters.

## Citation
#### If you find this work or code useful for your research, please cite this, Thank you!
~~~
@article{ye2023TasselELANet,  
  title={TasselELANet: A Vision Foundation Model for Plant Detection and Counting with Efficient Layer Aggregation Network},  
  author={Ye, Jianxiong and Yu, Zhenghong and Wang, Yangxu and Lu, Dunlu and Zhou, Huabing and Shengjie Liufu}, 
  year={2023}
}
~~~

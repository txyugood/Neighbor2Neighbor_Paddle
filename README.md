# 基于Paddle复现《Neighbor2Neighbor: Self-Supervised Denoising from Single Noisy Images》
## 1.简介
近年来，由于神经网络的快速发展，图像降噪也从中获得了巨大的好处。然而，由于需要大量的噪声-干净的图像对来进行模型的监督训练，限制了这些模型的推广。虽然已经有一些尝试训练一个只有单个噪声图像的图像去噪模型，但现有的自监督去噪方法存在网络训练效率低、有用信息丢失或依赖于噪声建模等问题。在这篇论文中，作者提出了一种非常简单但有效的方法，可以训练仅含噪声图像的图像去噪模型，名为Neighbor2Neighbor。
首先，提出一种随机邻域子采样器来生成训练图像对。具体的说，用于训练的输入和输出是从同一噪声图像中的子采样图像，满足了成对图像的对应像素是相邻的，同时彼此是相似的。其次，在第一阶段生成的子图片对去训练网络，并使用正则化器作为额外的损失以获得更好的性能。

<img src=./imgs/training.png></img>

上图是Neighbor2Neighbor架构概述。（a）训练的完整视图。通过噪声图片使用相邻子采样器生成一对图片。降噪网络分别使用g1(y)和g2(y)作为输入和目标。损失包含两个部分，左边部分，计算网络输出和噪声目标之间的Lrec。右边部分，考虑到子采样噪声图像和真实值之间的差异，进一步添加了损失Lreg。应该提到的是，出现两次的邻域子采样器（绿色）代表的是同一个邻域子采样器。(b)使用训练后的网络进行推理。

## 2.复现精度

在BSD300测试集的测试效果如下表,达到验收指标,PSNR: 30.79, SSIM:0.873。

| Network | opt | epoch | batch_size | dataset | PSNR | SSIM |
| --- | --- | --- | --- | --- | --- | --- |
| N2N | Adam  | 100 | 4 | BSD300 | 30.91 | 0.877 |

每一个epoch的精度可以在train.log或A_log_BSD300.csv中查看。

## 3.数据集

下载地址:

[https://aistudio.baidu.com/aistudio/datasetdetail/137360](https://aistudio.baidu.com/aistudio/datasetdetail/137360)

数据集下载解压后需要进行预处理。执行以下命令。

```shell
python dataset_tool.py --input_dir  /home/aistudio/data/ILSVRC2012_img_val \
--save_dir /home/aistudio/data/Imagenet_val
```

## 4.环境依赖
PaddlePaddle == 2.2.0

## 5.快速开始

### 模型训练

运行一下命令进行模型训练，在训练过程中会对模型进行评估，同时训练日志保存在train.log中。训练过程中的每一次评估指标都保存在result/unet_gauss25_b4e100r02/目录中。

```shell
nohup python -u train.py --data_dir=/home/aistudio/data/Imagenet_val/ \
--val_dirs=./validation --noisetype=gauss25 --save_model_path=./results \
--log_name=unet_gauss25_b4e100r02 --increase_ratio=2 >> train.log

# 查看日志
tail -f train.log
```

参数介绍：

data_dir:数据集路径

val_dirs:测试集路径

noisetype:噪声类型，根据验收指标，目前只支持gauss25。

save_model_path:模型保存路径

log_name:验证结果保存路径

increase_ratio:损失函数中Lambda的系数

最后一个epoch结束，模型验证日志如下：

```shell
0100 10300 Loss1=0.014816, Lambda=2.0, Loss2=0.019637, Loss_Full=0.034453, Time=0.0446
0100 10400 Loss1=0.018181, Lambda=2.0, Loss2=0.020429, Loss_Full=0.038610, Time=0.0452
0100 10500 Loss1=0.020329, Lambda=2.0, Loss2=0.020258, Loss_Full=0.040587, Time=0.0462
0100 10600 Loss1=0.014474, Lambda=2.0, Loss2=0.020083, Loss_Full=0.034557, Time=0.0470
0100 10700 Loss1=0.022801, Lambda=2.0, Loss2=0.020674, Loss_Full=0.043475, Time=0.0474
0100 10800 Loss1=0.018807, Lambda=2.0, Loss2=0.020078, Loss_Full=0.038885, Time=0.0472
0100 10900 Loss1=0.016025, Lambda=2.0, Loss2=0.019624, Loss_Full=0.035648, Time=0.0479
0100 11000 Loss1=0.014327, Lambda=2.0, Loss2=0.019858, Loss_Full=0.034186, Time=0.0475
Checkpoint saved to ./results/unet_gauss25_b4e100r02/2022-04-08-23-17/epoch_model_100.pdparams
[EVAL] BSD300: psnr:30.910791861936413 ssim:0.8766264295802032
```
达到验收指标。

### 模型验证

除了可以再训练过程中验证模型精度，还可以是val.py脚本加载模型验证精度，执行以下命令。

```shell
python val.py --model_path best_model.pdparams
```

输出如下：

```shell
W0409 16:05:08.099226 30040 device_context.cc:447] Please NOTE: device: 0, GPU Compute Capability: 7.0, Driver API Version: 10.1, Runtime API Version: 10.1
W0409 16:05:08.103406 30040 device_context.cc:465] device: 0, cuDNN Version: 7.6.
Loading pretrained model from best_model.pdparams
There are 50/50 variables loaded into UNet.
[EVAL] BSD300: psnr:30.910791861936413 ssim:0.8766264295802032
```

### 单张图片预测
本项目提供了单张图片的预测脚本，可生成降噪图片。使用方法如下：
```shell
python predict.py --image_path demo/42012_noise.png --model_path best_model.pdparams --save_dir ./demo/
```

参数说明：

image_path:需要预测的图片

model_path: 模型路径

save_dir: 输出图片保存路径


预测样例:


 <center>
    <img style="border-radius: 0.3125em;
    box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" 
    src=demo/42012.png width = "30%" alt=""/>
    <img style="border-radius: 0.3125em;
    box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" 
    src=demo/42012_noise.png width = "30%" alt=""/>
        <img style="border-radius: 0.3125em;
    box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" 
    src=demo/42012_noise_denoise.png width = "30%" alt=""/>
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">
      从左到右分别是clear、nosie、denoise
  	</div>
</center>

<center>
    <img style="border-radius: 0.3125em;
    box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" 
    src=demo/43074.png width = "30%" alt=""/>
    <img style="border-radius: 0.3125em;
    box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" 
    src=demo/43074_noise.png width = "30%" alt=""/>
        <img style="border-radius: 0.3125em;
    box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" 
    src=demo/43074_noise_denoise.png width = "30%" alt=""/>
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">
      从左到右分别是clear、nosie、denoise
  	</div>
</center>



### 模型导出
模型导出可执行以下命令：

```shell
python export_model.py --model_path best_model.pdparams --save_dir ./output/
```

参数说明：

model_path: 模型路径

save_dir: 输出图片保存路径

### Inference推理

可使用以下命令进行模型推理。该脚本依赖auto_log, 请参考下面TIPC部分先安装auto_log。infer命令运行如下：

```shell
python infer.py
--use_gpu=False --enable_mkldnn=False --cpu_threads=2
--model_file=output/model.pdmodel --batch_size=2 --input_file=validation/BSD300/test --enable_benchmark=True --precision=fp32 --params_file=output/model.pdiparams --save_dir output/inference_img
```

参数说明:

use_gpu:是否使用GPU

enable_mkldnn:是否使用mkldnn

cpu_threads: cpu线程数
 
model_file: 模型路径

batch_size: 批次大小

input_file: 输入文件路径

enable_benchmark: 是否开启benchmark

precision: 运算精度

params_file: 模型权重文件，由export_model.py脚本导出。

save_dir: 保存推理预测图片的路径


### TIPC基础链条测试

该部分依赖auto_log，需要进行安装，安装方式如下：

auto_log的详细介绍参考[https://github.com/LDOUBLEV/AutoLog](https://github.com/LDOUBLEV/AutoLog)。

```shell
git clone https://gitee.com/Double_V/AutoLog
cd AutoLog/
pip3 install -r requirements.txt
python3 setup.py bdist_wheel
pip3 install ./dist/auto_log-1.2.0-py3-none-any.whl
```


```shell
bash test_tipc/prepare.sh test_tipc/configs/N2N/train_infer_python.txt 'lite_train_lite_infer'

bash test_tipc/test_train_inference_python.sh test_tipc/configs/N2N/train_infer_python.txt 'lite_train_lite_infer'
```

测试结果如截图所示：

<img src=./test_tipc/data/tipc_result.png></img>

## 6.代码结构与详细说明

```
Neighbor2Neighbor_Paddle
├── A_log_BSD300.csv  # 验证模型日志
├── README.md  # 说明文件
├── arch_unet.py # 模型架构
├── best_model.pdparams # 最优模型权重
├── dataset.py # 数据集代码
├── dataset_tool.py # 数据集转换文件
├── export_model.py # 模型导出代码
├── imgs # Readme中的图片资源文件
├── infer.py # 推理代码
├── param_init.py # 模型参数初始化方法
├── test_tipc # TIPC 测试
├── train.log # 训练日志
├── train.py # 训练脚本
├── utils.py # 工具
├── val.py #验证脚本
└── validation # 验证数据集

```

## 7.模型信息

| 信息 | 描述 |
| --- | --- |
|模型名称| N2N |
|框架版本| PaddlePaddle==2.2.0|
|应用场景| 降噪 |

#!/bin/bash
#加载环境，此处加载 anaconda 环境以及通过 anaconda 创建的名为pytorch 的环境
module load anaconda/2020.11
module load cuda/10.2
source activate dassl
#python 程序运行，需在.py 文件指定调用 GPU，并设置合适的线程数，batch_size大小等
bash main_coco.sh
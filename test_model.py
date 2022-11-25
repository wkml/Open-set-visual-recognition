import torch
from model.models import SSCLIP
import argparse
import json
parser = argparse.ArgumentParser()
parser.add_argument('--backbone_name',default='RN101')
parser.add_argument('--crop_size',default=448)
args = parser.parse_args([])
graph_file='./data/coco/prob_train.npy'
word_file='./data/coco/vectors.npy'
with open('./data/coco/category_name.json', 'r') as load_category:
        category_map = json.load(load_category)
def get_parameter_number(net):
    total_num = sum(p.numel() for p in net.parameters())
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    lr_num = sum(p.numel() for p in net.word_semantic.parameters() if p.requires_grad)
    sd_num = sum(p.numel() for p in net.fc.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num, 'lr': lr_num, 'sd': sd_num}


model = SSCLIP(args=args,
                word_features=word_file,
                classname=category_map,
                num_classes=80,
                ).cuda()
for p in model.parameters():
        p.requires_grad = False
for p in model.word_semantic.parameters():
    p.requires_grad = True
for p in model.fc.parameters():
    p.requires_grad = True
x = torch.zeros([2,3,448,448]).cuda()
y = model(x)
print(get_parameter_number(model))
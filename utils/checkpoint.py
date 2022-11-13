import torch
import gc
from clip import clip
import os
import shutil

def load_pretrain_model(model, args):
    model_dict = model.resnet_101.state_dict()
    print('loading pretrained model from imagenet:')
    resnet_pretrained = torch.load(args.pretrain_model)
    print("Model Loaded")
    pretrain_dict = {k:v for k, v in resnet_pretrained.items() if not k.startswith('fc')}
    model_dict.update(pretrain_dict)
    model.resnet_101.load_state_dict(model_dict)
    del resnet_pretrained
    del pretrain_dict
    gc.collect()
    return model


def load_clip_to_cpu(args):
    backbone_name = args.backbone_name
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    model = clip.build_model(state_dict or model.state_dict(), args)

    return model

def save_code_file(args):
    prefixPath = os.path.join('exp/code/', args.post)
    if not os.path.exists(prefixPath):
        os.mkdir(prefixPath)

    fileNames = []
    if args.mode == 'SST':
        fileNames = ['scripts/SST.sh', 'SST.py', 'model/SST.py', 'loss/SST.py', 'config.py']

    for fileName in fileNames:
        shutil.copyfile(fileName, os.path.join(prefixPath, fileName.split('/')[-1]))


def save_checkpoint(args, state, isBest):
    outputPath = os.path.join('exp/checkpoint/', args.post)
    if not os.path.exists(outputPath):
        os.mkdir(outputPath)

    torch.save(state, os.path.join(outputPath, 'Checkpoint_Current.pth'))
    if isBest:
        torch.save(state, os.path.join(outputPath, 'Checkpoint_Best.pth'))
from datasets.vgdataset import VGDataset
from datasets.cocodataset import COCO2014
from datasets.voc07dataset import Voc07Dataset
from datasets.voc12dataset import Voc12Dataset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import os
def get_train_test_set(train_dir, test_dir, train_anno, test_anno, train_label=None, test_label=None,args = None):
    print('You will perform multi-scale on images for scale 640')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    scale_size = args.scale_size
    crop_size = args.crop_size
    
    train_data_transform = transforms.Compose([transforms.Resize((scale_size, scale_size)),
                                               transforms.RandomChoice([
                                               transforms.RandomCrop(512), 
                                               transforms.RandomCrop(448),
                                               transforms.RandomCrop(384),
                                               transforms.RandomCrop(320),
                                               transforms.RandomCrop(256),]),
                                               transforms.Resize((crop_size, crop_size)),
                                               transforms.ToTensor(),
                                               normalize])
    
    test_data_transform = transforms.Compose([transforms.Resize((scale_size, scale_size)),
                                             transforms.CenterCrop(crop_size),
                                             transforms.ToTensor(),
                                             normalize])
    
 
    if args.dataset == 'COCO':  
        train_set = COCO2014('train', train_dir, train_anno, train_label, train_data_transform)
        test_set = COCO2014('val', test_dir, test_anno, test_label, test_data_transform)
    elif  args.dataset == 'VG':
        train_set = VGDataset(train_dir, train_anno, train_data_transform, train_label)
        test_set = VGDataset(test_dir, test_anno, test_data_transform, test_label)
    elif args.dataset == 'VOC2007':
        train_set = Voc07Dataset(train_dir, train_anno, train_data_transform, train_label)
        test_set = Voc07Dataset(test_dir, test_anno, test_data_transform, test_label)
    elif args.dataset == 'VOC2012':
        train_set = Voc12Dataset(train_dir, train_anno, train_data_transform, train_label)
        test_set = Voc12Dataset(test_dir, test_anno, test_data_transform, test_label)
    else:
        print('%s Dataset Not Found'%args.dataset)
        exit(1)
    train_loader = DataLoader(dataset=train_set,
                              num_workers=args.workers,
                              batch_size=args.batch_size,
                              pin_memory=True,
                              shuffle = True)
                              
    test_loader = DataLoader(dataset=test_set,
                              num_workers=args.workers,
                              batch_size=args.batch_size,
                              pin_memory=True,
                              shuffle = False)
    return train_loader, test_loader

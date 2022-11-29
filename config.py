import logging
import warnings
import argparse

warnings.filterwarnings("ignore")

logger = logging.getLogger()
logger.setLevel(logging.INFO)

def show_args(args):

    logger.info("==========================================")
    logger.info("==========       CONFIG      =============")
    logger.info("==========================================")

    for arg, content in args.__dict__.items():
        logger.info("{}: {}".format(arg, content))

    logger.info("==========================================")
    logger.info("===========        END        ============")
    logger.info("==========================================")

    logger.info("\n")

def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', metavar='DATASET',
                        help='path to train dataset')
    parser.add_argument('--train_data', metavar='DIR',
                        help='path to train dataset')
    parser.add_argument('--test_data', metavar='DIR',
                        help='path to test dataset')
    parser.add_argument('--train_list', metavar='DIR',
                        help='path to train list')
    parser.add_argument('--test_list', metavar='DIR',
                        help='path to test list')
    parser.add_argument('--train_label', default='', type=str, metavar='PATH',
                        help='path to train label (default: none)')
    parser.add_argument('--graph_file', default='', type=str, metavar='PATH',
                       help='path to graph (default: none)')
    parser.add_argument('--word_file', default='', type=str, metavar='PATH',
                       help='path to word feature')
    parser.add_argument('--test_label', default='', type=str, metavar='PATH',
                        help='path to test label (default: none)')
    parser.add_argument('--print_freq', '-p', default=100, type=int, metavar='N',
                        help='number of print_freq (default: 100)')
    parser.add_argument('--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--epochs', default=90, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('--step_epoch', default=30, type=int, metavar='N',
                        help='decend the lr in epoch number')
    parser.add_argument('--batch_size', default=256, type=int,
                        metavar='N', help='mini-batch size (default: 256)')
    parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--weight_decay', '--wd', default=1e-4, type=float,
                        metavar='N', help='print frequency (default: 10)')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--crop_size', dest='crop_size',default=224, type=int,
                        help='crop size')
    parser.add_argument('--scale_size', dest = 'scale_size',default=448, type=int,
                        help='the size of the rescale image')
    parser.add_argument('--evaluate', action='store_true', help='evaluate model on validation set')
    parser.add_argument('--post', dest='post', type=str,default='',
                         help='postname of save model')
    parser.add_argument('--num_classes', default=80, type=int, metavar='N',
                        help='number of classes (default: 80)')
    parser.add_argument('--backbone_name', default="RN101",
                        help='backbone_name')
    parser.add_argument('--category_file', default="./data/coco/category_name.json",
                        help='class name of datasets')
    # parser.add_argument('--text_file', help='text_file')
    args = parser.parse_args()
    return args
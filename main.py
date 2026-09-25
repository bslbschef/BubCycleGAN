import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
from argparse import ArgumentParser
from train import BubbleModel
import torch
import numpy as np

# subfolder='exp_cropped_frames_105len/'


def get_args():
    parser = ArgumentParser(description='bub cyclegan')
    parser.add_argument('--task', type=str, default='multiple_bubble_s3_project')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--lr_split_ratio', type=float, default=0.75, help='Ratio of total epochs to switch from step LR decay to linear decay')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--epoch', type=int, default=200)
    parser.add_argument('--decay_epoch', type=int, default=10)
    parser.add_argument('--num_workers', type=int, default=20)
    parser.add_argument('--save_model', action='store_false')
    parser.add_argument('--load_model', action='store_true')

    parser.add_argument('--image_dir', type=str, default='data/images')
    parser.add_argument('--mask_dir', type=str, default='data/masks')
    parser.add_argument('--test_image_dir', type=str, default='./test/test_images')
    parser.add_argument('--test_mask_dir', type=str, default='./test/test_masks')
    parser.add_argument('--test_save_dir', type=str, default='./test/test_results')
    # parser.add_argument('--test_image_dir', type=str, default='./test/'+'more_densely/'+'test_images')
    # parser.add_argument('--test_mask_dir', type=str, default='./test/'+'more_densely/'+'test_masks')
    # parser.add_argument('--test_save_dir', type=str, default='./test/'+'more_densely/'+'test_results')
    # parser.add_argument('--test_image_dir', type=str, default='./test/'+subfolder+'test_images')
    # parser.add_argument('--test_mask_dir', type=str, default='./test/'+subfolder+'test_masks')
    # parser.add_argument('--test_save_dir', type=str, default='./test/'+subfolder+'test_results')

    parser.add_argument('--training', action='store_true')
    parser.add_argument('--testing', action='store_false')
    parser.add_argument('--fast_training', action='store_true')

    parser.add_argument('--image_channel', type=int, default=3)
    parser.add_argument('--mask_channel', type=int, default=3)

    
    parser.add_argument('--cropping', action='store_true')
    parser.add_argument('--crop_size', type=int, default=256)

    parser.add_argument('--resize', action='store_true', help='Whether to resize input images')
    parser.add_argument('--resize_size', type=int, default=512, help='Size to resize images to (square)')

    parser.add_argument('--adversarial_weight', type=float, default=1.0)
    parser.add_argument('--cycle_consistency_weight', type=float, default=10.0)
    parser.add_argument('--geometry_weight', type=float, default=1.0)

    parser.add_argument('--gen_image_type', type=str, default='resnet')
    parser.add_argument('--gen_mask_type', type=str, default='resnet')

    parser.add_argument('--load_pretrain', action='store_true')
    parser.add_argument('--pretrain_dir',type=str,default='default')
    parser.add_argument('--pretrain',action='store_true')

    args = parser.parse_args()
    return args

def main():
    torch.cuda.empty_cache()
    args = get_args()
    split_epoch = int(args.lr_split_ratio * args.epoch)
    lr_lambda1 = lambda epoch: args.lr * (0.5 ** (epoch // args.decay_epoch))
    lr_lambda2 = lambda epoch: args.lr * (0.5 ** (split_epoch // args.decay_epoch)) * (1 - (epoch - split_epoch) / (args.epoch - split_epoch))

    if args.training and args.testing == False:
        bubble = BubbleModel(args)

        # Resume from checkpoint if available
        if args.load_model:
            try:
                bubble.load_model(args)
                print(f'Resuming from epoch {bubble.epoch}')
            except FileNotFoundError:
                print('No checkpoint found, starting from scratch')

        for epoch in range(bubble.epoch, args.epoch):
            bubble.epoch = epoch

            if epoch < split_epoch:
                bubble.set_learning_rate(lr_lambda1(epoch))
            else:
                bubble.set_learning_rate(lr_lambda2(epoch))

            print('No of epoch:', epoch)
            bubble.train(args)

            print('--validating--')
            bubble.validate(args)
            bubble.save_model(args)
    elif args.testing and args.training == False:
        bubble = BubbleModel(args)
        args.load_model = True
        bubble.load_model(args)
        bubble.test(args)
    else:
        print('Check the arguments of training and testing!')


if __name__ == '__main__':
    main()
    
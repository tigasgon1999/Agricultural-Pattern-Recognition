from tqdm import tqdm
import network
import utils
import os
import random
import argparse
import sys
import numpy as np
np.set_printoptions(threshold=sys.maxsize)

from torch.utils import data
from datasets import VOCSegmentation, Cityscapes
from utils import ext_transforms as et
from metrics import StreamSegMetrics

import torch
import torch.nn as nn
from utils.visualizer import Visualizer

from PIL import Image
import matplotlib
import matplotlib.pyplot as plt

from config.configs_kf import *
import pandas as pd

prepare_gt(VAL_ROOT)
prepare_gt(TRAIN_ROOT)

train_args = agriculture_configs(net_name='MSCG-Rx50',
                                 data='Agriculture',
                                 bands_list=['NIR', 'RGB'],
                                 kf=0, k_folder=0,
                                 note='reproduce_ACW_loss2_adax'
                                 )

train_args.input_size = [512, 512]
train_args.scale_rate = 1.  # 256./512.  # 448.0/512.0 #1.0/1.0
train_args.val_size = [512, 512]
train_args.node_size = (32, 32)
train_args.train_batch = 10
train_args.val_batch = 10

train_args.lr = 1.5e-4 / np.sqrt(3)
train_args.weight_decay = 2e-5

train_args.lr_decay = 0.9
train_args.max_iter = 1e8

train_args.snapshot = ''

train_args.print_freq = 1
train_args.save_pred = False
# output training configuration to a text file
train_args.ckpt_path=os.path.abspath(os.curdir)


def get_argparser():
    parser = argparse.ArgumentParser()

    # Datset Options
    parser.add_argument("--data_root", type=str, default='./datasets/data',
                        help="path to Dataset")
    parser.add_argument("--dataset", type=str, default='voc',
                        choices=['voc', 'cityscapes', 'agr'], help='Name of dataset')
    parser.add_argument("--num_classes", type=int, default=None,
                        help="num classes (default: None)")

    # Deeplab Options
    parser.add_argument("--model", type=str, default='deeplabv3plus_mobilenet',
                        choices=['deeplabv3_resnet50',  'deeplabv3plus_resnet50',
                                 'deeplabv3_resnet101', 'deeplabv3plus_resnet101',
                                 'deeplabv3_mobilenet', 'deeplabv3plus_mobilenet'], help='model name')
    parser.add_argument("--separable_conv", action='store_true', default=False,
                        help="apply separable conv to decoder and aspp")
    parser.add_argument("--output_stride", type=int, default=16, choices=[8, 16, 32])

    # Train Options
    parser.add_argument("--test_only", action='store_true', default=False)
    parser.add_argument("--save_val_results", action='store_true', default=False,
                        help="save segmentation results to \"./results\"")
    parser.add_argument("--total_itrs", type=int, default=10e2,
                        help="epoch number (default: 30k)")
    parser.add_argument("--lr", type=float, default=0.01,
                        help="learning rate (default: 0.01)")
    parser.add_argument("--lr_policy", type=str, default='poly', choices=['poly', 'step'],
                        help="learning rate scheduler policy")
    parser.add_argument("--step_size", type=int, default=10000)
    parser.add_argument("--crop_val", action='store_true', default=False,
                        help='crop validation (default: False)')
    parser.add_argument("--batch_size", type=int, default=16,
                        help='batch size (default: 16)')
    parser.add_argument("--val_batch_size", type=int, default=4,
                        help='batch size for validation (default: 4)')
    parser.add_argument("--crop_size", type=int, default=513)
    
    parser.add_argument("--ckpt", default=None, type=str,
                        help="restore from checkpoint")
    parser.add_argument("--continue_training", action='store_true', default=False)

    parser.add_argument("--loss_type", type=str, default='cross_entropy',
                        choices=['cross_entropy', 'focal_loss'], help="loss type (default: False)")
    parser.add_argument("--gpu_id", type=str, default='0',
                        help="GPU ID")
    parser.add_argument("--weight_decay", type=float, default=1e-4,
                        help='weight decay (default: 1e-4)')
    parser.add_argument("--random_seed", type=int, default=1,
                        help="random seed (default: 1)")
    parser.add_argument("--print_interval", type=int, default=10,
                        help="print interval of loss (default: 10)")
    parser.add_argument("--val_interval", type=int, default=100,
                        help="epoch interval for eval (default: 100)")
    parser.add_argument("--download", action='store_true', default=False,
                        help="download datasets")

    # PASCAL VOC Options
    parser.add_argument("--year", type=str, default='2012',
                        choices=['2012_aug', '2012', '2011', '2009', '2008', '2007'], help='year of VOC')

    # Visdom options
    parser.add_argument("--enable_vis", action='store_true', default=False,
                        help="use visdom for visualization")
    parser.add_argument("--vis_port", type=str, default='13570',
                        help='port for visdom')
    parser.add_argument("--vis_env", type=str, default='main',
                        help='env for visdom')
    parser.add_argument("--vis_num_samples", type=int, default=8,
                        help='number of samples for visualization (default: 8)')
    return parser


def get_dataset(opts):
    """ Dataset And Augmentation
    """
    if opts.dataset == 'voc':
        train_transform = et.ExtCompose([
            #et.ExtResize(size=opts.crop_size),
            et.ExtRandomScale((0.5, 2.0)),
            et.ExtRandomCrop(size=(opts.crop_size, opts.crop_size), pad_if_needed=True),
            et.ExtRandomHorizontalFlip(),
            et.ExtToTensor(),
            et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
        ])
        if opts.crop_val:
            val_transform = et.ExtCompose([
                et.ExtResize(opts.crop_size),
                et.ExtCenterCrop(opts.crop_size),
                et.ExtToTensor(),
                et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225]),
            ])
        else:
            val_transform = et.ExtCompose([
                et.ExtToTensor(),
                et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225]),
            ])
        train_dst = VOCSegmentation(root=opts.data_root, year=opts.year,
                                    image_set='train', download=opts.download, transform=train_transform)
        val_dst = VOCSegmentation(root=opts.data_root, year=opts.year,
                                  image_set='val', download=False, transform=val_transform)

    if opts.dataset == 'cityscapes':
        train_transform = et.ExtCompose([
            #et.ExtResize( 512 ),
            et.ExtRandomCrop(size=(opts.crop_size, opts.crop_size)),
            et.ExtColorJitter( brightness=0.5, contrast=0.5, saturation=0.5 ),
            et.ExtRandomHorizontalFlip(),
            et.ExtToTensor(),
            et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
        ])

        val_transform = et.ExtCompose([
            #et.ExtResize( 512 ),
            et.ExtToTensor(),
            et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
        ])

        train_dst = Cityscapes(root=opts.data_root,
                               split='train', transform=train_transform)
        val_dst = Cityscapes(root=opts.data_root,
                             split='val', transform=val_transform)
    return train_dst, val_dst

def pixel_mapper(pixel_map):
    class_map = {
    0 : (0, 0, 0),        # background (Black)
    1 : (255, 255, 0),    # cloud_shadow (Yellow)
    2 : (255, 0, 255),    # double_plant (Purple)
    3 : (0, 255, 0),      # planter_skip (Green)
    4 : (0, 0, 255),      # standing_water (Blue)
    5 : (255, 255, 255),  # waterway (White)
    6 : (0, 255, 255),    # weed_cluster (Cian)
    }
    # Get new RGB channels
    R = pixel_map
    G = pixel_map
    B = pixel_map
    for classe in class_map.keys():
        R = np.where(R == classe, class_map[classe][0], R)
        G = np.where(G == classe, class_map[classe][1], G)
        B = np.where(B == classe, class_map[classe][2], B)

    return R,G,B

def validate(opts, model, loader, device, metrics, ret_samples_ids=None):
    """Do validation and return specified samples"""
    metrics.reset()
    ret_samples = []
    counter = 0
    image_interval = 100
    if opts.save_val_results:
        if not os.path.exists('results'):
            os.mkdir('results')
        if not os.path.exists('results/images'):
            os.mkdir('results/images')
        if not os.path.exists(f'results/images/os_{opts.output_stride}'):
            os.mkdir(f'results/images/os_{opts.output_stride}')
        if not os.path.exists('results/progress'):
            os.mkdir('results/progress')
        if not os.path.exists(f'results/progress/os_{opts.output_stride}'):
            os.mkdir(f'results/progress/os_{opts.output_stride}')
        denorm = utils.Denormalize(mean=[0.485, 0.456, 0.406], 
                                   std=[0.229, 0.224, 0.225])
        img_id = 0

    with torch.no_grad():
        for i, (images, labels) in tqdm(enumerate(loader)):
            images = images.to(device, dtype=torch.float32)
            labels = labels.to(device, dtype=torch.long)

            outputs = model(images)
            preds = outputs.detach().max(dim=1)[1].cpu().numpy()
            targets = labels.cpu().numpy()

            metrics.update(targets, preds)
            if ret_samples_ids is not None and i in ret_samples_ids:  # get vis samples
                ret_samples.append(
                    (images[0].detach().cpu().numpy(), targets[0], preds[0]))

            if opts.save_val_results:
                for j in range(len(images)):
                    counter = i+j
                    if counter % image_interval == 0:
                        # Extract predictions, labels and image
                        image = images[j].detach().cpu().numpy()
                        target = targets[j]
                        pred = preds[j]

                        # Reformat real image 
                        image = np.delete(image, 0, 0) # Delete NIR channel                        
                        image = (image * 255).transpose(1, 2, 0).astype(np.uint8) # No need to denorm it, since it was never normalized

                        # Reformat results from prediction
                        R_pred,G_pred,B_pred = pixel_mapper(pred)
                        # Create 3D image
                        formatted_pred = np.array([R_pred,G_pred,B_pred])
                        # Prepare for printing format
                        pred = formatted_pred.transpose(1, 2, 0).astype(np.uint8)
                        pred = np.clip(pred, 0, 255)  # Sanity check                   
                        # Reformat results from target
                        R_pred,G_pred,B_pred = pixel_mapper(target)
                        # Create 3D image
                        formatted_target = np.array([R_pred,G_pred,B_pred])
                        target = formatted_target.transpose(1, 2, 0).astype(np.uint8)
                        # Save results from validation
                        Image.fromarray(image).save(f'results/images/os_{opts.output_stride}/{img_id}_image.png')
                        Image.fromarray(target).save(f'results/images/os_{opts.output_stride}/{img_id}_target.png')
                        Image.fromarray(pred).save(f'results/images/os_{opts.output_stride}/{img_id}_pred.png')
                        # Overlap images
                        fig = plt.figure()
                        plt.imshow(image)
                        #plt.title("Printing overlay of target and image")
                        plt.axis('off')
                        plt.imshow(pred, alpha=0.35)
                        ax = plt.gca()
                        ax.xaxis.set_major_locator(matplotlib.ticker.NullLocator())
                        ax.yaxis.set_major_locator(matplotlib.ticker.NullLocator())
                        plt.savefig(f'results/images/os_{opts.output_stride}/{img_id}_overlay.png', bbox_inches='tight', pad_inches=0)
                        #plt.show()
                        plt.close()
                        img_id += 1
            
        score = metrics.get_results()
    return score, ret_samples


def main():
    opts = get_argparser().parse_args()
    if opts.dataset.lower() == 'voc':
        opts.num_classes = 21
    elif opts.dataset.lower() == 'cityscapes':
        opts.num_classes = 19

    # Setup visualization
    vis = Visualizer(port=opts.vis_port,
                     env=opts.vis_env) if opts.enable_vis else None
    if vis is not None:  # display options
        vis.vis_table("Options", vars(opts))

    os.environ['CUDA_VISIBLE_DEVICES'] = opts.gpu_id
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device: %s" % device)

    # Setup random seed
    torch.manual_seed(opts.random_seed)
    np.random.seed(opts.random_seed)
    random.seed(opts.random_seed)

    # Setup dataloader
    if opts.dataset=='voc' and not opts.crop_val:
        opts.val_batch_size = 1

    if opts.dataset == 'agr':
        train_dst, val_dst = train_args.get_dataset()
    else:
        train_dst, val_dst = get_dataset(opts)
    
    train_loader = data.DataLoader(
        train_dst, batch_size=opts.batch_size, shuffle=True, num_workers=2)
    val_loader = data.DataLoader(
        val_dst, batch_size=opts.val_batch_size, shuffle=True, num_workers=2)
    print("Dataset: %s, Train set: %d, Val set: %d" %
          (opts.dataset, len(train_dst), len(val_dst)))

    # Set up model
    model_map = {
        'deeplabv3_resnet50': network.deeplabv3_resnet50,
        'deeplabv3plus_resnet50': network.deeplabv3plus_resnet50,
        'deeplabv3_resnet101': network.deeplabv3_resnet101,
        'deeplabv3plus_resnet101': network.deeplabv3plus_resnet101,
        'deeplabv3_mobilenet': network.deeplabv3_mobilenet,
        'deeplabv3plus_mobilenet': network.deeplabv3plus_mobilenet
    }

    model = model_map[opts.model](num_classes=opts.num_classes, output_stride=opts.output_stride)
    if opts.separable_conv and 'plus' in opts.model:
        network.convert_to_separable_conv(model.classifier)
    utils.set_bn_momentum(model.backbone, momentum=0.01)
    
    # Set up metrics
    metrics = StreamSegMetrics(opts.num_classes)

    # Set up optimizer
    model_dict = model.backbone.state_dict()
    optimizer = torch.optim.SGD(params=[
        {'params': model.backbone.parameters(), 'lr': 0.1*opts.lr},
        {'params': model.classifier.parameters(), 'lr': opts.lr},
    ], lr=opts.lr, momentum=0.9, weight_decay=opts.weight_decay)
    #optimizer = torch.optim.SGD(params=model.parameters(), lr=opts.lr, momentum=0.9, weight_decay=opts.weight_decay)
    #torch.optim.lr_scheduler.StepLR(optimizer, step_size=opts.lr_decay_step, gamma=opts.lr_decay_factor)
    if opts.lr_policy=='poly':
        scheduler = utils.PolyLR(optimizer, opts.total_itrs, power=0.9)
    elif opts.lr_policy=='step':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=opts.step_size, gamma=0.1)

    # Set up criterion
    #criterion = utils.get_loss(opts.loss_type)
    if opts.loss_type == 'focal_loss':
        criterion = utils.FocalLoss(ignore_index=255, size_average=True)
    elif opts.loss_type == 'cross_entropy':
        criterion = nn.CrossEntropyLoss(ignore_index=255, reduction='mean')

    def save_ckpt(path):
        """ save current model
        """
        torch.save({
            "cur_itrs": cur_itrs,
            "model_state": model.module.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "best_score": best_score,
        }, path)
        print("Model saved as %s" % path)
    
    utils.mkdir('checkpoints')
    # Restore
    best_score = 0.0
    cur_itrs = 0
    cur_epochs = 0
    if opts.ckpt is not None and os.path.isfile(opts.ckpt):
        # https://github.com/VainF/DeepLabV3Plus-Pytorch/issues/8#issuecomment-605601402, @PytaichukBohdan
        checkpoint = torch.load(opts.ckpt, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint["model_state"])
        model = nn.DataParallel(model)
        model.to(device)
        if opts.continue_training:
            optimizer.load_state_dict(checkpoint["optimizer_state"])
            scheduler.load_state_dict(checkpoint["scheduler_state"])
            cur_itrs = checkpoint["cur_itrs"]
            best_score = checkpoint['best_score']
            print("Training state restored from %s" % opts.ckpt)
        print("Model restored from %s" % opts.ckpt)
        del checkpoint  # free memory
    else:
        print("[!] Retrain")
        model = nn.DataParallel(model)
        model.to(device)

    #==========   Train Loop   ==========#
    vis_sample_id = np.random.randint(0, len(val_loader), opts.vis_num_samples,
                                      np.int32) if opts.enable_vis else None  # sample idxs for visualization
    denorm = utils.Denormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # denormalization for ori images

    if opts.test_only:
        model.eval()
        val_score, ret_samples = validate(
            opts=opts, model=model, loader=val_loader, device=device, metrics=metrics, ret_samples_ids=vis_sample_id)
        print(metrics.to_str(val_score))
        return

    # Initializae placeholders for results
    mIoUs = [] 
    classIoUs0 = []
    classIoUs1 = []
    classIoUs2 = []
    classIoUs3 = []
    classIoUs4 = []
    classIoUs5 = []
    classIoUs6 = []
    train_losses = []
    iterations_train = []
    iterations_val = []
    mean_accuracies = []
    overall_accuracies = []
    train_epochs = []
    val_epochs = []

    interval_loss = 0
    while True: #cur_itrs < opts.total_itrs:
        # =====  Train  =====
        model.train()
        cur_epochs += 1
        for (images, labels) in train_loader:
            cur_itrs += 1

            images = images.to(device, dtype=torch.float32)
            labels = labels.to(device, dtype=torch.long)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            np_loss = loss.detach().cpu().numpy()
            interval_loss += np_loss
            if vis is not None:
                vis.vis_scalar('Loss', cur_itrs, np_loss)

            if (cur_itrs) % 10 == 0:
                interval_loss = interval_loss/10
                print("Epoch %d, Itrs %d/%d, Loss=%f" %
                      (cur_epochs, cur_itrs, opts.total_itrs, interval_loss))

                train_losses.append(interval_loss)
                iterations_train.append(cur_itrs)
                train_epochs.append(cur_epochs)
                interval_loss = 0.0

            if (cur_itrs) % opts.val_interval == 0:
                save_ckpt('checkpoints/latest_%s_%s_os%d.pth' %
                          (opts.model, opts.dataset, opts.output_stride))
                print("validation...")
                model.eval()
                val_score, ret_samples = validate(
                    opts=opts, model=model, loader=val_loader, device=device, metrics=metrics, ret_samples_ids=vis_sample_id)
                print(metrics.to_str(val_score))

                # Save results
                if opts.save_val_results:
                    mIoUs.append(val_score['Mean IoU'])
                    classIoUs0.append(val_score['Class IoU'][0])
                    classIoUs1.append(val_score['Class IoU'][1])
                    classIoUs2.append(val_score['Class IoU'][2])
                    classIoUs3.append(val_score['Class IoU'][3])
                    classIoUs4.append(val_score['Class IoU'][4])
                    classIoUs5.append(val_score['Class IoU'][5])
                    classIoUs6.append(val_score['Class IoU'][6])
                    mean_accuracies.append(val_score['Mean Acc'])
                    overall_accuracies.append(val_score['Overall Acc'])
                    iterations_val.append(cur_itrs)
                    val_epochs.append(cur_epochs)
                    np.save(f"./results/progress/os_{opts.output_stride}/confusion_matrix.npy", val_score['Confusion matrix'])

                if val_score['Mean IoU'] > best_score:  # save best model
                    best_score = val_score['Mean IoU']
                    save_ckpt('checkpoints/best_%s_%s_os%d.pth' %
                              (opts.model, opts.dataset,opts.output_stride))

                if vis is not None:  # visualize validation score and samples
                    vis.vis_scalar("[Val] Overall Acc", cur_itrs, val_score['Overall Acc'])
                    vis.vis_scalar("[Val] Mean IoU", cur_itrs, val_score['Mean IoU'])
                    vis.vis_table("[Val] Class IoU", val_score['Class IoU'])

                    for k, (img, target, lbl) in enumerate(ret_samples):
                        img = (denorm(img) * 255).astype(np.uint8)
                        target = train_dst.decode_target(target).transpose(2, 0, 1).astype(np.uint8)
                        lbl = train_dst.decode_target(lbl).transpose(2, 0, 1).astype(np.uint8)
                        concat_img = np.concatenate((img, target, lbl), axis=2)  # concat along width
                        vis.vis_image('Sample %d' % k, concat_img)
                model.train()
            scheduler.step()  

            if cur_itrs >=  opts.total_itrs:
                output_dict = {}
                output_dict['Epochs'] = val_epochs
                output_dict['Iterations'] = iterations_val
                output_dict['Mean IoUs'] = mIoUs
                output_dict['Class 0 IoU'] = classIoUs0
                output_dict['Class 1 IoU'] = classIoUs1
                output_dict['Class 2 IoU'] = classIoUs2
                output_dict['Class 3 IoU'] = classIoUs3
                output_dict['Class 4 IoU'] = classIoUs4
                output_dict['Class 5 IoU'] = classIoUs5
                output_dict['Class 6 IoU'] = classIoUs6
                output_dict['Mean Accs.'] = mean_accuracies
                output_dict['Overall Accs.'] = overall_accuracies

                train_dict = {}
                train_dict['Epochs'] = train_epochs
                train_dict['Iterations'] = iterations_train
                train_dict['Loss'] = train_losses

                output_df = pd.DataFrame.from_dict(output_dict)
                train_df = pd.DataFrame.from_dict(train_dict)

                output_df.to_csv(f"./results/progress/os_{opts.output_stride}/eval_results.csv", index = False)
                train_df.to_csv(f"./results/progress/os_{opts.output_stride}/train_results.csv", index = False)
                return

        
if __name__ == '__main__':
    main()

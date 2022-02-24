from tqdm import tqdm
import network
import utils
import os, sys
import random
import argparse
import numpy as np

from torch.utils import data
from datasets import VOCSegmentation, Cityscapes, BDD100k
from utils import ext_transforms as et
from metrics import StreamSegMetrics

import torch
import torch.nn as nn
from utils.visualizer import Visualizer

from PIL import Image
import matplotlib
import matplotlib.pyplot as plt

data_root = '/psi/home/li_s1/data/Season/bdd100k-data'
enable_vis = False
crop_size = 513
num_classes = 19
output_stride = 16
random_seed = 1
batch_size = 16
val_batch_size = 4 # 1?
num_workers = 2
separable_conv = False
lr = 0.1 #0.01
weight_decay = 1e-4
lr_policy = 'poly' # 'step'
total_itrs = 30e3
step_size = 10000
ckpt_path = '/psi/home/li_s1/data/Season/pretrained/checkpoints/latest_deeplabv3plus_mobilenet_BDD100k_os16.pth'
#'/psi/home/li_s1/data/Season/pretrained/best_deeplabv3plus_mobilenet_cityscapes_os16.pth'
vis_num_samples = 8
save_val_results = False
val_interval = 100
save_path = '/psi/home/li_s1/data/Season/pretrained/'
#
# def get_argparser():
#     parser = argparse.ArgumentParser()

    # Datset Options
    # parser.add_argument("--data_root", type=str, default='/psi/home/li_s1/data/Season/bdd100k-data',
    #                     help="path to Dataset")
    # parser.add_argument("--dataset", type=str, default='voc',
    #                     choices=['voc', 'cityscapes'], help='Name of dataset')
    # parser.add_argument("--num_classes", type=int, default=None,
    #                     help="num classes (default: None)")

    # Deeplab Options
    # parser.add_argument("--separable_conv", action='store_true', default=False,
    #                     help="apply separable conv to decoder and aspp")

    # Train Options
    # parser.add_argument("--test_only", action='store_true', default=False)
    # parser.add_argument("--save_val_results", action='store_true', default=False,
    #                     help="save segmentation results to \"./results\"")
    # parser.add_argument("--total_itrs", type=int, default=30e3,
    #                     help="epoch number (default: 30k)")
    # parser.add_argument("--lr", type=float, default=0.01,
    #                     help="learning rate (default: 0.01)")
    # parser.add_argument("--lr_policy", type=str, default='poly', choices=['poly', 'step'],
    #                     help="learning rate scheduler policy")
    # parser.add_argument("--step_size", type=int, default=10000)
    # parser.add_argument("--crop_val", action='store_true', default=False,
    #                     help='crop validation (default: False)')
    # parser.add_argument("--batch_size", type=int, default=16,
    #                     help='batch size (default: 16)')
    # parser.add_argument("--val_batch_size", type=int, default=4,
    #                     help='batch size for validation (default: 4)')
    # parser.add_argument("--crop_size", type=int, default=513)

    # parser.add_argument("--ckpt", default=None, type=str,
    #                     help="restore from checkpoint")
    # parser.add_argument("--continue_training", action='store_true', default=False)

    # parser.add_argument("--loss_type", type=str, default='cross_entropy',
    #                     choices=['cross_entropy', 'focal_loss'], help="loss type (default: False)")
    # parser.add_argument("--gpu_id", type=str, default='0',
    #                     help="GPU ID")
    # parser.add_argument("--weight_decay", type=float, default=1e-4,
    #                     help='weight decay (default: 1e-4)')
    # parser.add_argument("--random_seed", type=int, default=1,
    #                     help="random seed (default: 1)")
    # parser.add_argument("--print_interval", type=int, default=10,
    #                     help="print interval of loss (default: 10)")
    # parser.add_argument("--val_interval", type=int, default=100,
    #                     help="epoch interval for eval (default: 100)")
    # parser.add_argument("--download", action='store_true', default=False,
    #                     help="download datasets")

    # Visdom options
    # parser.add_argument("--enable_vis", action='store_true', default=False,
    #                     help="use visdom for visualization")
    # parser.add_argument("--vis_num_samples", type=int, default=8,
    #                     help='number of samples for visualization (default: 8)')
    # return parser

def get_dataset(target_type):
    """ Dataset And Augmentation
    """
    train_transform = et.ExtCompose([
        et.ExtRandomCrop(size=(crop_size, crop_size)),
        et.ExtColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),
        et.ExtRandomHorizontalFlip(),
        et.ExtToTensor(),
        et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
    ])

    val_transform = et.ExtCompose([
        et.ExtToTensor(),
        et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
    ])

    train_dst = BDD100k(root=data_root, split='train',
                        target_type=target_type,
                        transform=train_transform)
    val_dst = BDD100k(root=data_root, split='val',
                      target_type=target_type,
                      transform=val_transform)
    print('sizes:', len(train_dst), len(val_dst))
    return train_dst, val_dst

def validate(model, loader, device, metrics, ret_samples_ids=None):
    """Do validation and return specified samples"""
    metrics.reset()
    ret_samples = []
    if save_val_results:
        if not os.path.exists(os.path.join(save_path, 'results')):
            os.mkdir(os.path.join(save_path, 'results'))
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

            if save_val_results:
                for i in range(len(images)):
                    image = images[i].detach().cpu().numpy()
                    target = targets[i]
                    pred = preds[i]

                    image = (denorm(image) * 255).transpose(1, 2, 0).astype(np.uint8)
                    target = loader.dataset.decode_target(target).astype(np.uint8)
                    pred = loader.dataset.decode_target(pred).astype(np.uint8)

                    Image.fromarray(image).save(os.path.join(save_path, 'results/%d_image.png' % img_id))
                    Image.fromarray(target).save(os.path.join(save_path, 'results/%d_target.png' % img_id))
                    Image.fromarray(pred).save(os.path.join(save_path, 'results/%d_pred.png' % img_id))

                    fig = plt.figure()
                    plt.imshow(image)
                    plt.axis('off')
                    plt.imshow(pred, alpha=0.7)
                    ax = plt.gca()
                    ax.xaxis.set_major_locator(matplotlib.ticker.NullLocator())
                    ax.yaxis.set_major_locator(matplotlib.ticker.NullLocator())
                    plt.savefig(os.path.join(save_path, 'results/%d_overlay.png' % img_id), bbox_inches='tight', pad_inches=0)
                    plt.close()
                    img_id += 1

        score = metrics.get_results()
    return score, ret_samples

def main(target_type='masks', test_only=False):
    # Setup visualization
    vis = Visualizer(port='28333',
                     env='main') if enable_vis else None
    # if vis is not None:  # display options
    #     vis.vis_table("Options", vars(opts))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device: %s" % device)

    # Setup random seed
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)

    # Setup dataloader
    train_dst, val_dst = get_dataset(target_type)
    train_loader = data.DataLoader(
        train_dst, batch_size=batch_size, shuffle=True, num_workers=num_workers,
        drop_last=True)  # drop_last=True to ignore single-image batches.
    val_loader = data.DataLoader(
        val_dst, batch_size=val_batch_size, shuffle=True, num_workers=num_workers)
    print("Dataset: BDD100k, Train set: %d, Val set: %d" %
          (len(train_dst), len(val_dst)))

    # Set up model (all models are 'constructed at network.modeling)
    model = network.modeling.__dict__['deeplabv3plus_mobilenet'](num_classes=num_classes,
    output_stride=output_stride)
    if separable_conv:
        network.convert_to_separable_conv(model.classifier)
    utils.set_bn_momentum(model.backbone, momentum=0.01)

    # Set up metrics
    metrics = StreamSegMetrics(num_classes)

    # Set up optimizer
    optimizer = torch.optim.SGD(params=[
        {'params': model.backbone.parameters(), 'lr': 0.1 * lr},
        {'params': model.classifier.parameters(), 'lr': lr},
    ], lr=lr, momentum=0.9, weight_decay=weight_decay)
    # optimizer = torch.optim.SGD(params=model.parameters(), lr=opts.lr, momentum=0.9, weight_decay=opts.weight_decay)
    # torch.optim.lr_scheduler.StepLR(optimizer, step_size=opts.lr_decay_step, gamma=opts.lr_decay_factor)
    if lr_policy == 'poly':
        scheduler = utils.PolyLR(optimizer, total_itrs, power=0.9)
    elif lr_policy == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=0.1)

    # Set up criterion
    # criterion = utils.get_loss(opts.loss_type)
    # if opts.loss_type == 'focal_loss':
    #     criterion = utils.FocalLoss(ignore_index=255, size_average=True)
    # elif opts.loss_type == 'cross_entropy':
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

    utils.mkdir(os.path.join(save_path, 'checkpoints'))
    # Restore
    best_score = 0.0
    cur_itrs = 0
    cur_epochs = 0
    # if ckpt_path is not None and os.path.isfile(ckpt_path):
        # https://github.com/VainF/DeepLabV3Plus-Pytorch/issues/8#issuecomment-605601402, @PytaichukBohdan
    checkpoint = torch.load(ckpt_path, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint["model_state"])
    model = nn.DataParallel(model)
    model.to(device)
    # if continue_training:
    optimizer.load_state_dict(checkpoint["optimizer_state"])
    scheduler.load_state_dict(checkpoint["scheduler_state"])
    cur_itrs = checkpoint["cur_itrs"]
    best_score = checkpoint['best_score']
    # print("Training state restored from %s" % ckpt_path)
    print("Model restored from %s" % ckpt_path)
    del checkpoint  # free memory
    # else:
    #     print("[!] Retrain")
    #     model = nn.DataParallel(model)
    #     model.to(device)

    # ==========   Train Loop   ==========#
    vis_sample_id = np.random.randint(0, len(val_loader), vis_num_samples,
                                      np.int32) if enable_vis else None  # sample idxs for visualization
    denorm = utils.Denormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # denormalization for ori images

    if test_only:
        model.eval()
        val_score, ret_samples = validate(
        model=model, loader=val_loader, device=device, metrics=metrics, ret_samples_ids=vis_sample_id)
        print(metrics.to_str(val_score))
        return

    interval_loss = 0
    while True:  # cur_itrs < opts.total_itrs:
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
                interval_loss = interval_loss / 10
                print("Epoch %d, Itrs %d/%d, Loss=%f" %
                      (cur_epochs, cur_itrs, total_itrs, interval_loss))
                interval_loss = 0.0

            if (cur_itrs) % val_interval == 0:
                save_ckpt(os.path.join(save_path, 'checkpoints/latest_%s_%s_os%d.pth' %
                          ('deeplabv3plus_mobilenet', 'BDD100k', output_stride)))
                print("validation...")
                model.eval()
                val_score, ret_samples = validate(
                    model=model, loader=val_loader, device=device, metrics=metrics,
                    ret_samples_ids=vis_sample_id)
                print(metrics.to_str(val_score))
                if val_score['Mean IoU'] > best_score:  # save best model
                    best_score = val_score['Mean IoU']
                    save_ckpt(os.path.join(save_path, 'checkpoints/best_%s_%s_os%d.pth' %
                              ('deeplabv3plus_mobilenet', 'BDD100k', output_stride)))

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

            if cur_itrs >= total_itrs:
                return

if __name__ == '__main__':
    main(target_type=sys.argv[1], test_only=True)

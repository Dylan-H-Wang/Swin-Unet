import argparse
import logging
import os
import random
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from tqdm import tqdm

import albumentations as albu
from albumentations.pytorch import ToTensorV2

from datasets.dataset_synapse import Synapse_dataset, RandomGenerator
from datasets.dataset_bcss import Bcss_dataset, Bcss_dataset_val

def trainer_synapse(args, model, snapshot_path): 
    logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size * args.n_gpu
    # max_iterations = args.max_iterations
    db_train = Synapse_dataset(base_dir=args.root_path, list_dir=args.list_dir, split="train",
                               transform=transforms.Compose(
                                   [RandomGenerator(output_size=[args.img_size, args.img_size])]))
    print("The length of train set is: {}".format(len(db_train)))

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    trainloader = DataLoader(db_train, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True,
                             worker_init_fn=worker_init_fn)
    if args.n_gpu > 1:
        model = nn.DataParallel(model)
    model.train()
    ce_loss = CrossEntropyLoss()
    dice_loss = DiceLoss(num_classes)
    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    writer = SummaryWriter(snapshot_path + '/log')
    iter_num = 0
    max_epoch = args.max_epochs
    max_iterations = args.max_epochs * len(trainloader)  # max_epoch = max_iterations // len(trainloader) + 1
    logging.info("{} iterations per epoch. {} max iterations ".format(len(trainloader), max_iterations))
    best_performance = 0.0
    iterator = tqdm(range(max_epoch), ncols=70)
    for epoch_num in iterator:
        for i_batch, sampled_batch in enumerate(trainloader):
            image_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            image_batch, label_batch = image_batch.cuda(), label_batch.cuda()
            outputs = model(image_batch)
            loss_ce = ce_loss(outputs, label_batch[:].long())
            loss_dice = dice_loss(outputs, label_batch, softmax=True)
            loss = 0.4 * loss_ce + 0.6 * loss_dice
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_

            iter_num = iter_num + 1
            writer.add_scalar('info/lr', lr_, iter_num)
            writer.add_scalar('info/total_loss', loss, iter_num)
            writer.add_scalar('info/loss_ce', loss_ce, iter_num)

            logging.info('iteration %d : loss : %f, loss_ce: %f' % (iter_num, loss.item(), loss_ce.item()))

            if iter_num % 20 == 0:
                image = image_batch[1, 0:1, :, :]
                image = (image - image.min()) / (image.max() - image.min())
                writer.add_image('train/Image', image, iter_num)
                outputs = torch.argmax(torch.softmax(outputs, dim=1), dim=1, keepdim=True)
                writer.add_image('train/Prediction', outputs[1, ...] * 50, iter_num)
                labs = label_batch[1, ...].unsqueeze(0) * 50
                writer.add_image('train/GroundTruth', labs, iter_num)

        save_interval = 50  # int(max_epoch/6)
        if epoch_num > int(max_epoch / 2) and (epoch_num + 1) % save_interval == 0:
            save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
            torch.save(model.state_dict(), save_mode_path)
            logging.info("save model to {}".format(save_mode_path))

        if epoch_num >= max_epoch - 1:
            save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
            torch.save(model.state_dict(), save_mode_path)
            logging.info("save model to {}".format(save_mode_path))
            iterator.close()
            break

    writer.close()
    return "Training Finished!"


def trainer_bcss(args, model, snapshot_path):
    logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size * args.n_gpu
    # max_iterations = args.max_iterations
    db_train = Bcss_dataset(data_path=args.root_path, frac=args.frac,
                               transforms=albu.Compose(
                                   [
                                    albu.CenterCrop(args.img_size, args.img_size, True),
                                    albu.OneOf(
                                        [
                                            albu.Compose([albu.RandomRotate90(p=1), albu.HorizontalFlip(p=1)], p=0.5),
                                            albu.Rotate(limit=20, p=0.5),
                                        ],
                                        p=0.5,
                                    ),          
                                    albu.Normalize(
                                        mean=[0.6998, 0.4785, 0.6609],
                                        std=[0.2203, 0.2407, 0.1983],
                                        max_pixel_value=255.0,
                                        always_apply=True,
                                    ),
                                    ToTensorV2(transpose_mask=True),]))
    print("The length of train set is: {}".format(len(db_train)))

    db_val = Bcss_dataset_val(data_path=args.root_path,
                               transforms=albu.Compose(
                                   [
                                    albu.CenterCrop(args.img_size, args.img_size, True),         
                                    albu.Normalize(
                                        mean=[0.6998, 0.4785, 0.6609],
                                        std=[0.2203, 0.2407, 0.1983],
                                        max_pixel_value=255.0,
                                        always_apply=True,
                                    ),
                                    ToTensorV2(transpose_mask=True),]))
    print("The length of val set is: {}".format(len(db_val)))

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    trainloader = DataLoader(db_train, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True,
                             worker_init_fn=worker_init_fn)
    valloader = DataLoader(db_val, batch_size=1, shuffle=False, num_workers=4, pin_memory=True,
                             worker_init_fn=worker_init_fn)

    if args.n_gpu > 1:
        model = nn.DataParallel(model)
    
    import segmentation_models_pytorch as smp
    ce_loss = CrossEntropyLoss(ignore_index=0)
    dice_loss = smp.losses.DiceLoss(
        smp.losses.MULTICLASS_MODE, classes=[1, 2, 3, 4, 5], from_logits=True
    )
    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    writer = SummaryWriter(snapshot_path + '/log')
    iter_num = 0
    max_epoch = args.max_epochs
    max_iterations = args.max_epochs * len(trainloader)  # max_epoch = max_iterations // len(trainloader) + 1
    logging.info("{} iterations per epoch. {} max iterations ".format(len(trainloader), max_iterations))
    
    best_performance = 0.0
    iterator = tqdm(range(max_epoch), ncols=70)
    for epoch_num in iterator:
        model.train()
        for i_batch, sampled_batch in enumerate(trainloader):
            image_batch, label_batch = sampled_batch['image'], sampled_batch['mask']
            image_batch, label_batch = image_batch.cuda(), label_batch.cuda()
            outputs = model(image_batch)
            loss_ce = ce_loss(outputs, label_batch.long())
            loss_dice = dice_loss(outputs, label_batch.long())
            loss = 0.4 * loss_ce + 0.6 * loss_dice
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_

            iter_num = iter_num + 1
            writer.add_scalar('train/lr', lr_, iter_num)
            writer.add_scalar('train/total_loss', loss, iter_num)
            writer.add_scalar('train/loss_ce', loss_ce, iter_num)

            if i_batch % 50 == 0:
                logging.info('train iteration %d : loss : %f, loss_ce: %f' % (iter_num, loss.item(), loss_ce.item()))

        model.eval()
        with torch.no_grad():
            val_f1_micros = []
            val_f1_macros = []
            tumor_f1 = []
            stroma_f1 = []
            infla_f1 = []
            necr_f1 = []
            other_f1 = []
            val_acc_micros = []

            for i_batch, sampled_batch in enumerate(valloader):
                preds = []

                image_batch, label_batch = sampled_batch['image'], sampled_batch['mask']
                image_batch, label_batch = image_batch[0], label_batch[0]
                images_split = torch.split(image_batch, 128)

                for img in images_split:
                    img = img.cuda()
                    outputs = model(img)
                    preds.append(outputs.detach().cpu())

                preds = torch.cat(preds, dim=0)
                pred_mask = torch.argmax(preds, dim=1)
                tp, fp, fn, tn = smp.metrics.get_stats(
                    pred_mask.long() - 1,
                    label_batch.long() - 1,
                    mode="multiclass",
                    ignore_index=-1,
                    num_classes=5,
                )

                val_f1_micro = smp.metrics.f1_score(tp, fp, fn, tn, reduction="micro")
                val_f1_macro = smp.metrics.f1_score(tp, fp, fn, tn, reduction="macro")
                val_acc_micro = smp.metrics.accuracy(tp, fp, fn, tn, reduction="micro")
                val_acc_macro = smp.metrics.accuracy(tp, fp, fn, tn, reduction="macro")
                tumor_f1_ = (
                    smp.metrics.f1_score(
                        tp, fp, fn, tn, reduction="weighted", class_weights=[1, 0, 0, 0, 0]
                    )
                    * 5
                )
                stroma_f1_ = (
                    smp.metrics.f1_score(
                        tp, fp, fn, tn, reduction="weighted", class_weights=[0, 1, 0, 0, 0]
                    )
                    * 5
                )
                infla_f1_ = (
                    smp.metrics.f1_score(
                        tp, fp, fn, tn, reduction="weighted", class_weights=[0, 0, 1, 0, 0]
                    )
                    * 5
                )
                necr_f1_ = (
                    smp.metrics.f1_score(
                        tp, fp, fn, tn, reduction="weighted", class_weights=[0, 0, 0, 1, 0]
                    )
                    * 5
                )
                other_f1_ = (
                    smp.metrics.f1_score(
                        tp, fp, fn, tn, reduction="weighted", class_weights=[0, 0, 0, 0, 1]
                    )
                    * 5
                )

                val_f1_micros.append(val_f1_micro.item())
                val_f1_macros.append(val_f1_macro.item())
                tumor_f1.append(tumor_f1_.item())
                stroma_f1.append(stroma_f1_.item())
                infla_f1.append(infla_f1_.item())
                necr_f1.append(necr_f1_.item())
                other_f1.append(other_f1_.item())
                val_acc_micros.append(val_acc_micro.item())

            a, b, c, d, e, f, g, f = (
                np.mean(val_f1_micros),
                np.mean(val_f1_macros),
                np.mean(tumor_f1),
                np.mean(stroma_f1),
                np.mean(infla_f1),
                np.mean(necr_f1),
                np.mean(other_f1),
                np.mean(val_acc_micros),
            )
            writer.add_scalar('val/val_f1_micros', a, epoch_num)
            writer.add_scalar('val/val_f1_macros', b, epoch_num)
            writer.add_scalar('val/tumor_f1', c, epoch_num)
            writer.add_scalar('val/stroma_f1', d, epoch_num)
            writer.add_scalar('val/infla_f1', e, epoch_num)
            writer.add_scalar('val/necr_f1', f, epoch_num)
            writer.add_scalar('val/other_f1', g, epoch_num)
            writer.add_scalar('val/val_acc_micros', f, epoch_num)
            logging.info(f'val epoch {epoch_num} : f1_micro : {a}, f1_macro : {b}, tumor_f1 : {c}, stroma_f1 : {d}, infla_f1 : {e}, necr_f1 : {f}, other_f1 : {g}, acc_micro : {f}')

        save_interval = 50  # int(max_epoch/6)
        if epoch_num > int(max_epoch / 2) and (epoch_num + 1) % save_interval == 0:
            save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
            torch.save(model.state_dict(), save_mode_path)
            logging.info("save model to {}".format(save_mode_path))

        if epoch_num >= max_epoch - 1:
            save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
            torch.save(model.state_dict(), save_mode_path)
            logging.info("save model to {}".format(save_mode_path))
            iterator.close()
            break

    writer.close()
    return "Training Finished!"
from model import Model
import torch
import torch.nn.functional as F
from torch.utils.data import Sampler, BatchSampler

import os
import shutil
import time
import logging
import copy
import types
import importlib.machinery
import numpy as np
import h5py

from dataset import ShapeNetPartDataset, BalancedSampler
import hydra

data_path = "shapenet_part_seg_hdf5_data"
N_PARTS = 50
N_CATS = 16

seg_classes = {'Earphone': [16, 17, 18], 'Motorbike': [30, 31, 32, 33, 34, 35], 'Rocket': [41, 42, 43], 'Car': [8, 9, 10, 11], 'Laptop': [28, 29], 'Cap': [6, 7], 'Skateboard': [44, 45, 46], 'Mug': [36, 37], 'Guitar': [19, 20, 21], 'Bag': [4, 5], 'Lamp': [24, 25, 26, 27], 'Table': [47, 48, 49], 'Airplane': [0, 1, 2, 3], 'Pistol': [38, 39, 40], 'Chair': [12, 13, 14, 15], 'Knife': [22, 23]}
seg_label_to_cat = {} # {0:Airplane, 1:Airplane, ...49:Table}
for cat in seg_classes.keys():
    for label in seg_classes[cat]:
        seg_label_to_cat[label] = cat


def load_train_set(**kwargs):
    # load data
    f0 = h5py.File(hydra.utils.to_absolute_path(os.path.join(data_path, 'ply_data_train0.h5')))
    f1 = h5py.File(hydra.utils.to_absolute_path(os.path.join(data_path, 'ply_data_train1.h5')))
    f2 = h5py.File(hydra.utils.to_absolute_path(os.path.join(data_path, 'ply_data_train2.h5')))
    f3 = h5py.File(hydra.utils.to_absolute_path(os.path.join(data_path, 'ply_data_train3.h5')))
    f4 = h5py.File(hydra.utils.to_absolute_path(os.path.join(data_path, 'ply_data_train4.h5')))
    f5 = h5py.File(hydra.utils.to_absolute_path(os.path.join(data_path, 'ply_data_train5.h5')))
    f6 = h5py.File(hydra.utils.to_absolute_path(os.path.join(data_path, 'ply_data_val0.h5')))
    f = [f0, f1, f2, f3, f4, f5, f6]

    data = f[0]['data'][:]
    label = f[0]['label'][:]
    seg = f[0]['pid'][:]

    for i in range(1, 7):
        data = np.concatenate((data, f[i]['data'][:]), axis=0)
        label = np.concatenate((label, f[i]['label'][:]), axis=0)
        seg = np.concatenate((seg, f[i]['pid'][:]), axis=0)

    for ff in f:
        ff.close()

    print(data.shape, label.shape, seg.shape)
    return ShapeNetPartDataset(data, label, seg, **kwargs)


def load_test_set(**kwargs):
    # load data
    f0 = h5py.File(hydra.utils.to_absolute_path(os.path.join(data_path, 'ply_data_test0.h5')))
    f1 = h5py.File(hydra.utils.to_absolute_path(os.path.join(data_path, 'ply_data_test1.h5')))
    f = [f0, f1]

    data = f[0]['data'][:]
    label = f[0]['label'][:]
    seg = f[0]['pid'][:]

    for i in range(1, 2):
        data = np.concatenate((data, f[i]['data'][:]), axis=0)
        label = np.concatenate((label, f[i]['label'][:]), axis=0)
        seg = np.concatenate((seg, f[i]['pid'][:]), axis=0)

    for ff in f:
        ff.close()

    print(data.shape, label.shape, seg.shape)
    return ShapeNetPartDataset(data, label, seg, **kwargs)


import hydra
from multiprocessing import cpu_count
@hydra.main(config_path='config', config_name='shapenet')
def main(cfg):
    log_dir, batch_size, resume, num_workers = '.', cfg.batch_size, cfg.resume, cpu_count() // 2

    if not os.path.isdir(log_dir):
        os.mkdir(log_dir)

    logger = logging.getLogger(__name__)
    logger.info(cfg)
    torch.backends.cudnn.benchmark = True

    # Load the model
    model = Model(50, cfg)
    model.cuda()
    if resume > 0:
        model.load_state_dict(torch.load(os.path.join(log_dir, "state%d.pkl" % resume)))

    logger.info("{} paramerters in total".format(sum(x.numel() for x in model.parameters())))

    train_set = load_train_set(rand_rot=(cfg.rot[:2] == 'AR'), aug=True, bw=cfg.bw)
    test_set = load_test_set(rand_rot=(cfg.rot[2:] == 'AR'), aug=False, bw=cfg.bw)

    sampler = BatchSampler(BalancedSampler(train_set), batch_size, False)
    train_loader = torch.utils.data.DataLoader(train_set, batch_sampler=sampler, shuffle=False, num_workers=num_workers,
                                               pin_memory=True, drop_last=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=0)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=False, drop_last=False)

    def train_step(data, target_index, target, cat_onehot, backward=True):
        model.train()
        data, target_index, target = data.cuda(), target_index.cuda(), target.cuda()

        _, prediction = model(data, target_index, cat_onehot)

        prediction = prediction.view(-1, 50)
        target = target.view(-1)
        loss = F.nll_loss(prediction, target)

        if backward:
            optimizer.zero_grad()
            loss.backward()

            # torch.nn.utils.clip_grad_norm(model.parameters(), 1e-4)
            optimizer.step()

        correct = prediction.data.max(1)[1].eq(target.data).float().cpu().mean()

        # pred = prediction.data.argmax(1).cpu().numpy()
        # print(','.join([str(np.count_nonzero(pred == i)) for i in range(N_PARTS)]))
        return loss.item(), correct.item()

    def get_learning_rate(epoch):
        limits = [5, 10, 15, 20, 40]
        lrs = [0.01, 0.005, 0.001, 0.0005, 0.0001, 5e-5]

        assert len(lrs) == len(limits) + 1
        for lim, lr in zip(limits, lrs):
            if epoch < lim:
                return lr
        return lrs[-1]

    for epoch in range(resume, cfg.max_epoch):
        np.random.seed(epoch)
        lr = get_learning_rate(epoch)
        logger.info("learning rate = {} and batch size = {}".format(lr, cfg.batch_size))
        for p in optimizer.param_groups:
            p['lr'] = lr

        total_loss = 0
        total_correct = 0
        time_before_load = time.perf_counter()
        for batch_idx, (data, target_index, target, _, category) in enumerate(train_loader):
            # Transform category labels to one_hot.
            category_labels = torch.LongTensor(category)
            one_hot_labels = torch.zeros(category.size(0), 16).scatter_(1, category_labels, 1).cuda()

            time_after_load = time.perf_counter()
            time_before_step = time.perf_counter()
            loss, correct = train_step(data, target_index, target, one_hot_labels, True)

            total_loss += loss
            total_correct += correct

            logger.info("[{}:{}/{}] LOSS={:.2} <LOSS>={:.2} ACC={:.2} <ACC>={:.2} time={:.2}+{:.2}".format(
                epoch, batch_idx, len(train_loader),
                loss, total_loss / (batch_idx + 1),
                correct, total_correct / (batch_idx + 1),
                      time_after_load - time_before_load,
                      time.perf_counter() - time_before_step))
            time_before_load = time.perf_counter()

        # test
        model.eval()
        total_correct = 0
        mean_correct = 0
        shape_ious = {cat: [] for cat in seg_classes.keys()}
        for batch_idx, (data, target_index, target, pt_cloud, category) in enumerate(test_loader):
            model.eval()
            
            # Transform category labels to one_hot.
            category_labels = torch.LongTensor(category)
            one_hot_labels = torch.zeros(category.size(0), 16).scatter_(1, category_labels, 1).cuda()

            data, target_index, target = data.cuda(), target_index.cuda(), target.cuda()

            with torch.no_grad():
                _, prediction = model(data, target_index, one_hot_labels)

            prediction = prediction.view(-1, 2048, 50)

            target = target.view(-1, 2048)

            for j in range(target.size(0)):
                cat = seg_label_to_cat[target.cpu().numpy()[j][0]]
                prediction_np = prediction.cpu().numpy()[j][:, seg_classes[cat]].argmax(1) + seg_classes[cat][0]
                target_np = target.cpu().numpy()[j]
                correct = np.mean((prediction_np == target_np).astype(np.float32))

                total_correct += correct

                segp = prediction_np
                segl = target_np
                part_ious = [0.0 for _ in range(len(seg_classes[cat]))]
                for l in seg_classes[cat]:
                    if (np.sum(segl == l) == 0) and (
                            np.sum(segp == l) == 0):  # part is not present, no prediction as well
                        part_ious[l - seg_classes[cat][0]] = 1.0
                    else:
                        part_ious[l - seg_classes[cat][0]] = np.sum((segl == l) & (segp == l)) / float(
                            np.sum((segl == l) | (segp == l)))
                shape_ious[cat].append(np.mean(part_ious))

            mean_correct = total_correct / (batch_size * (batch_idx + 1))
            logger.info('Testing {}: acc {:.4f}'.format(batch_size * (batch_idx + 1), mean_correct))

        all_shape_ious = []
        for cat in shape_ious.keys():
            for iou in shape_ious[cat]:
                all_shape_ious.append(iou)
            shape_ious[cat] = np.mean(shape_ious[cat])
        logger.info('epoch %d test acc: %f, all shape mean IoU: %f, mean IoU: %f' % (
            epoch, mean_correct, np.mean(all_shape_ious), np.nanmean(list(shape_ious.values()))))

        torch.save(model.state_dict(), os.path.join(log_dir, "state%d.pkl" % epoch))



if __name__ == "__main__":
    main()

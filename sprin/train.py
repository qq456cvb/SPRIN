from model import SPRINCls, SPRINSeg
from dataset import ShapeNetPartDataset, seg_classes, seg_label_to_cat, BalancedSampler
import torch
import collections
import numpy as np
import contextlib

from dataset import read_data
import radam
import torch.nn.functional as F
import hydra
import logging
from multiprocessing import cpu_count

logger = logging.getLogger(__name__)

def worker_init_fn(worker_id):                                                          
    np.random.seed(torch.initial_seed() & 0xFFFFFFFF)


def run_epoch(net, data, segs_centered, segs, opt=None, epoch=1, ds=1024, batchsize=16, train=True, rand_rot=False):
    net.train(train)
    ds = ShapeNetPartDataset(data, segs_centered, segs, ds, rand_rot, aug=train)
    df = torch.utils.data.DataLoader(ds, batch_size=batchsize, shuffle=False, sampler=(BalancedSampler(ds) if train else None), num_workers=cpu_count() // 2, worker_init_fn=worker_init_fn)
    running_loss = 0.0
    running_acc = 0.0
    running_iou = 0.0
    shape_ious = collections.defaultdict(list)
    n = 0
    logger.info('Train' if train else 'Test')
    with contextlib.suppress() if train else torch.no_grad():
        for i, (pcs, segs_centered, segs, fps_idxs, cls_idxs) in enumerate(df):
            if train:
                opt.zero_grad()
            
            onehot_idxs = torch.eye(16)[cls_idxs]
            pred_batch = net(pcs.cuda().float(), fps_idxs, onehot_idxs.cuda().float())
            loss = F.cross_entropy(pred_batch, segs_centered.cuda().long())
            running_loss += loss.item()
            iou = []
            accs = []
            for pred, seg_centered, seg in zip(pred_batch.detach().cpu().numpy(), segs_centered.cpu().numpy(), segs):
                cat = seg_label_to_cat[seg[0].item()]
                pred = pred.argmax(0)
                acc = np.mean((pred == seg_centered).astype(np.float))
                part_ious = [0] * len(seg_classes[cat])
                for j in range(len(seg_classes[cat])):
                    if np.sum((pred == j) | (seg_centered == j)) == 0:
                        part_ious[j] = 1
                        continue
                    part_ious[j] = np.sum((pred == j) & (seg_centered == j)) / np.sum((pred == j) | (seg_centered == j))
                iou.append(np.mean(part_ious))
                accs.append(acc.item())
                shape_ious[cat].append(iou[-1])
            running_iou += np.mean(iou)
            running_acc += np.mean(accs)
            cmiou = np.mean(list(map(np.mean, shape_ious.values())))
            n += 1
            if train:
                loss.backward()
                opt.step()
            logger.info("[%s%d, %d] Loss: %.4f, Acc: %.4f, mIoU: %.4f, cmIoU: %.4f "
                % ("VT"[train], epoch, n,
                    running_loss / n, running_acc / n, running_iou / n, cmiou))


        for cat in shape_ious.keys():
            logger.info('{} IoU: {:.4f}'.format(cat, np.mean(shape_ious[cat])))

@hydra.main(config_path='config', config_name='shapenet')
def main(cfg):
    net = SPRINSeg(6, cfg.fps_n).cuda()
    if len(cfg.resume_path) > 0:
        net.load_state_dict(torch.load(hydra.utils.to_absolute_path(cfg.resume_path)))
    opt = radam.RAdam(net.parameters(), cfg.lr, weight_decay=cfg.weight_decay)
    pcs_train, segs_centered_train, segs_train = read_data(hydra.utils.to_absolute_path('shapenet_part_seg_hdf5_data'), r'ply_data_(train|val).*\.h5')
    pcs_test, segs_centered_test, segs_test = read_data(hydra.utils.to_absolute_path('shapenet_part_seg_hdf5_data'), r'ply_data_test.*\.h5')
    
    print(len(pcs_train))
    print(len(pcs_test))

    for e in range(1, cfg.max_epoch):
        run_epoch(net, pcs_train, segs_centered_train, segs_train, opt, e, ds=cfg.npoints, batchsize=cfg.batch_size)
        
        if e % 10 == 0:
            run_epoch(net, pcs_test, segs_centered_test, segs_test, opt, e, train=False, ds=cfg.npoints, batchsize=cfg.batch_size, rand_rot=True)
            torch.save(net.state_dict(), 'epoch{}.pt'.format(e))
            
if __name__ == '__main__':
    main()
    
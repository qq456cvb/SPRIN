import torch.utils.data
import torch

import os
import numpy as np
from model import Model
import omegaconf
from train import load_test_set
from dataset import ShapeNetPartDataset
from tqdm import tqdm
from multiprocessing import cpu_count

N_PARTS = 50
N_CATS = 16

seg_classes = {'Earphone': [16, 17, 18], 'Motorbike': [30, 31, 32, 33, 34, 35], 'Rocket': [41, 42, 43], 'Car': [8, 9, 10, 11], 'Laptop': [28, 29], 'Cap': [6, 7], 'Skateboard': [44, 45, 46], 'Mug': [36, 37], 'Guitar': [19, 20, 21], 'Bag': [4, 5], 'Lamp': [24, 25, 26, 27], 'Table': [47, 48, 49], 'Airplane': [0, 1, 2, 3], 'Pistol': [38, 39, 40], 'Chair': [12, 13, 14, 15], 'Knife': [22, 23]}
seg_label_to_cat = {} # {0:Airplane, 1:Airplane, ...49:Table}
for cat in seg_classes.keys():
    for label in seg_classes[cat]:
        seg_label_to_cat[label] = cat

color_map = {
    0: (0.65, 0.95, 0.05),
    1: (0.35, 0.05, 0.35),
    2: (0.65, 0.35, 0.65),
    3: (0.95, 0.95, 0.65),
    4: (0.95, 0.65, 0.05),
    5: (0.35, 0.05, 0.05),
    8: (0.05, 0.05, 0.65),
    9: (0.65, 0.05, 0.35),
    10: (0.05, 0.35, 0.35),
    11: (0.65, 0.65, 0.35),
    12: (0.35, 0.95, 0.05),
    13: (0.05, 0.35, 0.65),
    14: (0.95, 0.95, 0.35),
    15: (0.65, 0.65, 0.65),
    16: (0.95, 0.95, 0.05),
    17: (0.65, 0.35, 0.05),
    18: (0.35, 0.65, 0.05),
    19: (0.95, 0.65, 0.95),
    20: (0.95, 0.35, 0.65),
    21: (0.05, 0.65, 0.95),
    36: (0.05, 0.95, 0.05),
    37: (0.95, 0.65, 0.65),
    38: (0.35, 0.95, 0.95),
    39: (0.05, 0.95, 0.35),
    40: (0.95, 0.35, 0.05),
    47: (0.35, 0.05, 0.95),
    48: (0.35, 0.65, 0.95),
    49: (0.35, 0.05, 0.65)
}

import hydra
@hydra.main(config_path='config', config_name='shapenet')
def main(cfg):
    weight_path = hydra.utils.to_absolute_path('prin/state79.pkl')
    torch.backends.cudnn.benchmark = True

    model = Model(N_PARTS, cfg)
    model.cuda()
    model.load_state_dict(torch.load(weight_path))

    print("{} paramerters in total".format(sum(x.numel() for x in model.parameters())))

    test_set = load_test_set(rand_rot=True, aug=False, bw=cfg.bw)

    test_loader = torch.utils.data.DataLoader(test_set, batch_size=cfg.batch_size, shuffle=False, num_workers=cpu_count() // 2, drop_last=False)

    model.eval()

    # -------------------------------------------------------------------------------- #
    total_correct = 0
    shape_ious = {cat: [] for cat in seg_classes.keys()}
    for batch_idx, (data, target_index, target, pt_cloud, category) in enumerate(test_loader):

        # Transform category labels to one_hot.
        category_labels = torch.LongTensor(category)
        one_hot_labels = torch.zeros(category.size(0), 16).scatter_(1, category_labels, 1).cuda()

        data, target_index, target = data.cuda(), target_index.cuda(), target.cuda()

        # print (data.shape)
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
                if (np.sum(segl == l) == 0) and (np.sum(segp == l) == 0):  # part is not present, no prediction as well
                    part_ious[l - seg_classes[cat][0]] = 1.0
                else:
                    part_ious[l - seg_classes[cat][0]] = np.sum((segl == l) & (segp == l)) / float(
                        np.sum((segl == l) | (segp == l)))
            shape_ious[cat].append(np.mean(part_ious))

        print('acc: ', (batch_idx + 1) * cfg.batch_size, total_correct / (batch_idx + 1) / cfg.batch_size)

    print(total_correct / cfg.batch_size / len(test_loader))
    all_shape_ious = []
    for cat in shape_ious.keys():
        for iou in shape_ious[cat]:
            all_shape_ious.append(iou)
        shape_ious[cat] = np.mean(shape_ious[cat])
    print("all shape mIoU: {:.4f}, shape mIoU: {:.4f}".format(np.mean(all_shape_ious), np.nanmean(list(shape_ious.values()))))
    for cat in shape_ious.keys():
        print('{} IoU: {:.4f}'.format(cat, shape_ious[cat]))


if __name__ == "__main__":
    main()
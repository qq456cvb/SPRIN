import torch
import numpy as np

from dataset import read_data
from model import SPRINCls, SPRINSeg
import hydra
from train import run_epoch

@hydra.main(config_path='config', config_name='shapenet')
def main(cfg):
    net = SPRINSeg(6, cfg.fps_n).to("cuda")
    net.load_state_dict(torch.load(hydra.utils.to_absolute_path('sprin/epoch250.pt')))
    pcs_test, segs_centered_test, segs_test = read_data(hydra.utils.to_absolute_path("shapenet_part_seg_hdf5_data/ply_data_test*"))

    print(len(pcs_test))

    run_epoch(net, pcs_test, segs_centered_test, segs_test, None, 1, train=False, ds=cfg.npoints, batchsize=1)
    
    
if __name__ == '__main__':
    main()

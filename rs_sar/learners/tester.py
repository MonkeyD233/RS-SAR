"""Class to test a model"""
import json
import numpy as np
import os
import torch
from torch.utils.data import DataLoader
from torchvision.utils import make_grid

from rs_sar.utils.functions import transform_masks_viz, get_metrics, normalize, define_loss, get_transformations, get_qualitatives
from rs_sar.utils.paths import Paths
from rs_sar.utils.metrics import Evaluator
from rs_sar.loaders.dataloaders import RSSARDataset
# from rs_sar.utils.distributed_utils import reduce_value, get_rank, all_gather
# from rs_sar.utils.profiler import timings

class Tester:
    """
    Class to test a model

    PARAMETERS
    ----------
    cfg: dict
        Configuration parameters used for train/test
    visualizer: object or None
        Add a visulization during testing
        Default: None
    """

    def __init__(self, cfg, visualizer=None):
        self.cfg = cfg
        self.visualizer = visualizer
        self.model = self.cfg['model']
        self.dataset = self.cfg['dataset']
        self.nb_classes = self.cfg['nb_classes']
        # self.annot_type = self.cfg['annot_type']
        self.process_signal = self.cfg['process_signal']
        self.w_size = self.cfg['w_size']
        self.h_size = self.cfg['h_size']
        self.n_frames = self.cfg['nb_input_channels']
        self.batch_size = self.cfg['batch_size']
        self.device = self.cfg['device']
        self.custom_loss = self.cfg['custom_loss']
        self.transform_names = self.cfg['transformations'].split(',')
        self.norm_type = self.cfg['norm_type']
        # self.distributed = self.cfg['distributed']
        self.num_workers = self.cfg['num_workers']
        self.paths = Paths().get()
        self.test_results = dict()

    def predict(self, net, seq_loader, iteration=None, get_quali=False, add_temp=False, timer=False, mode='Test', iteration_loss=0):
        """
        Method to predict on a given dataset using a fixed model

        PARAMETERS
        ----------
        net: PyTorch Model
            Network to test
        seq_loader: DataLoader
            Specific to the dataset used for test
        iteration: int
            Iteration used to display visualization
            Default: None
        get_quali: boolean
            If you want to save qualitative results
            Default: False
        add_temp: boolean
            Is the data are considered as a sequence
            Default: False
        """
        net.eval()
        transformations = get_transformations(self.transform_names, split='test',
                                              sizes=(self.w_size, self.h_size))
        ra_criterion = define_loss(self.dataset, 'range_angle', self.custom_loss, self.device)
        nb_losses = len(ra_criterion)
        running_losses = list()
        ra_running_losses = list()
        ra_running_global_losses = [list(), list()]
        ra_metrics = Evaluator(num_class=self.nb_classes)
        if iteration:
            rand_seq = np.random.randint(len(seq_loader))
        with torch.no_grad():
            for i, sequence_data in enumerate(seq_loader):
                seq_name, seq = sequence_data
                if self.dataset == 'RS-SAR':
                    path_to_frames = self.paths['RS-SAR'] / seq_name[0]
                # elif self.dataset == 'KuRALS_PD':
                #     path_to_frames = self.paths['KuRALS_PD'] / seq_name[0]
                else:
                    raise KeyError(f'Dataset {self.dataset} has not been supported yet.')
                
                rs_sar_dataset = RSSARDataset(seq,
                                                        path_to_frames,
                                                        self.process_signal,
                                                        self.n_frames,
                                                        transformations,
                                                        add_temp)
                # if self.distributed:
                #     sampler_test = torch.utils.data.distributed.DistributedSampler(rs_sar_dataset, shuffle=False)
                # else:
                sampler_test = torch.utils.data.SequentialSampler(rs_sar_dataset)
                # lt@20240301 for pdr dataset, fix batch size to 2.
                if self.dataset == 'RS-SAR':
                    frame_dataloader = DataLoader(rs_sar_dataset,
                                              shuffle=False,
                                              sampler=sampler_test,
                                              batch_size=self.batch_size,
                                              num_workers=self.num_workers)
                # elif self.dataset == 'KuRALS_PD':
                #     frame_dataloader = DataLoader(kurals_dataset,
                #                               shuffle=False,
                #                               sampler=sampler_test,
                #                               batch_size=1,
                #                               num_workers=0)
                else:
                    raise KeyError(f'Dataset {self.dataset} has not been supported yet.')
                
                if iteration and i == rand_seq:
                    rand_frame = np.random.randint(len(frame_dataloader))
                if get_quali:
                    quali_iter_rd = self.n_frames-1
                for j, frame in enumerate(frame_dataloader):
                    ra_data = frame['ra_matrix'].to(self.device).float()
                    ra_mask = frame['ra_mask'].to(self.device).float()
                    ra_data = normalize(ra_data, self.dataset, 'range_angle', norm_type=self.norm_type)
                    
                    ra_outputs = net(ra_data)
                    
                    ra_outputs = ra_outputs.to(self.device)

                    if get_quali:
                        quali_iter_rd = get_qualitatives(ra_outputs, ra_mask, self.paths,
                                                         seq_name, quali_iter_rd, 'range_angle',
                                                         self.dataset)

                    ra_metrics.add_batch_dist(torch.argmax(ra_mask, axis=1),
                                         torch.argmax(ra_outputs, axis=1))

                    ra_losses = [c(ra_outputs, torch.argmax(ra_mask, axis=1))
                                    for c in ra_criterion]
                    ra_loss = torch.mean(torch.stack(ra_losses))

                    loss = ra_loss
                    ra_losses = ra_losses
                    ra_loss = ra_loss
                    loss = loss
                    
                    running_losses.append(loss.data.cpu().numpy()[()])
                    ra_running_losses.append(ra_loss.data.cpu().numpy()[()])
                    ra_running_global_losses[0].append(ra_losses[0].data.cpu().numpy()[()])
                    ra_running_global_losses[1].append(ra_losses[1].data.cpu().numpy()[()])
                
                    if iteration and i == rand_seq:
                        if j == rand_frame:
                            ra_pred_masks = torch.argmax(ra_outputs, axis=1)[:5]
                            ra_gt_masks = torch.argmax(ra_mask, axis=1)[:5]
                            ra_pred_grid = make_grid(transform_masks_viz(ra_pred_masks,
                                                                         self.nb_classes))
                            ra_gt_grid = make_grid(transform_masks_viz(ra_gt_masks,
                                                                       self.nb_classes))
                            if self.visualizer:
                                self.visualizer.update_multi_img_masks(ra_pred_grid, ra_gt_grid,
                                                                   iteration)
            self.test_results = dict()
            self.test_results['range_angle'] = get_metrics(ra_metrics, np.mean(ra_running_losses),
                                                             [np.mean(sub_loss) for sub_loss
                                                              in ra_running_global_losses])
            self.test_results['global_acc'] = self.test_results['range_angle']['acc']
            self.test_results['global_prec'] = self.test_results['range_angle']['prec']
            self.test_results['global_dice'] = self.test_results['range_angle']['dice']
            # add global recall and mIoU zlw@20220704
            self.test_results['global_recall'] = self.test_results['range_angle']['recall']
            self.test_results['global_miou'] = self.test_results['range_angle']['miou']

            ra_metrics.reset()
            
        return self.test_results

    def write_params(self, path):
        """Write quantitative results of the Test"""
        with open(path, 'w') as fp:
            json.dump(self.test_results, fp)

    def set_device(self, device):
        """Set device used for test (supported: 'cuda', 'cpu')"""
        self.device = device

    def set_annot_type(self, annot_type):
        """Set annotation type to test on (specific to CARRADA)"""
        self.annot_type = annot_type

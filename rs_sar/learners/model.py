"""Class to train a PyTorch model"""
import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ExponentialLR, CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from functools import partial
import random
import time

from rs_sar.loaders.dataloaders import RSSARDataset
from rs_sar.learners.tester import Tester
from rs_sar.utils.functions import normalize, define_loss, get_transformations
from rs_sar.utils.tensorboard_visualizer import TensorboardMultiLossVisualizer
# from rs_sar.utils.distributed_utils import get_rank, reduce_value


class Model(nn.Module):
    """Class to train a model

    PARAMETERS
    ----------
    net: PyTorch Model
        Network to train
    data: dict
        Parameters and configurations for training
    """

    def __init__(self, net, data):
        super().__init__()
        self.net = net
        self.cfg = data['cfg']
        self.paths = data['paths']
        self.dataloaders = data['dataloaders']
        self.resume_path = self.cfg['resume_path']
        self.model_name = self.cfg['model']
        self.dataset = self.cfg['dataset']
        self.process_signal = self.cfg['process_signal']
        # self.annot_type = self.cfg['annot_type']
        self.w_size = self.cfg['w_size']
        self.h_size = self.cfg['h_size']
        self.batch_size = self.cfg['batch_size']
        self.nb_epochs = self.cfg['nb_epochs']
        self.lr = self.cfg['lr']
        self.lr_step = self.cfg['lr_step']
        self.schedular_type = self.cfg['schedular']
        self.T_max = self.cfg['Tmax']
        self.loss_step = self.cfg['loss_step']
        self.val_step = self.cfg['val_step']
        self.viz_step = self.cfg['viz_step']
        self.torch_seed = self.cfg['torch_seed']
        self.numpy_seed = self.cfg['numpy_seed']
        self.nb_classes = self.cfg['nb_classes']
        self.custom_loss = self.cfg['custom_loss']
        self.comments = self.cfg['comments']
        self.n_frames = self.cfg['nb_input_channels']
        self.transform_names = self.cfg['transformations'].split(',')
        self.norm_type = self.cfg['norm_type']
        self.is_shuffled = self.cfg['shuffle']
        self.device = self.cfg['device']
        # self.distributed = self.cfg['distributed']
        self.num_workers = self.cfg['num_workers']
        self.rank = 0
        
        self.writer = SummaryWriter(self.paths['writer'])
        self.visualizer = TensorboardMultiLossVisualizer(self.writer)
        self.tester = Tester(self.cfg, self.visualizer)
        
        self.results = dict()

    def train(self, add_temp=False):
        """
        Method to train a network

        PARAMETERS
        ----------
        add_temp: boolean
            Add a temporal dimension during training?
            Considering the input as a sequence.
            Default: False
        """
        if self.rank == 0:
            self.writer.add_text('Comments', self.comments)
        train_loader, val_loader, test_loader = self.dataloaders
        transformations = get_transformations(self.transform_names,
                                              sizes=(self.w_size, self.h_size))
        ra_criterion = define_loss(self.dataset, 'range_angle', self.custom_loss, self.device)
        nb_losses = len(ra_criterion)
        running_losses = list()
        ra_running_losses = list()
        ra_running_global_losses = [list(), list()]
        optimizer = optim.Adam(self.net.parameters(), lr=self.lr)
        # lt @20230417
        if self.schedular_type == 'exp':
            scheduler = ExponentialLR(optimizer, gamma=0.9)
        elif self.schedular_type == 'cos':
            scheduler = CosineAnnealingLR(optimizer, T_max=self.T_max-1, eta_min=1e-7)
        else:
            raise KeyError("we only implement schedular of exp and cos")
        
        start_epoch = 0
        iteration = 0
        best_val_prec = 0
        best_val_recall = 0
        # lt @20230522
        flag_save = False
        name_result = []
        best_val_dop = 0
        best_val_ang = 0
        best_val_dop_plus_ang = 0
        best_test_dop = 0
        best_test_ang = 0
        best_test_dop_plus_ang = 0
        # zlw@20211012
        # if torch.cuda.device_count() > 1:
        #     self.net = nn.DataParallel(self.net, device_ids=[2, 3])
            #self.net = nn.DataParallel(self.net, device_ids=[3])
        # self.net.to(self.device)


        if self.resume_path:
            checkpoint = torch.load(self.resume_path, map_location=torch.device('cpu'))
            start_epoch = checkpoint['epoch']
            start_dict = checkpoint['state_dict']
            new_state_dict = {}
            for k, v in start_dict.items():
                name = k[7:] if k.startswith('module.') else k
                new_state_dict[name] = v

            if hasattr(self.net, 'module'):
                self.net.module.load_state_dict(new_state_dict)
            else:
                self.net.load_state_dict(new_state_dict)

            optimizer.load_state_dict(checkpoint['optimizer'])
            if 'scheduler' in checkpoint:
                scheduler.load_state_dict(checkpoint['scheduler'])
            else:
                for e in range(1, start_epoch):
                    print(f"There is no scheduler state in the checkpoint, fast-forwarding scheduler to epoch {e}.")
                    if self.schedular_type == 'exp' and e % self.lr_step == 0:
                        scheduler.step()
                    elif self.schedular_type == 'cos' and e < self.T_max:
                        scheduler.step()
            
            best_val_ang = checkpoint['best_val_ang']
            best_test_ang = checkpoint['best_test_ang']

            if 'iteration' in checkpoint:
                iteration = checkpoint['iteration']
                print('Resuming training from epoch {}, iteration {}'.format(start_epoch, iteration))
            else:
                steps_per_epoch = len(train_loader)
                iteration = (start_epoch - 1) * steps_per_epoch
                print(f"[Warning] Old checkpoint detected, which does not contain iteration information. Resuming training from epoch {start_epoch}, iteration {iteration}.")
                print(f"Estimated iteration: {iteration}.")
            print(f"=> Loaded checkpoint at epoch {start_epoch}, resuming from iteration {iteration}")

            print("=> Loaded checkpoint at epoch {} w/ best dice in valuation: {}, best dice in testing: {})".format(
                checkpoint['epoch'], best_val_ang, best_test_ang))




        for epoch in range(start_epoch, self.nb_epochs):
            # lt @20230417
            if self.schedular_type == 'exp':
                if epoch % self.lr_step == 0 and epoch != 0:
                    scheduler.step()
            else:
                if epoch != 0 and epoch < self.T_max:
                    scheduler.step()
            for _, sequence_data in enumerate(train_loader):
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
                #     sampler_train = torch.utils.data.distributed.DistributedSampler(kurals_dataset)
                #     sampler_train.set_epoch(epoch)
                # else:
                sampler_train = torch.utils.data.RandomSampler(rs_sar_dataset)
                train_batch_sampler = torch.utils.data.BatchSampler(sampler_train, self.batch_size, drop_last=True)
                frame_dataloader = DataLoader(rs_sar_dataset,
                                              batch_sampler=train_batch_sampler,
                                              num_workers=self.num_workers,
                                              worker_init_fn=partial(worker_init_fn, rank=0, seed=self.numpy_seed))
                for _, frame in enumerate(frame_dataloader):
                    ra_data = frame['ra_matrix'].to(self.device).float()
                    ra_mask = frame['ra_mask'].to(self.device).float()
                    ra_data = normalize(ra_data, self.dataset, 'range_angle', norm_type=self.norm_type)
                    
                    optimizer.zero_grad()

                    ra_outputs = self.net(ra_data)
                    ra_outputs = ra_outputs.to(self.device)

                    ra_losses = [c(ra_outputs, torch.argmax(ra_mask, axis=1))
                                for c in ra_criterion]
                    ra_loss = torch.mean(torch.stack(ra_losses))

                    loss = ra_loss
                    ra_losses_reduced = ra_losses
                    ra_loss_reduced = ra_loss
                    loss_reduced = loss
                    
                    loss.backward()
                    optimizer.step()
            
                    if self.rank == 0:
                        running_losses.append(loss_reduced.data.cpu().numpy()[()])
                        ra_running_losses.append(ra_loss_reduced.data.cpu().numpy()[()])
                        ra_running_global_losses[0].append(ra_losses_reduced[0].data.cpu().numpy()[()])
                        ra_running_global_losses[1].append(ra_losses_reduced[1].data.cpu().numpy()[()])

                        if iteration % self.loss_step == 0:
                            train_loss = np.mean(running_losses)
                            ra_train_loss = np.mean(ra_running_losses)
                            ra_train_losses = [np.mean(sub_loss) for sub_loss in ra_running_global_losses]
                            # zlw@20220302
                            print('[{}][Epoch {}/{}, iter {}]: '
                                'Train loss {}'.format(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()),
                                                        epoch+1,
                                                        self.nb_epochs,
                                                        iteration,
                                                        train_loss))
                            print('[Epoch {}/{}, iter {}]: '
                                'Train losses: RA={}'.format(epoch+1,
                                                                    self.nb_epochs,
                                                                    iteration,
                                                                    ra_train_loss))
                            
                    
                            self.visualizer.update_multi_train_loss(train_loss, ra_train_loss,
                                                                        ra_train_losses, iteration)
                            running_losses = list()
                            ra_running_losses = list()
                            # zlw@20220107 get_lr() --> get_last_lr()
                            self.visualizer.update_learning_rate(scheduler.get_last_lr()[0], iteration)

                    if iteration % self.val_step == 0 and iteration > 0:
                        if iteration % self.viz_step == 0 and iteration > 0:
                            val_metrics = self.tester.predict(self.net, val_loader, iteration,
                                                              add_temp=add_temp, mode='Validation', iteration_loss=iteration)
                        else:
                            val_metrics = self.tester.predict(self.net, val_loader, add_temp=add_temp, mode='Validation', iteration_loss=iteration)

                        if self.rank == 0:
                            self.visualizer.update_multi_val_metrics(val_metrics, iteration)
                            print('[Epoch {}/{}] Val losses: '
                                'RA={}'.format(epoch+1,
                                                        self.nb_epochs,
                                                        val_metrics['range_angle']['loss'],
                                                        ))
                            
                            print('[Epoch {}/{}] Val Pixel Prec: '
                                'RA={}'.format(epoch+1,
                                                        self.nb_epochs,
                                                        val_metrics['range_angle']['prec'],
                                                        ))
                            # zlw@20220227
                            print('[Epoch {}/{}] Val mIoU: '
                                'RA={}'.format(epoch+1,
                                                        self.nb_epochs,
                                                        val_metrics['range_angle']['miou'],
                                                        ))
                            print('[Epoch {}/{}] Val Dice: '
                                'RA={}'.format(epoch+1,
                                                        self.nb_epochs,
                                                        val_metrics['range_angle']['dice'],
                                                        ))

                            # lt @20230522
                            if val_metrics['range_angle']['dice'] > best_val_ang and iteration > 0:
                                best_val_ang = val_metrics['range_angle']['dice']
                                flag_save = True
                                name_result.append('val_angle')
                            
                        # saving results w/ best dice zlw@20220704
                        test_metrics = self.tester.predict(self.net, test_loader,
                                                               add_temp=add_temp, mode='Test', iteration_loss=iteration)
                        if self.rank == 0:
                            self.visualizer.update_multi_test_metrics(test_metrics, iteration)
                            print('[Epoch {}/{}] Test losses: '
                                'RA={}'.format(epoch+1,
                                                        self.nb_epochs,
                                                        test_metrics['range_angle']['loss']))
                            print('[Epoch {}/{}] Test Prec: '
                                'RA={}'.format(epoch+1,
                                                        self.nb_epochs,
                                                        test_metrics['range_angle']['prec'],
                                                        ))
                            # zlw@20220227
                            print('[Epoch {}/{}] Test mIoU: '
                                'RA={}'.format(epoch+1,
                                                        self.nb_epochs,
                                                        test_metrics['range_angle']['miou'],
                                                        ))
                            print('[Epoch {}/{}] Test Dice: '
                                'RA={}'.format(epoch+1,
                                                        self.nb_epochs,
                                                        test_metrics['range_angle']['dice'],
                                                        ))

                            self.results['epoch'] = epoch + 1
                            self.results['ra_train_loss'] = ra_train_loss.item()
                            self.results['train_loss'] = train_loss.item()
                            self.results['val_metrics'] = val_metrics
                            self.results['test_metrics'] = test_metrics


                            if test_metrics['range_angle']['dice'] > best_test_ang and iteration > 0:
                                best_test_ang = test_metrics['range_angle']['dice']
                                flag_save = True
                                name_result.append('test_angle')
                            
                            if flag_save:
                                self._save_results(name_result)
                                flag_save = False
                                name_result = []
                        self.net.train()  # Train mode after evaluation process
                    iteration += 1

            if self.rank == 0:
                model_state = self.net.module.state_dict() if hasattr(self.net, 'module') else self.net.state_dict()
                arch_name = str(self.net.module) if hasattr(self.net, 'module') else str(self.net)

                state = {
                    'epoch': epoch + 1,
                    'arch': arch_name,
                    'state_dict': model_state,
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'best_val_ang': best_val_ang,
                    'best_test_ang': best_test_ang,
                    'iteration': iteration,
                }

                self._save_latest(['latest'], state)
                print(f"Epoch {epoch+1} completed. Latest checkpoint saved (overwritten).")

        if self.rank == 0:
            self.writer.close()

    def _save_results(self, names, state=None):
        for name in names:
            results_path = self.paths['results'] / (name + '_' + 'results.json')
            model_path = self.paths['results'] / (name + '_' + 'model.pt')
            with open(results_path, "w") as fp:
                json.dump(self.results, fp)
            torch.save(self.net.state_dict(), model_path)

    def _save_latest(self, names, state):
        for name in names:
            if self.rank == 0:
                results_path = self.paths['results'] / (name + '_' + 'results.json')
                with open(results_path, "w") as fp:
                    json.dump(self.results, fp)
                model_path = self.paths['results'] / (name + '_' + 'model.pt')
                torch.save(state, model_path)

def worker_init_fn(worker_id, rank, seed):
    worker_seed = rank + seed
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)
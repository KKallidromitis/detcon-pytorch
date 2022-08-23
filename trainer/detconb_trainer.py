#-*- coding:utf-8 -*-
import os
import time
import datetime

import numpy as np
import torch
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

from tensorboardX import SummaryWriter
import apex
import wandb
from apex.parallel import DistributedDataParallel as DDP
from apex import amp

from model import DetconBModel
from optimizer import LARS
from data import ImageLoader,ImageLoadeCOCO
from utils import distributed_utils, params_util, logging_util, eval_util
from utils.data_prefetcher import data_prefetcher
from losses import DetconBInfoNCECriterion

class DetconBTrainer():
    def __init__(self, config):
        self.config = config
        
        """set seed"""
        distributed_utils.set_seed(self.config['seed'])
        
        """device parameters"""
        self.world_size = self.config['world_size']
        self.rank = self.config['rank']
        self.gpu = self.config['local_rank']
        self.distributed = self.config['distributed']

        """get the train parameters!"""
        self.total_epochs = self.config['optimizer']['total_epochs']
        self.warmup_epochs = self.config['optimizer']['warmup_epochs']

        self.train_batch_size = self.config['data']['train_batch_size']
        self.val_batch_size = self.config['data']['val_batch_size']
        self.global_batch_size = self.world_size * self.train_batch_size

        self.num_examples = self.config['data']['num_examples']
        self.warmup_steps = self.warmup_epochs * self.num_examples // self.global_batch_size
        self.total_steps = self.total_epochs * self.num_examples // self.global_batch_size

        base_lr = self.config['optimizer']['base_lr'] / 256
        self.max_lr = base_lr * self.global_batch_size
        self.lr_type = self.config['optimizer']['lr_type']

        self.base_mm = self.config['model']['base_momentum']
        self.forward_loss = DetconBInfoNCECriterion(config)
        
        """construct the whole network"""
        self.resume_path = self.config['checkpoint']['resume_path']
        if torch.cuda.is_available():
            self.device = torch.device(f'cuda:{self.gpu}')
            torch.cuda.set_device(self.device)
            cudnn.benchmark = True
        else:
            self.device = torch.device('cpu')
        self.construct_model()

        """save checkpoint path"""
        self.time_stamp = self.config['checkpoint']['time_stamp']
        if self.time_stamp == None:
            self.time_stamp = datetime.datetime.now().strftime('%m_%d_%H-%M')
            
        self.save_epoch = self.config['checkpoint']['save_epoch']
        self.ckpt_path = self.config['checkpoint']['ckpt_path'].format(
            self.time_stamp,self.time_stamp, self.config['model']['backbone']['type'], {})

        save_dir = '/'.join(self.ckpt_path.split('/')[:-1])
        self.log_all = self.config['log']['log_all']
        
        if self.gpu==0 or self.log_all:
            wandb.init(project="detcon_byol",name = save_dir+'_gpu_'+str(self.rank))
        
        try:
            os.makedirs(save_dir)
        except:
            pass

        """log tools in the running phase"""
        self.steps = 0
        self.wandb_id = self.config['log']['wandb_id']
        self.epoch_count = -1
        self.log_step = self.config['log']['log_step']
        self.logging = logging_util.get_std_logging()
        if self.rank == 0:
            self.writer = SummaryWriter(self.config['log']['log_dir'])

    def construct_model(self):
        """get data loader"""
        self.stage = self.config['stage']
        assert self.stage == 'train', ValueError(f'Invalid stage: {self.stage}, only "train" for DetconBYOL training')
        if self.config['data']['mask_type'] == 'coco':
            print("DEBUG: Using Coco GT Mask")
            self.data_ins = ImageLoadeCOCO(self.config)
        else:
            self.data_ins = ImageLoader(self.config)
        self.train_loader = self.data_ins.get_loader(self.stage, self.train_batch_size)

        self.sync_bn = self.config['amp']['sync_bn']
        self.opt_level = self.config['amp']['opt_level']
        print(f"sync_bn: {self.sync_bn}")

        """build model"""
        print("init DetconB model!")
        net = DetconBModel(self.config)
        if self.sync_bn:
            net = apex.parallel.convert_syncbn_model(net)
        self.model = net.to(self.device)
        print("init DetconB model end!")

        """build optimizer"""
        print("get optimizer!")
        momentum = self.config['optimizer']['momentum']
        weight_decay = self.config['optimizer']['weight_decay']
        exclude_bias_and_bn = self.config['optimizer']['exclude_bias_and_bn']
        params = params_util.collect_params([self.model.online_network, self.model.predictor],
                                            exclude_bias_and_bn=exclude_bias_and_bn)
        self.optimizer = LARS(params, lr=self.max_lr, momentum=momentum, weight_decay=weight_decay)

        """init amp"""
        print("amp init!")
        self.model, self.optimizer = amp.initialize(
            self.model, self.optimizer, opt_level=self.opt_level)

        if self.distributed:
            self.model = DDP(self.model, delay_allreduce=True)
        print("amp init end!")

    # resume snapshots from pre-train
    def resume_model(self, model_path=None):
        if model_path is None and not self.resume_path:
            self.start_epoch = 0
            self.logging.info("--> No loaded checkpoint!")
        else:
            model_path = model_path or self.resume_path
            checkpoint = torch.load(model_path, map_location=self.device)

            self.start_epoch = checkpoint['epoch']
            self.steps = checkpoint['steps']
            self.model.load_state_dict(checkpoint['model'], strict=True)
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            amp.load_state_dict(checkpoint['amp'])
            self.logging.info(f"--> Loaded checkpoint '{model_path}' (epoch {self.start_epoch})")

    # save snapshots
    def save_checkpoint(self, epoch):
        if epoch % self.save_epoch == 0 and self.rank == 0:
            state = {'config': self.config,
                     'epoch': epoch,
                     'steps': self.steps,
                     'model': self.model.state_dict(),
                     'optimizer': self.optimizer.state_dict(),
                     'amp': amp.state_dict()
                    }
            torch.save(state, self.ckpt_path.format(epoch))

    def adjust_learning_rate(self, step):
        """learning rate warm up and decay"""
        max_lr = self.max_lr
        min_lr = 1e-3 * self.max_lr
        
        if step < self.warmup_steps:
            lr =  step / int(self.warmup_steps) * max_lr #Following deepmind implementation, returns lr = 0. during first step!
                    
        elif self.lr_type=='piecewise':
            if step >= (0.96*self.total_steps):
                lr = self.max_lr/10 
            elif step >= (0.98*self.total_steps):
                lr = self.max_lr/100
            else:
                lr = self.max_lr
                
        elif self.lr_type=='cosine': # For lr from detcon paper, returns lr as smalls ~1e-8
            max_steps = self.total_steps - self.warmup_steps
            global_step = np.minimum((step - self.warmup_steps), max_steps)
            cosine_decay_value = 0.5 * (1 + np.cos(np.pi * global_step / max_steps))
            lr = max_lr * cosine_decay_value
                 
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def adjust_mm(self, step):
        self.mm = 1 - (1 - self.base_mm) * (np.cos(np.pi * step / self.total_steps) + 1) / 2
        
    def train_epoch(self, epoch, printer=print):
        batch_time = eval_util.AverageMeter()
        data_time = eval_util.AverageMeter()
        forward_time = eval_util.AverageMeter()
        backward_time = eval_util.AverageMeter()
        log_time = eval_util.AverageMeter()
        loss_meter = eval_util.AverageMeter()

        self.model.train()

        end = time.time()
        self.data_ins.set_epoch(epoch)

        prefetcher = data_prefetcher(self.train_loader)
        images, masks = prefetcher.next()
        i = 0
        while images is not None:
            i += 1
            self.adjust_learning_rate(self.steps)
            self.adjust_mm(self.steps)
            self.steps += 1
            #import ipdb;ipdb.set_trace()
            assert images.dim() == 5, f"Input must have 5 dims, got: {images.dim()}"
            view1 = images[:, 0, ...].contiguous()
            view2 = images[:, 1, ...].contiguous()
            
            # measure data loading time
            data_time.update(time.time() - end)
            
            wandb_id = None
            if self.gpu==0 and self.epoch_count<epoch:
                if isinstance(self.wandb_id, int):
                    wandb_id = int(self.wandb_id)
                elif self.wandb_id=='random':
                    wandb_id = torch.randint(0,self.train_batch_size,(1,1)).item()
                self.epoch_count = epoch     
                
            # forward
            tflag = time.time()
            q, target_z,pinds, tinds = self.model(view1, view2, self.mm, masks.to('cuda'),wandb_id)
            forward_time.update(time.time() - tflag)

            tflag = time.time()
            loss = self.forward_loss(target_z, q, tinds.to('cuda'), pinds.to('cuda'))

            self.optimizer.zero_grad()
            if self.opt_level == 'O0':
                loss.backward()
            else:
                with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                    scaled_loss.backward()
            self.optimizer.step()
            backward_time.update(time.time() - tflag)
            loss_meter.update(loss.item(), view1.size(0))

            tflag = time.time()
            if self.steps % self.log_step == 0 and self.rank == 0:
                self.writer.add_scalar('lr', round(self.optimizer.param_groups[0]['lr'], 5), self.steps)
                self.writer.add_scalar('mm', round(self.mm, 5), self.steps)
                self.writer.add_scalar('loss', loss_meter.val, self.steps)
            log_time.update(time.time() - tflag)

            batch_time.update(time.time() - end)
            end = time.time()
            #import ipdb;ipdb.set_trace()
            # Print log info
            if (self.gpu == 0 or self.log_all) and self.steps % self.log_step == 0:
                
                # Log per batch stats to wandb (average per epoch is also logged at the end of function)
                wandb.log({
                    'lr': round(self.optimizer.param_groups[0]["lr"], 5),
                    'mm': round(self.mm, 5),
                    'loss': round(loss_meter.val, 5),
                    'Batch Time': round(batch_time.val, 5),
                    'Data Time': round(data_time.val, 5),
                    'Forward Time': round(forward_time.val, 5),
                    'Backward Time': round(backward_time.val, 5),
                })

                printer(f'Epoch: [{epoch}][{i}/{len(self.train_loader)}]\t'
                        f'Step {self.steps}\t'
                        f'lr {round(self.optimizer.param_groups[0]["lr"], 5)}\t'
                        f'mm {round(self.mm, 5)}\t'
                        f'Loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t'
                        f'Batch Time {batch_time.val:.4f} ({batch_time.avg:.4f})\t'
                        f'Data Time {data_time.val:.4f} ({data_time.avg:.4f})\t'
                        f'Forward Time {forward_time.val:.4f} ({forward_time.avg:.4f})\t'
                        f'Backward Time {backward_time.val:.4f} ({backward_time.avg:.4f})\t'
                        f'Log Time {log_time.val:.4f} ({log_time.avg:.4f})\t')

            images, masks = prefetcher.next()
        if self.gpu == 0 or self.log_all: 
            # Log averages at end of Epoch
            wandb.log({
                'Average Loss (Per-Epoch)': round(loss_meter.avg, 5),
                'Average Batch-Time (Per-Epoch)': round(batch_time.avg, 5),
                'Average Data-Time (Per-Epoch)': round(data_time.avg, 5),
                'Average Forward-Time (Per-Epoch)': round(forward_time.avg, 5),
                'Average Backward-Time (Per Epoch)': round(backward_time.avg, 5),
            })


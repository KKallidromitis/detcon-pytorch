#-*- coding:utf-8 -*-
import torch
from .basic_modules import EncoderwithProjection, Predictor
from utils.mask_utils import convert_binary_mask

class DetconBModel(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.pool_size = config['loss']['pool_size']
        self.train_batch_size = config['data']['train_batch_size']
        
        # online network
        self.online_network = EncoderwithProjection(config)

        # target network
        self.target_network = EncoderwithProjection(config)
        
        # predictor
        self.predictor = Predictor(config)

        self._initializes_target_network()

    @torch.no_grad()
    def _initializes_target_network(self):
        for param_q, param_k in zip(self.online_network.parameters(), self.target_network.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False     # not update by gradient

    @torch.no_grad()
    def _update_target_network(self, mm):
        """Momentum update of target network"""
        for param_q, param_k in zip(self.online_network.parameters(), self.target_network.parameters()):
            param_k.data.mul_(mm).add_(1. - mm, param_q.data)

    def forward(self, view1, view2, mm, masks):
        # online network forward
        #import ipdb;ipdb.set_trace()
        
        masks = torch.cat([ masks[:,i,:,:,:] for i in range(masks.shape[1])])
            
        masks = convert_binary_mask(masks,pool_size = self.pool_size)
        q,pinds = self.predictor(*self.online_network(torch.cat([view1, view2], dim=0),masks,'online'))

        # target network forward
        with torch.no_grad():
            self._update_target_network(mm)
            target_z, tinds = self.target_network(torch.cat([view1, view2], dim=0),masks,'target')
            target_z = target_z.detach().clone()

        return q, target_z, pinds, tinds

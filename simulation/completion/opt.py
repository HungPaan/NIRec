import numpy as np
import torch
from torch import nn
from model import mf

class opter(object):
    def __init__(self, config):
        self.num_users = config['num_users']
        self.num_items = config['num_items']
        self.device = config['device']
        self.is_tensorboard = config['is_tensorboard']
        self.loss_type = config['loss_type']
        self.ips_type = config['ips_type']
        self.propensity_per_item = torch.from_numpy(config['propensity_per_item']).to(self.device)
        #self.propensity_per_item = torch.abs(torch.randn(self.num_items)).pow(2).to(self.device)
        #self.propensity_per_item[self.propensity_per_item == 0.0] = 1.0
        print('=================================opter=======================================')
        print('opter init')

        print('num_users', self.num_users)
        print('num_items', self.num_items)
        print('device', self.device)
        print('is_tensorboard', self.is_tensorboard)
        print('loss_type', self.loss_type)
        print('ips_type', self.ips_type)
        print('propensity_per_item', self.propensity_per_item)

        self.emb_model_name = config['emb_model_name']
        print('emb_model_name', self.emb_model_name)
        self.emb_model = mf(config)

        self.emb_lr = config['emb_lr']
        self.emb_decay = config['emb_decay']
        self.emb_model.to(self.device)
        self.emb_optimizer = torch.optim.Adam(self.emb_model.parameters(), lr=self.emb_lr, weight_decay=self.emb_decay)

        #self.ps_user_emb = torch.from_numpy(np.loadtxt(f"../para/{config['data_name']}_ps_user_emb.txt")).float().to(self.device)
        #self.ps_item_emb = torch.from_numpy(np.loadtxt(f"../para/{config['data_name']}_ps_item_emb.txt")).float().to(self.device)

        print('emb_lr', self.emb_lr)
        print('emb_decay', self.emb_decay)
        #print('ps_user_emb', self.ps_user_emb, self.ps_user_emb.shape)
        #print('ps_item_emb', self.ps_item_emb, self.ps_item_emb.shape)

        print('=================================opter=======================================')


    def update_and_log(self, train_user_tensor, train_item_tensor, train_label_tensor, epoch, tb_log):
        if self.loss_type == 'mse':
            emb_loss = self.update_emb_model_mse(train_user_tensor, train_item_tensor, train_label_tensor)
        elif self.loss_type == 'mat_mse':
            emb_loss = self.update_emb_model_mat_mse(train_label_tensor)


        else:
            pass

        if (epoch % 25 == 0 or epoch == 1) and self.is_tensorboard:
            tb_log.add_scalar('Loss/emb_loss', emb_loss, epoch)

        return emb_loss



    def update_emb_model_mse(self, train_user_tensor, train_item_tensor, train_label_tensor):
        if self.ips_type == 'pop':
            ps = self.propensity_per_item[train_item_tensor.long()]
        elif self.ips_type == 'logi':
            ps = (self.ps_user_emb[train_user_tensor] * self.ps_item_emb[train_item_tensor]).sum(dim=1)
            ps = torch.sigmoid(ps)
        elif self.ips_type == 'none':
            ps = torch.ones(len(train_user_tensor), device=self.device)

        #ps = torch.ones(len(train_user_tensor), device=self.device)
        ips = 1.0 / ps
        out_train_rating = self.emb_model.out_forward(train_user_tensor, train_item_tensor)
        loss_per_ui = nn.functional.mse_loss(out_train_rating, train_label_tensor, reduction='none')  # 不降维
        #print('ips', ips, ips.shape)
        #print('loss_per_ui', loss_per_ui, loss_per_ui.shape)

        emb_loss = (ips * loss_per_ui).sum() / (ips.sum())


        self.emb_optimizer.zero_grad()
        emb_loss.backward()
        self.emb_optimizer.step()

        emb_loss_scalar = emb_loss.detach()

        return emb_loss_scalar

    def update_emb_model_mat_mse(self, train_label_tensor):
        out_train_rating = self.emb_model.out_forward_mat()
        out_train_loss = nn.functional.mse_loss(out_train_rating, train_label_tensor)

        emb_loss = out_train_loss

        self.emb_optimizer.zero_grad()
        emb_loss.backward()
        self.emb_optimizer.step()

        emb_loss_scalar = emb_loss.detach()

        return emb_loss_scalar


import torch
import numpy as np
import random

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def evaluate_and_log_train(config, epoch, tb_log, op, train_data=None):
    with torch.no_grad():
        if epoch == 1:
            print('train_data', train_data, train_data.shape)

        if train_data.any() != None:
            user_array = train_data[:, 0]
            item_array = train_data[:, 1]
            label_array = train_data[:, 2]

            user_tensor = torch.from_numpy(user_array).long().to(config['device'])
            item_tensor = torch.from_numpy(item_array).long().to(config['device'])
            label_tensor = torch.from_numpy(label_array).float().to(config['device'])
            rating = op.emb_model.get_rating(user_tensor, item_tensor)

            if config['ips_type'] == 'pop':
                ps = op.propensity_per_item[item_tensor]
            elif config['ips_type'] == 'logi':
                ps = (op.ps_user_emb[user_tensor] * op.ps_item_emb[item_tensor]).sum(dim=1)
                ps = torch.sigmoid(ps)
            elif config['ips_type'] == 'none':
                ps = torch.ones(len(train_data), device=config['device'])

            ips = 1.0 / ps
            train_loss_per_ui = torch.nn.functional.mse_loss(rating, label_tensor, reduction='none')

            train_loss = (ips * train_loss_per_ui).sum() / ips.sum()
            non_ips_train_loss = train_loss_per_ui.mean()

            if config['is_tensorboard']:
                tb_log.add_scalar(f"train/train_loss", train_loss, epoch)
                tb_log.add_scalar(f"train/non_ips_train_loss", non_ips_train_loss, epoch)


        else:
            train_loss = 0.0


    return train_loss


def evaluate_and_log_val(config, epoch, tb_log, op, val_data=None):
    with torch.no_grad():
        if epoch == 1:
            print('evaluated data', val_data, val_data.shape)

        if val_data.any() != None:
            user_array = val_data[:, 0]
            item_array = val_data[:, 1]
            label_array = val_data[:, 2]

            user_tensor = torch.from_numpy(user_array).long().to(config['device'])
            item_tensor = torch.from_numpy(item_array).long().to(config['device'])
            label_tensor = torch.from_numpy(label_array).float().to(config['device'])
            rating = op.emb_model.get_rating(user_tensor, item_tensor)
            #ps = torch.sigmoid((op.ps_user_emb[user_tensor] * op.ps_item_emb[item_tensor]).sum(dim=1))

            if config['ips_type'] == 'pop':
                ps = op.propensity_per_item[item_tensor]
            elif config['ips_type'] == 'logi':
                ps = (op.ps_user_emb[user_tensor] * op.ps_item_emb[item_tensor]).sum(dim=1)
                ps = torch.sigmoid(ps)
            elif config['ips_type'] == 'none':
                ps = op.propensity_per_item[item_tensor]

            ips = 1.0 / ps

            val_loss_per_ui = torch.nn.functional.mse_loss(rating, label_tensor, reduction='none')

            val_loss = (ips * val_loss_per_ui).sum() / ips.sum()
            non_ips_val_loss = val_loss_per_ui.mean()

            if config['is_tensorboard']:
                tb_log.add_scalar(f"validation/val_loss", val_loss, epoch)
                tb_log.add_scalar(f"validation/non_ips_val_loss", non_ips_val_loss, epoch)


        else:
            val_loss = 0.0


    return non_ips_val_loss
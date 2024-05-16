from tensorboardX import SummaryWriter
from pprint import pprint
import time
from world import config
from data_loader import load_data
from utils import set_seed, evaluate_and_log_train, evaluate_and_log_val
from opt import opter
import torch
import numpy as np
import csv
import itertools
if __name__ == '__main__':
    print('no batch')
    print('explicit')
    # dim_values = [20]
    # elr_values = [0.001, 0.005, 0.01, 0.05, 0.1]
    # ede_values = [0.0]
    dim_values = [20]
    elr_values = [0.05]
    ede_values = [0.0]

    for dim, elr, ede in itertools.product(dim_values, elr_values, ede_values):
        config['emb_dim'] = dim
        config['emb_lr'] = elr
        config['emb_decay'] = ede

        set_seed(config['seed'])
        dataset = load_data(config)
        tensorborad_path = config['tensorborad_path']

        config['num_users'] = dataset.num_users
        config['num_items'] = dataset.num_items
        config['rating_train_shape'] = dataset.rating_train.shape
        config['rating_val_shape'] = dataset.rating_val.shape
        config['propensity_per_item'] = dataset.propensity_per_item


        print('===============config====================')
        pprint(config)
        print('===============config====================')
        file_name = str(config['data_name']) + f"{config['core']}" + 'p' + str(int(config['val_ratio'] * 100)) + '_' \
                    + (f"{config['emb_dim']}{config['emb_model_name']}_{config['ips_type']}{config['power']}_"
                       f"ips_explicit_{config['loss_type']}_seed{config['seed']}_ex1")

        hyper_param = str(config['emb_lr']) + '_' + str(config['emb_decay'])

        if config['is_tensorboard']:
            tb_log = SummaryWriter(tensorborad_path + file_name + '/' + hyper_param)

        else:
            tb_log = None

        op = opter(config)

        # train procedure
        if config['loss_type'] == 'mse':
            train_user_tensor = torch.from_numpy(dataset.rating_train[:, 0]).long().to(config['device'])
            train_item_tensor = torch.from_numpy(dataset.rating_train[:, 1]).long().to(config['device'])
            train_label_tensor = torch.from_numpy(dataset.rating_train[:, 2]).float().to(config['device'])
            print('train_user_tensor', train_user_tensor, train_user_tensor.shape)
            print('train_item_tensor', train_item_tensor, train_item_tensor.shape)
            print('train_label_tensor', train_label_tensor, train_label_tensor.shape)

        elif config['loss_type'] == 'mat_mse':
            train_user_tensor = torch.zeros(1)
            train_item_tensor = torch.zeros(1)
            train_label_tensor = torch.from_numpy(dataset.full_rating).float().to(config['device'])



        for epoch in range(1, config['max_epochs'] + 1):
            epoch_start = time.time()
            emb_loss = op.update_and_log(train_user_tensor, train_item_tensor, train_label_tensor,
                                         epoch, tb_log)

            train_over = time.time()

            if epoch % 25 == 0 or epoch == 1:
                eval_start = time.time()
                train_loss = evaluate_and_log_train(config, epoch, tb_log, op,  train_data=dataset.rating_train)
                val_loss = evaluate_and_log_val(config, epoch, tb_log, op, val_data=dataset.rating_val)

                if epoch == 2000:
                    print(f'{epoch}save para', emb_loss, val_loss)
                    user_emb = op.emb_model.user_embedding.weight.detach().cpu().numpy()
                    item_emb = op.emb_model.item_embedding.weight.detach().cpu().numpy()

                    np.savetxt(f"../para/{config['data_name']}{config['core']}{config['emb_dim']}_full_mf_user_emb.txt", user_emb)
                    np.savetxt(f"../para/{config['data_name']}{config['core']}{config['emb_dim']}_full_mf_item_emb.txt", item_emb)


            if epoch % 1000 == 0:
                print(f'{epoch} train_cost, eval_cost, epoch_cost', train_over - epoch_start,
                      time.time() - eval_start, time.time() - epoch_start)



        if config['is_tensorboard']:
            tb_log.close()




































































































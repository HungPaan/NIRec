from pprint import pprint
from world import config
from utils import set_seed
from opt import opter
import csv
import numpy as np
from data_loader import load_data
from utils import check_and_create_folder
import torch
if __name__ == '__main__':
    print('no batch')
    print('group policy optimization var2')
    if config['is_rm'] == 0:
        config['folder_for_gpo_res'] \
            = (f"group_user_var2_{config['sample_type']}_seed{config['seed']}/"
               f"{config['filter_model_type']}/"
               f"{config['prefer_thres']}_{config['num_neigh_thres']}_"
               f"{config['num_guided_item']}_{config['num_group']}_{config['group_size']}/"
               f"{config['data_name']}{config['core']}"
               f"{config['alpha1']}{config['alpha2']}{config['alpha3']}"
               f"{config['indi_scale']}{config['neigh_scale']}{config['att_scale']}{config['beta']}{config['power']}/"
               f"{config['baseline_reward_type']}_{config['click_model_name']}_check/")

    else:
        config['folder_for_gpo_res'] \
            = (f"group_user_var2_{config['sample_type']}_seed{config['seed']}/"
               f"{config['filter_model_type']}/"
               f"{config['prefer_thres']}_{config['num_neigh_thres']}_"
               f"{config['num_guided_item']}_{config['num_group']}_{config['group_size']}/"
               f"{config['data_name']}{config['core']}"
               f"{config['alpha1']}{config['alpha2']}{config['alpha3']}"
               f"{config['indi_scale']}{config['neigh_scale']}{config['att_scale']}{config['beta']}{config['power']}_choosed25_var2/"
               f"{config['rm_click_model_name']}_{config['baseline_reward_type']}_{config['click_model_name']}/")


    if 'simulation' in config['click_model_name'] or 'simulation' in config['rm_click_model_name']:
        config['folder_for_gpo_res'] = config['folder_for_gpo_res'][:-1] + '_scale_revision/'
        
        
    check_and_create_folder(config['metric_path'] + config['folder_for_gpo_res'])

    set_seed(config['seed'])

    dataset = load_data(config)
    config['num_users'] = dataset.num_users
    config['num_items'] = dataset.num_items

    print('===============config====================')
    pprint(config)
    print('===============config====================')


    for gi_index, gi in enumerate(dataset.guided_items):
        for gu_index, gu in enumerate(dataset.guided_user_groups_list[gi_index]):
            if config["is_rm"] == 1:
                config['choosed25_guided_users_var2'] = dataset.choosed25_guided_user_groups_list_var2[gi_index][gu_index]
            else:
                config['choosed25_guided_users_var2'] = np.ones(2)

            config['guided_users'] = gu
            config['guided_item'] = gi

            print('choosed25_guided_users_var2 guided_users guided_item',
                  config['choosed25_guided_users_var2'], gu, gi, config['guided_users'], config['guided_item'])

            file_name = f"group{gi_index}_item{config['guided_item']}_ex1"

            set_seed(config['seed'])

            op = opter(config, dataset)

            pw_powers = np.arange(-3, 7) + 0.0
            pw_list = np.sort(np.concatenate((np.zeros(1),
                                              np.power(10, pw_powers) * 1.0,
                                              np.power(10, pw_powers) * 2.0,
                                              np.power(10, pw_powers) * 3.0,
                                              np.power(10, pw_powers) * 4.0,
                                              np.power(10, pw_powers) * 5.0,
                                              np.power(10, pw_powers) * 6.0,
                                              np.power(10, pw_powers) * 7.0,
                                              np.power(10, pw_powers) * 8.0,
                                              np.power(10, pw_powers) * 9.0)))

            print('pw_list', pw_list)

            res_oi_fn_pws = []
            res_metric_pws = []
            for pw in pw_list:
                config['penalty_weight'] = pw
                op.penalty_weight = pw

                set_seed(config['seed'])
                print('penalty_weight', config['penalty_weight'], op.penalty_weight, pw)

                hyper_param = str(config['penalty_weight'])

                if (config['click_model_name'] == 'mf'
                        or config['click_model_name'] == 'ncf'
                        or config['click_model_name'] == 'lightgcn'
                        or config['click_model_name'] == 'diffnet'):
                    res_oi_fn, res_metric = op.get_policy_baseline()
                else:
                    res_oi_fn, res_metric = op.get_policy()

                res_oi_fn_pws.append(res_oi_fn.cpu().numpy())
                res_metric_pws.append(res_metric.cpu().numpy())


            res_oi_fn_pws = np.hstack(res_oi_fn_pws)
            res_metric_pws = np.hstack(res_metric_pws)

            np.savetxt(config['metric_path'] + config['folder_for_gpo_res']
                          + file_name
                          + f"_res_metric_pws2.txt", res_metric_pws.T)
            np.savetxt(config['metric_path'] + config['folder_for_gpo_res']
                          + file_name
                          + f"_res_oi_fn_pws2.txt", res_oi_fn_pws.T, fmt='%d')

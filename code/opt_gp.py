from data_loader import load_data
from world import config
import numpy as np
import torch
from simulation import click_conv
from model import mf, ncf, lightgcn, diffnet, gatv2_spill_var2, simulation_mf, simulation_ncf, simulation_lightgcn, simulation_diffnet
# torch.set_printoptions(precision=6)
import os
from torch_geometric.utils import scatter
from os.path import join

def get_edge_wrt_guided_users(trust_edge, guided_users):
    arrange_index = torch.arange(trust_edge.shape[1], device=config['device'])
    trust_edge_index1_wrt_gu = arrange_index[torch.tensor([x in guided_users for x in trust_edge[1]])]
    trust_edge_index2_wrt_gu = arrange_index[torch.tensor([x in guided_users for x in trust_edge[0]])]

    flag1 = torch.zeros_like(arrange_index)
    flag1[trust_edge_index1_wrt_gu] = 1.0

    flag2 = torch.zeros_like(arrange_index)
    flag2[trust_edge_index2_wrt_gu] = 1.0

    flag = flag1 * flag2

    trust_edge_index_wrt_gu = arrange_index[flag.bool()]
    trust_edge_wrt_gu = trust_edge[:, trust_edge_index_wrt_gu]

    return trust_edge_index_wrt_gu, trust_edge_wrt_gu, torch.unique(trust_edge_wrt_gu[0])


def get_best_para(config, search_string1):
    for best_para_name in os.listdir(config['best_para_path']):
        if config['click_model_name'] == 'simulation_mf':
            choosed_click_model_name = 'mf'
        elif config['click_model_name'] == 'simulation_diffnet':
            choosed_click_model_name = 'diffnet'
        elif config['click_model_name'] == 'simulation_ncf':
            choosed_click_model_name = 'ncf'
        else:
            choosed_click_model_name = config['click_model_name']

        if (f"{choosed_click_model_name}_ex" in best_para_name
                and search_string1 in best_para_name):
            choosed_best_para = best_para_name

            if config['click_model_name'] == 'diffnet' or config['click_model_name'] == 'simulation_diffnet':
                num_layer_index = best_para_name.index('ex1_')
                config['num_layers'] = int(best_para_name[num_layer_index + len('ex1_')])
                print('choosed_num_layers', config['num_layers'])

                assert config['num_layers'] != 0

            break

    return choosed_best_para


dataset = load_data(config)
config['num_users'] = dataset.num_users
config['num_items'] = dataset.num_items

choosed_guided_user_groups_list = np.zeros_like(dataset.guided_user_groups_list).astype(int) - 1
if config['click_model_name'] == 'random':
    for gi_index, gi in enumerate(dataset.guided_items):
        for gu_index, gu in enumerate(dataset.guided_user_groups_list[gi_index]):
            choosed_index = np.random.permutation(50)[:25]
            choosed_guided_user_groups_list[gi_index, gu_index, :25] = gu[choosed_index]

else:
    for gi_index, gi in enumerate(dataset.guided_items):
        for gu_index, gu in enumerate(dataset.guided_user_groups_list[gi_index]):
            config['guided_users'] = gu
            config['guided_item'] = gi
            guided_users = torch.from_numpy(config['guided_users']).long().to(config['device'])
            guided_item = config['guided_item']
            print('guided_users guided_item', gu, gi, config['guided_users'], config['guided_item'])

            u_trust = torch.from_numpy(dataset.trust_data[:, [1, 0]]).long().t().contiguous().to(config['device'])
            trust_edge_index_wrt_gu, trust_edge_wrt_gu, neigh_wrt_gu = get_edge_wrt_guided_users(u_trust, guided_users)
            print('trust_edge_index_wrt_gu', trust_edge_index_wrt_gu, trust_edge_index_wrt_gu.shape)
            print('trust_edge_wrt_gu', trust_edge_wrt_gu, trust_edge_wrt_gu.shape)
            print('neigh_wrt_gu', neigh_wrt_gu, neigh_wrt_gu.shape)

            ground_truth_click_model = click_conv(config, dataset)
            groundtruth_linear_reward1_per_gte = ground_truth_click_model.get_linear_reward1(guided_item, trust_edge_wrt_gu)

            estimated_click_model = ground_truth_click_model
            estimated_linear_reward1_per_gte = groundtruth_linear_reward1_per_gte

            click_model_name = config['click_model_name']
            search_string1 = (f"{config['data_name']}{config['core']}"
                              f"{config['alpha1']}{config['alpha2']}{config['alpha3']}"
                              f"{config['indi_scale']}{config['neigh_scale']}{config['att_scale']}{config['beta']}{config['power']}_")

            if (click_model_name != 'groundtruth'):
                if click_model_name != 'lightgcn' and click_model_name != 'simulation_lightgcn':
                    choosed_best_para = get_best_para(config, search_string1)
                    print('choosed_best_para', choosed_best_para)

                    estimated_click_model = eval(f'{click_model_name}(config, dataset)')
                    estimated_click_model.load_state_dict(
                        torch.load(config['best_para_path'] + choosed_best_para, map_location=config['device']))
                else:
                    lgn_user_emb = np.loadtxt(config['best_para_path'] + search_string1 +
                                            f"base_lightgcn_user_emb.txt")
                    lgn_item_emb = np.loadtxt(config['best_para_path'] + search_string1 +
                                            f"base_lightgcn_item_emb.txt")

                    lgn_user_emb = torch.from_numpy(lgn_user_emb).float()
                    lgn_item_emb = torch.from_numpy(lgn_item_emb).float()
                    estimated_click_model = eval(f'{click_model_name}(config, dataset)')
                    estimated_click_model.user_embedding.weight.data = lgn_user_emb
                    estimated_click_model.item_embedding.weight.data = lgn_item_emb

                estimated_click_model.to(config['device'])

                for p in estimated_click_model.parameters():
                    print(f'{click_model_name} para', p, p.size())

                estimated_linear_reward1_per_edge \
                    = eval(f"estimated_click_model.get_linear_{config['baseline_reward_type']}_reward1(guided_item)")
                estimated_linear_reward1_per_gte = estimated_linear_reward1_per_edge[trust_edge_index_wrt_gu]
            print('estimated_linear_reward1_per_gte', estimated_linear_reward1_per_gte, estimated_linear_reward1_per_gte.shape)

            masked_index_per_user = torch.ones(config['num_users'], device=config['device'], dtype=torch.bool)

            for count in range(25):
                masked_guided_users = guided_users[masked_index_per_user[guided_users]]
                masked_trust_edge_wrt_gu = trust_edge_wrt_gu[:, masked_index_per_user[trust_edge_wrt_gu[0]]&masked_index_per_user[trust_edge_wrt_gu[1]]]

                masked_estimated_linear_reward1_per_gte = estimated_linear_reward1_per_gte[masked_index_per_user[trust_edge_wrt_gu[0]]&masked_index_per_user[trust_edge_wrt_gu[1]]]



                reward = scatter(masked_estimated_linear_reward1_per_gte, masked_trust_edge_wrt_gu[0],
                                                   dim=0, dim_size=config['num_users'], reduce='sum')

                reward_per_mgu = reward[masked_guided_users]

                min_reward_index_wrt_mgu = torch.argmin(reward_per_mgu, keepdim=False)
                min_reward_wrt_mgu = reward_per_mgu[min_reward_index_wrt_mgu]
                print('min_reward_wrt_mgu', min_reward_wrt_mgu)
                if min_reward_wrt_mgu >= 0:
                    break

                choosed_gu = masked_guided_users[min_reward_index_wrt_mgu]
                print('choosed_gu', choosed_gu)

                masked_index_per_user[choosed_gu] = False

            choosed_guided_users = guided_users[masked_index_per_user[guided_users]]

            choosed_guided_users = choosed_guided_users.cpu().numpy()
            choosed_guided_user_groups_list[gi_index, gu_index, :len(choosed_guided_users)] = choosed_guided_users
            print('choosed_guided_user_groups_list', choosed_guided_user_groups_list)

if 'simulation' in config['click_model_name']:
    np.save(
    join(config['data_path'] + config['data_name'] + '/',
         f"{config['sample_type']}_{config['seed']}_{config['data_name']}{config['core']}"
         f"{config['alpha1']}{config['alpha2']}{config['alpha3']}_"
         f"{config['prefer_thres']}_{config['num_neigh_thres']}_"
         f"{config['num_guided_item']}_{config['num_group']}_{config['group_size']}_"
         f"{config['filter_model_type']}_{config['click_model_name']}_scale_revision_choosed25_guided_user_groups_list_var2.npy"), choosed_guided_user_groups_list)
else:
    np.save(
        join(config['data_path'] + config['data_name'] + '/',
            f"{config['sample_type']}_{config['seed']}_{config['data_name']}{config['core']}"
            f"{config['alpha1']}{config['alpha2']}{config['alpha3']}_"
            f"{config['prefer_thres']}_{config['num_neigh_thres']}_"
            f"{config['num_guided_item']}_{config['num_group']}_{config['group_size']}_"
            f"{config['filter_model_type']}_{config['click_model_name']}_choosed25_guided_user_groups_list_var2.npy"), choosed_guided_user_groups_list)
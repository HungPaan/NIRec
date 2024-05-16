import numpy as np
import random
from data_loader import load_data
from world import config
from utils import set_seed
from os.path import join
import itertools


set_seed(0)

dataset = load_data(config)
config['num_users'] = dataset.num_users
config['num_items'] = dataset.num_items

data_name = config['data_name']
completion_type = config['completion_type']
core = config['core']
alpha1 = config['alpha1']
alpha2 = config['alpha2']
alpha3 = config['alpha3']
indi_scale = config['indi_scale']
neigh_scale = config['neigh_scale']
att_scale = config['att_scale']
beta = config['beta']
power = config['power']


num_neigh_per_user = np.bincount(dataset.trust_data[:, 0], minlength=dataset.num_users)
collected_target_user = np.where(num_neigh_per_user >= config['num_neigh_thres'])[0]
print('num_neigh_per_user', num_neigh_per_user, num_neigh_per_user.shape)
print('collected_target_user', collected_target_user, collected_target_user.shape)
print('num_neigh_per_collected_user', num_neigh_per_user[collected_target_user], num_neigh_per_user[collected_target_user].min())

collected_data = []
for u in collected_target_user:
    collected_data.append(dataset.trust_data[dataset.trust_data[:, 0]==u])
collected_data = np.vstack(collected_data)
print('collected_data', collected_data)

collected_source_user = np.unique(collected_data[:, 1])
print('collected_source_user', collected_source_user, collected_source_user.shape)

target_users_per_source_user_list = []
for u in collected_source_user:
    target_users_per_source_user_list.append(collected_data[collected_data[:, 1]==u][:, 0])

print('target_users_per_source_user_list', target_users_per_source_user_list)

groundtruth_user_emb = np.loadtxt(join(config['groundtruth_para_path'],
                                       f"{config['data_name']}{config['core']}{config['completion_dim']}_"
                                       f"{config['completion_type']}_user_emb.txt"))
groundtruth_item_emb = np.loadtxt(join(config['groundtruth_para_path'] + '/',
                                       f"{config['data_name']}{config['core']}{config['completion_dim']}_"
                                       f"{config['completion_type']}_item_emb.txt"))
guided_population_wrt_groundtruth = []

for i in range(dataset.num_items):
    collected_source_user_neigh_prefer = (groundtruth_user_emb[collected_source_user]
                                          * groundtruth_item_emb[i].reshape(-1, groundtruth_user_emb.shape[1])).sum(axis=1)
    collected_source_user_neigh_prefer = collected_source_user_neigh_prefer - 3.0

    temp_list = []
    for j in np.where(collected_source_user_neigh_prefer > 0)[0]:
        temp_list.append(target_users_per_source_user_list[j])
    if len(temp_list) != 0:
        unique_temp_list = np.unique(np.hstack(temp_list)).reshape(-1, 1)
        guided_population_wrt_groundtruth.append(np.hstack((unique_temp_list, np.ones_like(unique_temp_list) * i)))

guided_population_wrt_groundtruth = np.vstack(guided_population_wrt_groundtruth)

print('guided_population_wrt_groundtruth', guided_population_wrt_groundtruth, guided_population_wrt_groundtruth.shape)

np.random.shuffle(guided_population_wrt_groundtruth)

####
guided_population_wrt_groundtruth_rating = (
    (groundtruth_user_emb[guided_population_wrt_groundtruth[:, 0]]
     * groundtruth_item_emb[guided_population_wrt_groundtruth[:, 1]]).sum(axis=1)) - 3.0
guided_population_wrt_groundtruth = guided_population_wrt_groundtruth[guided_population_wrt_groundtruth_rating<config['prefer_thres']]
####

if config['group_size'] == 1:
    final_guided_population_wrt_groundtruth = guided_population_wrt_groundtruth[:1000]
    print('final_guided_population_wrt_groundtruth', final_guided_population_wrt_groundtruth)

    np.save(
        join(config['data_path'] + config['data_name'] + '/',
             f"{config['sample_type']}_{config['seed']}_{config['data_name']}{config['core']}"
             f"{config['alpha1']}{config['alpha2']}{config['alpha3']}_"
             f"{config['prefer_thres']}_{config['num_neigh_thres']}_"
             f"{config['num_guided_item']}_{config['num_group']}_{config['group_size']}_"
             f"{config['filter_model_type']}_guided_ui_pairs.npy"), final_guided_population_wrt_groundtruth)

else:
    temp_rational_item_freq = np.bincount(guided_population_wrt_groundtruth[:, 1])
    rational_item = np.where(temp_rational_item_freq > config['group_size'] + 10)[0]
    np.random.shuffle(rational_item)
    num_guided_item = min(len(rational_item), config['num_guided_item'])

    guided_items_list = []
    guided_user_groups_list = []
    num_users_for_ri_list = []
    print('len(rational_item)', len(rational_item))

    for ri in rational_item[:num_guided_item]:
        guided_user_groups_list_for_ri = []
        guided_items_list.append(ri)
        temp_users_for_ri = guided_population_wrt_groundtruth[guided_population_wrt_groundtruth[:, 1] == ri][:, 0]
        num_users_for_ri_list.append(len(temp_users_for_ri))
        for ng in range(config['num_group']):
            users_for_ri = np.random.choice(temp_users_for_ri, config['group_size'], replace=False)
            guided_user_groups_list_for_ri.append(users_for_ri)
        guided_user_groups_list.append(guided_user_groups_list_for_ri)

    guided_user_groups_list = np.array(guided_user_groups_list)
    guided_items_list = np.array(guided_items_list)
    num_users_for_ri_list = np.array(num_users_for_ri_list)
    print('guided_user_groups_list', guided_user_groups_list, guided_user_groups_list.shape)
    print('guided_items_list', guided_items_list, guided_items_list.shape)
    print('num_users_for_ri_list', num_users_for_ri_list, num_users_for_ri_list.shape,
          num_users_for_ri_list.max(), num_users_for_ri_list.min())

    np.save(
        join(config['data_path'] + config['data_name'] + '/',
             f"{config['sample_type']}_{config['seed']}_{config['data_name']}{config['core']}"
             f"{config['alpha1']}{config['alpha2']}{config['alpha3']}_"
             f"{config['prefer_thres']}_{config['num_neigh_thres']}_"
             f"{config['num_guided_item']}_{config['num_group']}_{config['group_size']}_"
             f"{config['filter_model_type']}_guided_user_groups_list.npy"), guided_user_groups_list)

    np.save(
        join(config['data_path'] + config['data_name'] + '/',
             f"{config['sample_type']}_{config['seed']}_{config['data_name']}{config['core']}"
             f"{config['alpha1']}{config['alpha2']}{config['alpha3']}_"
             f"{config['prefer_thres']}_{config['num_neigh_thres']}_"
             f"{config['num_guided_item']}_{config['num_group']}_{config['group_size']}_"
             f"{config['filter_model_type']}_guided_items.npy"), guided_items_list)





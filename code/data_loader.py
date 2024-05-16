import numpy as np
import os
from os.path import join

class load_data(object):
    def __init__(self, config):
        data_path = config['data_path']
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

        self.trust_data = np.loadtxt(join(data_path + data_name + '/', f'processed_trust{core}.txt')).astype(int)

        # [user, item, is_click]
        self.rated_data = np.loadtxt(
            join(data_path + data_name + '/', f'{data_name}{core}{alpha1}{alpha2}{alpha3}'
                                              f'{indi_scale}{neigh_scale}{att_scale}{beta}{power}'
                                              f'_{completion_type}_sim_data.txt')).astype(int)
        self.rated_data_wrt_click_prob = np.loadtxt(
            join(data_path + data_name + '/', f'{data_name}{core}{alpha1}{alpha2}{alpha3}'
                                              f'{indi_scale}{neigh_scale}{att_scale}{beta}{power}'
                                              f'_{completion_type}_sim_data_wrt_click_prob.txt'))


        guided_user_groups_list_path = join(config['data_path'] + config['data_name'] + '/',
                                            f"{config['sample_type']}_{config['seed']}_{config['data_name']}{config['core']}"
                                            f"{config['alpha1']}{config['alpha2']}{config['alpha3']}_"
                                            f"{config['prefer_thres']}_{config['num_neigh_thres']}_"
                                            f"{config['num_guided_item']}_{config['num_group']}_{config['group_size']}_"
                                            f"{config['filter_model_type']}_guided_user_groups_list.npy")


        choosed25_guided_user_groups_list_path_var2 = join(config['data_path'] + config['data_name'] + '/',
                                            f"{config['sample_type']}_{config['seed']}_{config['data_name']}{config['core']}"
                                            f"{config['alpha1']}{config['alpha2']}{config['alpha3']}_"
                                            f"{config['prefer_thres']}_{config['num_neigh_thres']}_"
                                            f"{config['num_guided_item']}_{config['num_group']}_{config['group_size']}_"
                                            f"{config['filter_model_type']}_"
                                            f"{config['rm_click_model_name']}_choosed25_guided_user_groups_list_var2.npy")


        guided_items_path = join(config['data_path'] + config['data_name'] + '/',
                                 f"{config['sample_type']}_{config['seed']}_{config['data_name']}{config['core']}"
                                 f"{config['alpha1']}{config['alpha2']}{config['alpha3']}_"
                                 f"{config['prefer_thres']}_{config['num_neigh_thres']}_"
                                 f"{config['num_guided_item']}_{config['num_group']}_{config['group_size']}_"
                                 f"{config['filter_model_type']}_guided_items.npy")


        guided_population_path = join(config['data_path'] + config['data_name'] + '/',
                                 f"{config['sample_type']}_{config['seed']}_{config['data_name']}{config['core']}"
                                 f"{config['alpha1']}{config['alpha2']}{config['alpha3']}_"
                                 f"{config['prefer_thres']}_{config['num_neigh_thres']}_"
                                 f"{config['num_guided_item']}_{config['num_group']}_{config['group_size']}_"
                                 f"{config['filter_model_type']}_guided_ui_pairs.npy")


        if os.path.exists(guided_user_groups_list_path):
            self.guided_user_groups_list = np.load(guided_user_groups_list_path).astype(int)
            self.guided_items = np.load(guided_items_path).astype(int)

        else:
            self.guided_user_groups_list = np.zeros(1).astype(int)
            self.guided_items = np.zeros(1).astype(int)


        if os.path.exists(choosed25_guided_user_groups_list_path_var2):
            self.choosed25_guided_user_groups_list_var2 = np.load(choosed25_guided_user_groups_list_path_var2).astype(int)

        else:
            self.choosed25_guided_user_groups_list_var2 = np.zeros(1).astype(int)


        if os.path.exists(guided_population_path):
            self.guided_population = np.load(guided_population_path).astype(int)
        else:
            self.guided_population = np.zeros(1).astype(int)


        self.num_users = int(np.max(self.rated_data[:, 0])) + 1
        self.num_items = int(np.max(self.rated_data[:, 1])) + 1

        print(f'{data_name} num_users', self.num_users)
        print(f'{data_name} num_items', self.num_items)
        print(f'{data_name} trust_data', self.trust_data, self.trust_data.shape)

        print(f'{data_name} rated_data', self.rated_data, self.rated_data.shape)
        print(f'{data_name} rated_data_wrt_click_prob', self.rated_data_wrt_click_prob, self.rated_data_wrt_click_prob.shape)

        print(f'{data_name} guided_user_groups_list', self.guided_user_groups_list, self.guided_user_groups_list.shape)
        print(f'{data_name} choosed25_guided_user_groups_list_path', self.choosed25_guided_user_groups_list_var2,
              self.choosed25_guided_user_groups_list_var2.shape)
        print(f'{data_name} guided_items', self.guided_items, self.guided_items.shape)
        print(f'{data_name} guided_population', self.guided_population, self.guided_population.shape)


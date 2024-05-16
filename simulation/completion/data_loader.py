import numpy as np
from os.path import join
from world import config
class load_data(object):
    def __init__(self, config):
        data_path = config['data_path']
        data_name = config['data_name']
        val_ratio = config['val_ratio']
        ips_type = config['ips_type']
        power = config['power']
        core = config['core']
        emb_dim = config['emb_dim']

        rating_data = np.loadtxt(join(data_path + data_name + '/', f'processed_rating{core}.txt')).astype(int)
        # rating_data = np.loadtxt(join(data_path + data_name + '/', f'processed_rating{core}.txt'))
        rating_data_rand_index = np.random.permutation(len(rating_data))

        if config['loss_type'] == 'mat_mse':
            self.full_rating = np.loadtxt(join(data_path + data_name + '/', f'{data_name}{core}{emb_dim}_mf_full_rating.txt')).astype(int)
            print('full_rating', self.full_rating, self.full_rating.shape)

            print("min max", self.full_rating[0].max(), self.full_rating[0].min())

        self.rating_train = rating_data[rating_data_rand_index[int(val_ratio * len(rating_data)): ]]
        self.rating_val = rating_data[rating_data_rand_index[0: int(val_ratio * len(rating_data))]]

        self.num_users = int(np.max(rating_data[:, 0])) + 1
        self.num_items = int(np.max(rating_data[:, 1])) + 1


        print(f'{data_name} num_users', self.num_users)
        print(f'{data_name} num_items', self.num_items)

        # train & validation & test
        print(f'{data_name} rating_train', self.rating_train, self.rating_train.shape)
        print(f'{data_name} rating_val', self.rating_val, self.rating_val.shape)

        count_inter_per_item = np.bincount(rating_data[:, 1].astype(int), minlength=self.num_items)
        propensity_per_item = count_inter_per_item / np.max(count_inter_per_item)
        self.propensity_per_item = np.power(propensity_per_item, power)
        print('propensity_per_item', self.propensity_per_item, self.propensity_per_item.shape)
        print('propensity_per_item', self.propensity_per_item.min(), self.propensity_per_item.max())

if __name__ == '__main__':
    print('=================data_loader=====================')
    data = load_data(config)






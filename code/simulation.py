import numpy as np
from os.path import join
import torch
import data_loader
from world import config
import torch_geometric
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import scatter
from utils import set_seed
class click_conv(MessagePassing):
    def __init__(self, config, dataset):
        super().__init__(aggr='add')
        self.num_users = config['num_users']
        self.num_items = config['num_items']
        self.device = config['device']

        self.indi_scale = config['indi_scale']
        self.neigh_scale = config['neigh_scale']
        self.att_scale = config['att_scale']
        self.beta = config['beta']
        self.power = config['power']
        self.completion_dim = config['completion_dim']

        # [2, num_edge] the first row contains source nodes, the second row contains target nodes
        self.u_trust = torch.from_numpy(dataset.trust_data[:, [1, 0]]).long().t().contiguous().to(config['device'])

        self.groundtruth_user_emb = torch.from_numpy(
            np.loadtxt(join(config['groundtruth_para_path'],
                            f"{config['data_name']}{config['core']}{config['completion_dim']}_"
                            f"{config['completion_type']}_user_emb.txt"))).float().to(config['device'])
        self.groundtruth_item_emb = torch.from_numpy(
            np.loadtxt(join(config['groundtruth_para_path'] + '/',
                            f"{config['data_name']}{config['core']}{config['completion_dim']}_"
                            f"{config['completion_type']}_item_emb.txt"))).float().to(config['device'])

        self.in_degree = torch_geometric.utils.degree(self.u_trust[1], self.num_users).float()
        self.in_norm = torch.pow(self.in_degree, config['power'])
        self.in_norm[self.in_norm == 0.0] = 1.0

        print('---------------------------------click_conv---------------------------------------')
        print('click_conv init')
        print('num_users num_items', self.num_users, self.num_items)
        print('device', self.device)
        print('indi_scale, neigh_scale, att_scale, beta power',
              self.indi_scale, self.neigh_scale, self.att_scale, self.beta, self.power)
        print('completion_dim', self.completion_dim)
        print('u_trust', self.u_trust, self.u_trust.shape)
        print('groundtruth_user_emb', self.groundtruth_user_emb, self.groundtruth_user_emb.shape)
        print('groundtruth_item_emb', self.groundtruth_item_emb, self.groundtruth_item_emb.shape)
        print('in_degree', self.in_degree, self.in_degree.shape)
        print('in_norm', self.in_norm, self.in_norm.shape)
        print('---------------------------------click_conv---------------------------------------')

    def get_att(self, trust_edge):
        att_per_edge = torch.sigmoid((self.groundtruth_user_emb[trust_edge[0]]
                                      * self.groundtruth_user_emb[trust_edge[1]]).sum(dim=1, keepdim=True)
                                     * self.att_scale) / self.in_norm[trust_edge[1]].reshape(-1, 1)

        return att_per_edge


    def click_forward(self, oi, guided_users, guided_item):
        base_prefer = (self.groundtruth_user_emb
                       * self.groundtruth_item_emb[guided_item].unsqueeze(dim=0)).sum(dim=1, keepdim=True)
        base_prefer = base_prefer - 3.0

        indi_prefer = base_prefer * self.indi_scale

        neigh_prefer = base_prefer * self.neigh_scale
        prefer_factor = oi * neigh_prefer

        att_per_edge = self.get_att(self.u_trust)
        neigh_effect = self.propagate(edge_index=self.u_trust, x=prefer_factor, coeff=att_per_edge)

        click_logit = indi_prefer + self.beta * neigh_effect
        guided_indi_click_prob = torch.sigmoid(indi_prefer[guided_users])
        guided_click_prob = torch.sigmoid(click_logit[guided_users])

        return click_logit, guided_indi_click_prob, guided_click_prob


    def message(self, x_j, coeff):
        return x_j * coeff


    def get_linear_reward1(self, guided_item, trust_edge):
        att_per_edge = self.get_att(trust_edge)
        neigh_prefer = (self.groundtruth_user_emb[trust_edge[0]]
                       * self.groundtruth_item_emb[guided_item].unsqueeze(dim=0)).sum(dim=1, keepdim=True)
        neigh_prefer = (neigh_prefer - 3.0) * self.neigh_scale

        reward1_per_edge = (self.beta * att_per_edge * neigh_prefer)


        return reward1_per_edge


    def get_rating_per_ui(self, users, items):
        rating = (self.groundtruth_user_emb[users]
                  * self.groundtruth_item_emb[items].reshape(-1, self.groundtruth_user_emb.shape[1])).sum(dim=1)

        return rating



if __name__ == '__main__':
    print('=================simulation=====================')
    set_seed(0)
    dataset = data_loader.load_data(config)
    config['num_users'] = dataset.num_users
    config['num_items'] = dataset.num_items
    test_click_model = click_conv(config, dataset)
    # oi = torch.ones((test_click_model.num_users, 1), device=config['device'])
    # oi[test_click_model.filtered_source_neigh] = torch.randint(0, 2, (10, 1)).float().to(config['device'])
    # indi_click_prob, neigh_effect, click_prob = test_click_model.click_forward(oi)
    # neigh_reward = test_click_model.get_mean_reward_per_1_sub_mean()
    # choosed_neigh_reward = neigh_reward[oi[test_click_model.filtered_source_neigh].bool()]
    # check_indi_click_prob = torch.sigmoid(test_click_model.base_prefer[test_click_model.target_user]*0.25)
    # check_neigh_effect = choosed_neigh_reward.sum()
    # check_click_prob = torch.sigmoid(test_click_model.base_prefer[test_click_model.target_user]*0.25 + check_neigh_effect)
    # print('indi_click_prob', indi_click_prob, check_indi_click_prob)
    # print('neigh_effect', neigh_effect, check_neigh_effect)
    # print('check_click_prob', click_prob, check_click_prob)

    # oi = torch.randint(0, 2, (config['num_users'], 1)).float().to(config['device'])
    # target_indi_click_prob, target_click_prob \
    #     = test_click_model.click_forward(oi, torch.arange(config['num_users'], device=config['device']), 1)
    # print('target_indi_click_prob', target_indi_click_prob)
    # print('target_click_prob', target_click_prob)
    # np.savetxt('oi.txt', oi.cpu().numpy())
    # np.savetxt('target_indi_click_prob.txt', target_indi_click_prob.cpu().numpy())
    # np.savetxt('target_click_prob.txt', target_click_prob.cpu().numpy())



import numpy as np
import torch
from simulation import click_conv
from model import mf, ncf, lightgcn, diffnet, gatv2_spill_var2, simulation_mf, simulation_ncf, simulation_lightgcn, simulation_diffnet
# torch.set_printoptions(precision=6)
import os
from torch_geometric.utils import scatter

class opter(object):
    def __init__(self, config, dataset):
        self.num_users = config['num_users']
        self.num_items = config['num_items']
        self.device = config['device']
        self.is_rm = config['is_rm']

        self.penalty_weight = config['penalty_weight']
        self.base_penalty_type = config['base_penalty_type']
        self.baseline_reward_type = config['baseline_reward_type']

        choosed25_guided_users_var2 = torch.from_numpy(config['choosed25_guided_users_var2']).long().to(self.device)
        self.choosed25_guided_users_var2 = choosed25_guided_users_var2[torch.ge(choosed25_guided_users_var2, 0)]
        self.guided_users = torch.from_numpy(config['guided_users']).long().to(self.device)
        self.guided_item = config['guided_item']

        if self.is_rm == 0:
            self.choosed25_guided_users_var2 = self.guided_users

        # [2, num_trust]
        self.u_trust = torch.from_numpy(dataset.trust_data[:, [1, 0]]).long().t().contiguous().to(config['device'])

        # [len(trust_edge_index_wrt_gu)], [len(filtered_neigh)]
        self.filtered_neigh, filtered_trust_edge_index_wrt_gu = self.get_filtered_neigh(self.u_trust, self.guided_users)
        self.filtered_u_trust_wrt_gu = self.u_trust[:, filtered_trust_edge_index_wrt_gu]
        self.num_filter_neigh = len(self.filtered_neigh)

        self.ground_truth_click_model = click_conv(config, dataset)

        groundtruth_linear_reward1_per_fte = self.ground_truth_click_model.get_linear_reward1(self.guided_item,
                                                                                              self.filtered_u_trust_wrt_gu)

        self.estimated_click_model = self.ground_truth_click_model
        self.estimated_linear_reward1_per_fte = groundtruth_linear_reward1_per_fte

        self.click_model_name = config['click_model_name']
        search_string1 = (f"{config['data_name']}{config['core']}"
                          f"{config['alpha1']}{config['alpha2']}{config['alpha3']}"
                          f"{config['indi_scale']}{config['neigh_scale']}{config['att_scale']}{config['beta']}{config['power']}_")


        if (self.click_model_name != 'groundtruth'):
            if self.click_model_name != 'lightgcn' and self.click_model_name != 'simulation_lightgcn':
                choosed_best_para = self.get_best_para(config, search_string1)
                print('choosed_best_para', choosed_best_para)

                self.estimated_click_model = eval(f'{self.click_model_name}(config, dataset)')
                self.estimated_click_model.load_state_dict(
                    torch.load(config['best_para_path'] + choosed_best_para, map_location=config['device']))
            else:
                lgn_user_emb = np.loadtxt(config['best_para_path'] + search_string1 +
                                           f"base_lightgcn_user_emb.txt")
                lgn_item_emb = np.loadtxt(config['best_para_path'] + search_string1 +
                                           f"base_lightgcn_item_emb.txt")

                lgn_user_emb = torch.from_numpy(lgn_user_emb).float()
                lgn_item_emb = torch.from_numpy(lgn_item_emb).float()
                self.estimated_click_model = eval(f'{self.click_model_name}(config, dataset)')
                self.estimated_click_model.user_embedding.weight.data = lgn_user_emb
                self.estimated_click_model.item_embedding.weight.data = lgn_item_emb

            self.estimated_click_model.to(config['device'])

            for p in self.estimated_click_model.parameters():
                print(f'{self.click_model_name} para', p, p.size())

            estimated_linear_reward1_per_edge \
                = eval(f'self.estimated_click_model.get_linear_{self.baseline_reward_type}_reward1(self.guided_item)')
            self.estimated_linear_reward1_per_fte = estimated_linear_reward1_per_edge[filtered_trust_edge_index_wrt_gu]


        self.oi_his = self.set_his(dataset.rated_data)

        self.num_base_rec_per_user = config['num_base_rec_per_user']
        base_user_emb = np.loadtxt(config['best_para_path'] + search_string1 +
                                   f"base_{config['base_model_name']}_user_emb.txt")
        base_item_emb = np.loadtxt(config['best_para_path'] + search_string1 +
                                   f"base_{config['base_model_name']}_item_emb.txt")

        self.oi_base, self.candi_penalty, self.base_value = self.set_base(dataset.rated_data, base_user_emb, base_item_emb)

        if self.base_penalty_type == 'fix':
            self.candi_penalty = - torch.ones(self.num_users, device=self.device) * 0.5


        print('=================================opter=======================================')
        print('opter init')

        print('num_users', self.num_users)
        print('num_items', self.num_items)
        print('choosed25_guided_users_var2', self.choosed25_guided_users_var2, self.choosed25_guided_users_var2.shape)
        print('guided_users', self.guided_users)
        print('guided_item', self.guided_item)
        print('device', self.device)

        print('penalty_weight', self.penalty_weight)

        print('click_model_name', self.click_model_name)
        print('base_penalty_type', self.base_penalty_type)
        print('baseline_reward_type', self.baseline_reward_type)

        print('filtered_neigh', self.filtered_neigh, self.filtered_neigh.shape)
        print('filtered_trust_edge_index_wrt_gu', filtered_trust_edge_index_wrt_gu, filtered_trust_edge_index_wrt_gu.shape)
        print('filtered_u_trust_wrt_gu', self.filtered_u_trust_wrt_gu, self.filtered_u_trust_wrt_gu.shape)

        print('groundtruth_linear_reward1_per_fte', groundtruth_linear_reward1_per_fte,
              groundtruth_linear_reward1_per_fte.shape)
        print('estimated_linear_reward1_per_fte', self.estimated_linear_reward1_per_fte,
              self.estimated_linear_reward1_per_fte.shape)

        print('oi_his', self.oi_his, self.oi_his.shape, self.oi_his.sum())
        print('oi_base', self.oi_base, self.oi_base.shape, self.oi_base.sum())
        print('candi_penalty', self.candi_penalty, self.candi_penalty.shape)
        print('base_value', self.base_value, self.base_value.shape)
        print('=================================opter=======================================')

    def get_filtered_neigh(self, trust_edge, guided_users):
        arrange_index = torch.arange(trust_edge.shape[1], device=self.device)
        trust_edge_index_wrt_gu = arrange_index[torch.tensor([x in guided_users for x in trust_edge[1]])]
        source_users = torch.unique(trust_edge[0][trust_edge_index_wrt_gu])
        filtered_source_users = source_users[~torch.tensor([x in guided_users for x in source_users])]
        filtered_trust_edge_index_wrt_gu \
            = trust_edge_index_wrt_gu[torch.tensor([x in filtered_source_users for x in trust_edge[0][trust_edge_index_wrt_gu]])]

        return filtered_source_users, filtered_trust_edge_index_wrt_gu


    def get_best_para(self, config, search_string1):
        for best_para_name in os.listdir(config['best_para_path']):
            if self.click_model_name == 'simulation_mf':
                choosed_click_model_name = 'mf'
            elif self.click_model_name == 'simulation_diffnet':
                choosed_click_model_name = 'diffnet'
            elif self.click_model_name == 'simulation_ncf':
                choosed_click_model_name = 'ncf'

            else:
                choosed_click_model_name = self.click_model_name

            if (f"{choosed_click_model_name}_ex" in best_para_name
                    and search_string1 in best_para_name):
                choosed_best_para = best_para_name

                if self.click_model_name == 'diffnet' or self.click_model_name == 'simulation_diffnet':
                    num_layer_index = best_para_name.index('ex1_')
                    config['num_layers'] = int(best_para_name[num_layer_index + len('ex1_')])
                    print('choosed_num_layers', config['num_layers'])

                    assert config['num_layers'] != 0

                break

        return choosed_best_para


    def set_his(self, his_data):
        his_data_for_guided_item = his_data[his_data[:, 1] == self.guided_item]

        oi_his = np.zeros(self.num_users)
        oi_his[his_data_for_guided_item[:, 0]] = 1.0

        oi_his = torch.from_numpy(oi_his).float().to(self.device).reshape(-1, 1)

        return oi_his


    def set_base(self, his_data, base_user_emb, base_item_emb):
        base_user_emb = torch.from_numpy(base_user_emb).float().to(self.device)
        base_item_emb = torch.from_numpy(base_item_emb).float().to(self.device)

        all_prob = torch.sigmoid(torch.matmul(base_user_emb, base_item_emb.t()))
        all_prob[his_data[:, 0], his_data[:, 1]] = -100.0
        base_prob, base_rec_list = torch.topk(all_prob, self.num_base_rec_per_user+1)

        oi_base = ((base_rec_list[:, :-1] == self.guided_item).sum(dim=1, keepdim=True) > 0).float()
        penalty0 = (all_prob[:, self.guided_item] - base_prob[:, -2]).reshape(-1, 1)
        penalty1 = (base_prob[:, -1] - all_prob[:, self.guided_item]).reshape(-1, 1)
        penalty2 = (0.0 - base_prob[:, -2]).reshape(-1, 1)

        penalty = torch.cat([penalty0, penalty1, penalty2], dim=1)
        candi_penalty = torch.gather(penalty, 1, (self.oi_his * 2.0 + oi_base).long())

        return oi_base, candi_penalty, base_prob[:, -2]

    def get_policy_baseline(self):
        res_metric = torch.zeros((10, 1), device=self.device)
        click_reward_per_user = scatter(self.estimated_linear_reward1_per_fte, self.filtered_u_trust_wrt_gu[0],
                                        dim=0, dim_size=self.num_users, reduce='sum')
        click_reward_per_fn = click_reward_per_user[self.filtered_neigh]

        penalty1_per_fn = (self.candi_penalty * (1.0 - self.oi_base).abs())[self.filtered_neigh]
        penalty0_per_fn = (self.candi_penalty * (0.0 - self.oi_base).abs())[self.filtered_neigh]


        reward1_per_fn = click_reward_per_fn + self.penalty_weight * penalty1_per_fn
        reward0_per_fn = self.penalty_weight * penalty0_per_fn

        reward_per_fn = torch.cat([reward0_per_fn.reshape(-1, 1), reward1_per_fn.reshape(-1, 1)], dim=1)
        oi_per_fn = torch.argmax(reward_per_fn, dim=1).float().reshape(-1, 1)

        oi = self.oi_base.clone()
        oi[self.filtered_neigh] = oi_per_fn
        oi[self.guided_users] = 1.0

        _, guided_indi_click_prob, guided_click_prob = self.ground_truth_click_model.click_forward(oi,
                                                                                                   self.guided_users,
                                                                                                   self.guided_item)

        res_metric[0] = indi_click_prob = guided_indi_click_prob.mean(dim=0)
        res_metric[1] = global_click_prob = guided_click_prob.mean(dim=0)
        res_metric[2] = click_improv_wrt_user = ((guided_click_prob - guided_indi_click_prob) / guided_indi_click_prob).mean(dim=0)
        res_metric[3] = click_improv_wrt_group = (global_click_prob - indi_click_prob) / indi_click_prob

        penalty_per_user = self.candi_penalty * (oi - self.oi_base).abs()
        res_metric[4] = penalty_all_users = penalty_per_user.sum(dim=0)
        res_metric[5] = penalty_fn = penalty_per_user[self.filtered_neigh].sum(dim=0)

        res_metric[6] = penalty_all_users_percent = penalty_all_users / self.base_value.sum(dim=0)
        res_metric[7] = penalty_fn_percent = penalty_fn / self.base_value[self.filtered_neigh].sum(dim=0)

        res_metric[8] = num_replaced_items_all_users = (oi - self.oi_base).abs().sum(dim=0)
        res_metric[9] = num_replaced_items_fn = (oi - self.oi_base)[self.filtered_neigh].abs().sum(dim=0)

        return oi[self.filtered_neigh], res_metric


    def get_policy(self):
        res_metric = torch.zeros((10, 1), device=self.device)
        # Todo: initial strategy
        current_oi = self.oi_base.clone()
        #current_oi[self.filtered_neigh] = 0.0
        current_oi[self.guided_users] = 1.0
        #current_oi[self.choosed25_guided_users_var2] = 1.0


        current_click_logit, _, _ = self.estimated_click_model.click_forward(current_oi, self.guided_users, self.guided_item)
        masked_index_per_user = torch.ones(self.num_users, device=self.device, dtype=torch.bool)


        for ne in range(self.num_filter_neigh):
            masked_filtered_neigh = self.filtered_neigh[masked_index_per_user[self.filtered_neigh]]
            masked_filtered_u_trust_wrt_gu = self.filtered_u_trust_wrt_gu[:, masked_index_per_user[self.filtered_u_trust_wrt_gu[0]]]
            estimated_linear_reward1_per_mfte = self.estimated_linear_reward1_per_fte[masked_index_per_user[self.filtered_u_trust_wrt_gu[0]]]

            current_click_logit_per_mfte = current_click_logit[masked_filtered_u_trust_wrt_gu[1]]
            current_click_prob_per_mfte = torch.sigmoid(current_click_logit_per_mfte)


            flipped_current_oi = 1.0 - current_oi
            diff_oi_per_mfte = flipped_current_oi[masked_filtered_u_trust_wrt_gu[0]] - current_oi[masked_filtered_u_trust_wrt_gu[0]]

            candi_next_click_logit_per_mfte = current_click_logit_per_mfte + diff_oi_per_mfte * estimated_linear_reward1_per_mfte
            candi_next_click_prob_per_mfte = torch.sigmoid(candi_next_click_logit_per_mfte)

            diff_click_prob_per_mfte = candi_next_click_prob_per_mfte - current_click_prob_per_mfte

            diff_click_prob_per_user = scatter(diff_click_prob_per_mfte, masked_filtered_u_trust_wrt_gu[0],
                                               dim=0, dim_size=self.num_users, reduce='sum')
            diff_click_prob_per_mfn = diff_click_prob_per_user[masked_filtered_neigh]


            current_penalty_per_mfn = (self.candi_penalty * (current_oi - self.oi_base).abs())[masked_filtered_neigh]
            candi_next_penalty_per_mfn = (self.candi_penalty * (flipped_current_oi - self.oi_base).abs())[masked_filtered_neigh]
            diff_penalty_per_mfn = candi_next_penalty_per_mfn - current_penalty_per_mfn

            diff_per_mfn = diff_click_prob_per_mfn + self.penalty_weight * diff_penalty_per_mfn
            top1_diff_index_wrt_mfn = torch.argmax(diff_per_mfn, keepdim=False)
            top1_diff_wrt_mfn = diff_per_mfn[top1_diff_index_wrt_mfn]

            choosed_neigh = masked_filtered_neigh[top1_diff_index_wrt_mfn]

            if top1_diff_wrt_mfn > 0:
                current_oi[choosed_neigh] = flipped_current_oi[choosed_neigh]
                masked_index_per_user[choosed_neigh] = False
            elif top1_diff_wrt_mfn == 0:
                if diff_click_prob_per_mfn[top1_diff_index_wrt_mfn] > 0:
                    current_oi[choosed_neigh] = flipped_current_oi[choosed_neigh]
                    masked_index_per_user[choosed_neigh] = False
                else:
                    break
            else:
                break

            influenced_gu_index = (masked_filtered_u_trust_wrt_gu[0] == choosed_neigh)
            influenced_gu = masked_filtered_u_trust_wrt_gu[1][influenced_gu_index]
            current_click_logit[influenced_gu] = candi_next_click_logit_per_mfte[influenced_gu_index]

        oi = current_oi.clone()
        oi[self.guided_users] = 0.0
        oi[self.choosed25_guided_users_var2] = 1.0
        _, guided_indi_click_prob, guided_click_prob = self.ground_truth_click_model.click_forward(oi,
                                                                                                   self.guided_users,
                                                                                                   self.guided_item)

        res_metric[0] = indi_click_prob = guided_indi_click_prob.mean(dim=0)
        res_metric[1] = global_click_prob = guided_click_prob.mean(dim=0)
        res_metric[2] = click_improv_wrt_user = ((guided_click_prob - guided_indi_click_prob) / guided_indi_click_prob).mean(dim=0)
        res_metric[3] = click_improv_wrt_group = (global_click_prob - indi_click_prob) / indi_click_prob

        penalty_per_user = self.candi_penalty * (oi - self.oi_base).abs()
        res_metric[4] = penalty_all_users = penalty_per_user.sum(dim=0)
        res_metric[5] = penalty_fn = penalty_per_user[self.filtered_neigh].sum(dim=0)

        res_metric[6] = penalty_all_users_percent = penalty_all_users / self.base_value.sum(dim=0)
        res_metric[7] = penalty_fn_percent = penalty_fn / self.base_value[self.filtered_neigh].sum(dim=0)

        res_metric[8] = num_replaced_items_all_users = (oi - self.oi_base).abs().sum(dim=0)
        res_metric[9] = num_replaced_items_fn = (oi - self.oi_base)[self.filtered_neigh].abs().sum(dim=0)

        return oi[self.filtered_neigh], res_metric







if __name__ == '__main__':
    print('=================opt=====================')

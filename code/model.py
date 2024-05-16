import numpy as np
import torch
from torch import nn
import torch_geometric
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import (degree, scatter,)
from utils import softmax_1


class ncf(nn.Module):
    def __init__(self, config, dataset):
        super(ncf, self).__init__()
        self.num_users = config['num_users']
        self.num_items = config['num_items']
        self.device = config['device']
        self.emb_dim = config['emb_dim']

        self.u_trust = torch.from_numpy(dataset.trust_data[:, [1, 0]]).long().t().contiguous().to(
            self.device)

        self.in_degree = degree(self.u_trust[1], self.num_users)

        self.relu = torch.nn.ReLU()

        self.user_embedding = nn.Embedding(num_embeddings=self.num_users, embedding_dim=self.emb_dim)
        self.item_embedding = nn.Embedding(num_embeddings=self.num_items, embedding_dim=self.emb_dim)

        self.linear_1 = torch.nn.Linear(self.emb_dim * 2, self.emb_dim)
        self.linear_2 = torch.nn.Linear(self.emb_dim, 1, bias=False)


        print('---------------------------------ncf---------------------------------------')
        print('ncf init')
        print('num_users num_items emb_dim', self.num_users, self.num_items,  self.emb_dim)
        print('u_trust', self.u_trust)
        print('in_degree', self.in_degree)
        print('---------------------------------ncf---------------------------------------')

    def get_emb(self):
        return self.user_embedding.weight, self.item_embedding.weight

    def ncf_forward(self, user_emb, item_emb):
        z_emb = torch.cat([user_emb, item_emb], dim=1)
        h1 = self.linear_1(z_emb)
        h1 = self.relu(h1)

        rating = self.linear_2(h1)

        return rating


    def get_linear_spill_reward1(self, guided_item):
        with torch.no_grad():
            final_user_emb, final_item_emb = self.get_emb()

            source_user_emb = final_user_emb[self.u_trust[0]]
            target_user_emb = final_user_emb[self.u_trust[1]]

            att_per_edge = (torch.sigmoid((source_user_emb * target_user_emb).sum(dim=1, keepdim=True))
                            / self.in_degree[self.u_trust[1]].reshape(-1, 1))

            guided_item_emb = final_item_emb[guided_item].expand(source_user_emb.shape[0], self.emb_dim)
            neigh_prefer = self.ncf_forward(source_user_emb, guided_item_emb)

            reward1_per_edge = torch.sigmoid(att_per_edge * neigh_prefer) - 0.5


            return reward1_per_edge


    def get_linear_sub_mean_reward1(self, guided_item):
        with torch.no_grad():
            final_user_emb, final_item_emb = self.get_emb()

            source_user_emb = final_user_emb[self.u_trust[0]]
            guided_item_emb = final_item_emb[guided_item].expand(source_user_emb.shape[0], self.emb_dim)

            rating = self.ncf_forward(source_user_emb, guided_item_emb)

            reward1_per_edge = torch.sigmoid(rating) - 0.5


            return reward1_per_edge



class lightgcn(nn.Module):
    def __init__(self, config, dataset):
        super(lightgcn, self).__init__()
        self.num_users = config['num_users']
        self.num_items = config['num_items']
        self.device = config['device']
        self.emb_dim = config['emb_dim']

        self.u_trust = torch.from_numpy(dataset.trust_data[:, [1, 0]]).long().t().contiguous().to(
            self.device)

        self.in_degree = degree(self.u_trust[1], self.num_users)

        self.user_embedding = nn.Embedding(num_embeddings=self.num_users, embedding_dim=self.emb_dim)
        self.item_embedding = nn.Embedding(num_embeddings=self.num_items, embedding_dim=self.emb_dim)


        print('---------------------------------lightgcn---------------------------------------')
        print('lightgcn init')
        print('num_users num_items emb_dim', self.num_users, self.num_items,  self.emb_dim)
        print('u_trust', self.u_trust)
        print('in_degree', self.in_degree)
        print('---------------------------------lightgcn---------------------------------------')

    def get_emb(self):
        return self.user_embedding.weight, self.item_embedding.weight


    def get_linear_spill_reward1(self, guided_item):
        with torch.no_grad():
            final_user_emb, final_item_emb = self.get_emb()

            source_user_emb = final_user_emb[self.u_trust[0]]
            target_user_emb = final_user_emb[self.u_trust[1]]

            att_per_edge = (torch.sigmoid((source_user_emb * target_user_emb).sum(dim=1, keepdim=True))
                            / self.in_degree[self.u_trust[1]].reshape(-1, 1))

            guided_item_emb = final_item_emb[guided_item].unsqueeze(dim=0)
            neigh_prefer = (source_user_emb * guided_item_emb).sum(dim=1, keepdim=True)

            reward1_per_edge = torch.sigmoid(att_per_edge * neigh_prefer) - 0.5


            return reward1_per_edge


    def get_linear_sub_mean_reward1(self, guided_item):
        with torch.no_grad():
            final_user_emb, final_item_emb = self.get_emb()

            source_user_emb = final_user_emb[self.u_trust[0]]
            guided_item_emb = final_item_emb[guided_item].unsqueeze(dim=0)

            rating = (source_user_emb * guided_item_emb).sum(dim=1, keepdim=True)

            reward1_per_edge = torch.sigmoid(rating) - 0.5


            return reward1_per_edge





class mf(nn.Module):
    def __init__(self, config, dataset):
        super(mf, self).__init__()
        self.num_users = config['num_users']
        self.num_items = config['num_items']
        self.device = config['device']
        self.emb_dim = config['emb_dim']

        self.u_trust = torch.from_numpy(dataset.trust_data[:, [1, 0]]).long().t().contiguous().to(
            self.device)

        self.in_degree = degree(self.u_trust[1], self.num_users)

        self.user_embedding = nn.Embedding(num_embeddings=self.num_users, embedding_dim=self.emb_dim)
        self.item_embedding = nn.Embedding(num_embeddings=self.num_items, embedding_dim=self.emb_dim)


        print('---------------------------------mf---------------------------------------')
        print('mf init')
        print('num_users num_items emb_dim', self.num_users, self.num_items,  self.emb_dim)
        print('u_trust', self.u_trust)
        print('in_degree', self.in_degree)
        print('---------------------------------mf---------------------------------------')

    def get_emb(self):
        return self.user_embedding.weight, self.item_embedding.weight


    def get_linear_spill_reward1(self, guided_item):
        with torch.no_grad():
            final_user_emb, final_item_emb = self.get_emb()

            source_user_emb = final_user_emb[self.u_trust[0]]
            target_user_emb = final_user_emb[self.u_trust[1]]

            att_per_edge = (torch.sigmoid((source_user_emb * target_user_emb).sum(dim=1, keepdim=True))
                            / self.in_degree[self.u_trust[1]].reshape(-1, 1))

            guided_item_emb = final_item_emb[guided_item].unsqueeze(dim=0)
            neigh_prefer = (source_user_emb * guided_item_emb).sum(dim=1, keepdim=True)

            reward1_per_edge = torch.sigmoid(att_per_edge * neigh_prefer) - 0.5


            return reward1_per_edge


    def get_linear_sub_mean_reward1(self, guided_item):
        with torch.no_grad():
            final_user_emb, final_item_emb = self.get_emb()

            source_user_emb = final_user_emb[self.u_trust[0]]
            guided_item_emb = final_item_emb[guided_item].unsqueeze(dim=0)

            rating = (source_user_emb * guided_item_emb).sum(dim=1, keepdim=True)

            reward1_per_edge = torch.sigmoid(rating) - 0.5


            return reward1_per_edge




class diffnet(MessagePassing):
    def __init__(self, config, dataset):
        super().__init__(aggr='add')  # "Add" aggregation (Step 5).
        self.num_users = config['num_users']
        self.num_items = config['num_items']
        self.u_trust = torch.from_numpy(dataset.trust_data[:, [1, 0]]).long().t().contiguous().to(config['device'])
        self.in_degree = torch_geometric.utils.degree(self.u_trust[1], self.num_users).float()

        self.num_layers = config['num_layers']
        self.emb_dim = config['emb_dim']
        self.user_embedding = nn.Embedding(num_embeddings=self.num_users, embedding_dim=self.emb_dim)
        self.item_embedding = nn.Embedding(num_embeddings=self.num_items, embedding_dim=self.emb_dim)


        print('---------------------------------diffnet---------------------------------------')
        print('diffnet init')
        print('num_users num_items num_layers emb_dim', self.num_users, self.num_items, self.num_layers, self.emb_dim)
        print('u_trust', self.u_trust)
        print('in_degree', self.in_degree)
        print('---------------------------------diffnet---------------------------------------')


    def message(self, x_j):
        with torch.no_grad():
            return x_j / self.in_degree[self.u_trust[1]].reshape(-1, 1)


    def get_emb(self):
        with torch.no_grad():
            node_emb_list = [self.user_embedding.weight]

            for i in range(self.num_layers):
                temp_node_emb = self.propagate(edge_index=self.u_trust, x=node_emb_list[i])
                node_emb_list.append(temp_node_emb)

            node_emb_per_layer = torch.stack(node_emb_list, dim=1)
            final_user_emb = node_emb_per_layer.sum(dim=1)

            return final_user_emb, self.item_embedding.weight


    def get_linear_spill_reward1(self, guided_item):
        with torch.no_grad():
            final_user_emb, final_item_emb = self.get_emb()

            source_user_emb = final_user_emb[self.u_trust[0]]
            target_user_emb = final_user_emb[self.u_trust[1]]

            att_per_edge = (torch.sigmoid((source_user_emb * target_user_emb).sum(dim=1, keepdim=True))
                            / self.in_degree[self.u_trust[1]].reshape(-1, 1))

            guided_item_emb = final_item_emb[guided_item].unsqueeze(dim=0)
            neigh_prefer = (source_user_emb * guided_item_emb).sum(dim=1, keepdim=True)

            reward1_per_edge = torch.sigmoid(att_per_edge * neigh_prefer) - 0.5


            return reward1_per_edge


    def get_linear_sub_mean_reward1(self, guided_item):
        with torch.no_grad():
            final_user_emb, final_item_emb = self.get_emb()

            source_user_emb = final_user_emb[self.u_trust[0]]
            guided_item_emb = final_item_emb[guided_item].unsqueeze(dim=0)

            rating = (source_user_emb * guided_item_emb).sum(dim=1, keepdim=True)

            reward1_per_edge = torch.sigmoid(rating) - 0.5


            return reward1_per_edge


class gatv2_spill_var2(MessagePassing):
    def __init__(self, config, dataset):
        super().__init__(aggr='add')
        self.num_users = config['num_users']
        self.num_items = config['num_items']
        self.emb_dim = config['emb_dim']
        self.device = config['device']

        self.u_trust = torch.from_numpy(dataset.trust_data[:, [1, 0]]).long().t().contiguous().to(self.device)

        self.user_embedding = nn.Embedding(num_embeddings=self.num_users, embedding_dim=self.emb_dim)
        self.item_embedding = nn.Embedding(num_embeddings=self.num_items, embedding_dim=self.emb_dim)

        self.linear_att_out = nn.Linear(2 * config['emb_dim'], 1, bias=False)
        self.linear_trans_neigh = nn.Linear(config['emb_dim'], config['emb_dim'], bias=False)

        print('---------------------------------gatv2_spill_var2---------------------------------------')
        print('gatv2_spill_var2 init')
        print('num_users, num_items, emb_dim, device', self.num_users, self.num_items, self.emb_dim, self.device)
        print('u_trust', self.u_trust, self.u_trust.shape)
        print('---------------------------------gatv2_spill_var2---------------------------------------')


    def message(self, x_j, att_per_edge):
        return x_j * att_per_edge


    def click_forward(self, oi, guided_users, guided_item):
        with torch.no_grad():
            source_user_emb = self.user_embedding(self.u_trust[0])
            target_user_emb = self.user_embedding(self.u_trust[1])

            fused_user_emb = torch.cat([source_user_emb, target_user_emb], dim=1)

            att_per_edge = self.linear_att_out(fused_user_emb)
            att_per_edge = softmax_1(att_per_edge, self.u_trust[1])

            ui_emb = self.user_embedding.weight * self.item_embedding.weight[guided_item].unsqueeze(dim=0)

            g_ui_emb = self.propagate(edge_index=self.u_trust, x=ui_emb * oi, att_per_edge=att_per_edge)
            g_ui_emb = self.linear_trans_neigh(g_ui_emb)

            sg_ui_emb = torch.cat([ui_emb, g_ui_emb], dim=1)

            rating = sg_ui_emb.sum(dim=1, keepdim=True)

            return rating, 0, 0


    def get_linear_spill_reward1(self, guided_item):
        with torch.no_grad():
            source_user_emb = self.user_embedding(self.u_trust[0])
            target_user_emb = self.user_embedding(self.u_trust[1])

            fused_user_emb = torch.cat([source_user_emb, target_user_emb], dim=1)

            att_per_edge = self.linear_att_out(fused_user_emb)
            att_per_edge = softmax_1(att_per_edge, self.u_trust[1])

            guided_item_emb = self.item_embedding.weight[guided_item].unsqueeze(dim=0)

            neigh_prefer_emb = att_per_edge * source_user_emb * guided_item_emb
            reward1_per_edge = self.linear_trans_neigh(neigh_prefer_emb).sum(dim=1, keepdim=True)


            return reward1_per_edge

class simulation_ncf(MessagePassing):
    def __init__(self, config, dataset):
        super().__init__(aggr='add')  # "Add" aggregation (Step 5).
        self.num_users = config['num_users']
        self.num_items = config['num_items']
        self.device = config['device']
        self.emb_dim = config['emb_dim']
        self.beta = config['beta']
        self.indi_scale = config['indi_scale']
        self.neigh_scale = config['neigh_scale']
        self.att_scale = config['att_scale']

        self.u_trust = torch.from_numpy(dataset.trust_data[:, [1, 0]]).long().t().contiguous().to(
            self.device)

        self.in_degree = degree(self.u_trust[1], self.num_users)
        self.in_norm = torch.pow(self.in_degree, config['power'])
        self.in_norm[self.in_norm == 0.0] = 1.0

        self.user_embedding = nn.Embedding(num_embeddings=self.num_users, embedding_dim=self.emb_dim)
        self.item_embedding = nn.Embedding(num_embeddings=self.num_items, embedding_dim=self.emb_dim)

        self.linear_1 = torch.nn.Linear(self.emb_dim * 2, self.emb_dim)
        self.linear_2 = torch.nn.Linear(self.emb_dim, 1, bias=False)

        print('---------------------------------simulation_ncf---------------------------------------')
        print('simulation_ncf init')
        print('num_users num_items emb_dim beta', self.num_users, self.num_items,  self.emb_dim, self.beta)
        print('indi_scale', self.indi_scale)
        print('neigh_scale', self.neigh_scale)
        print('att_scale', self.att_scale)
        print('u_trust', self.u_trust)
        print('in_degree', self.in_degree)
        print('in_norm', self.in_norm)

        print('---------------------------------simulation_ncf---------------------------------------')


    def message(self, x_j, coeff):
        with torch.no_grad():
            return x_j * coeff

    def get_emb(self):
        return self.user_embedding.weight, self.item_embedding.weight


    def click_forward(self, oi, guided_users, guided_item):
        with torch.no_grad():
            final_user_emb, final_item_emb = self.get_emb()

            base_prefer = (final_user_emb
                           * final_item_emb[guided_item].unsqueeze(dim=0)).sum(dim=1, keepdim=True)
            indi_prefer = base_prefer * self.indi_scale
            neigh_prefer = base_prefer * self.neigh_scale
            prefer_factor = oi * neigh_prefer

            source_user_emb = final_user_emb[self.u_trust[0]]
            target_user_emb = final_user_emb[self.u_trust[1]]
            att_per_edge = (torch.sigmoid((source_user_emb * target_user_emb).sum(dim=1, keepdim=True))
                            / self.in_norm[self.u_trust[1]].reshape(-1, 1))

            neigh_effect = self.propagate(edge_index=self.u_trust, x=prefer_factor, coeff=att_per_edge)

            click_logit = indi_prefer + self.beta * neigh_effect
            guided_indi_click_prob = torch.sigmoid(indi_prefer[guided_users])
            guided_click_prob = torch.sigmoid(click_logit[guided_users])

            return click_logit, guided_indi_click_prob, guided_click_prob


    def get_linear_spill_reward1(self, guided_item):
        with torch.no_grad():
            final_user_emb, final_item_emb = self.get_emb()

            source_user_emb = final_user_emb[self.u_trust[0]]
            target_user_emb = final_user_emb[self.u_trust[1]]

            att_per_edge = (torch.sigmoid((source_user_emb * target_user_emb).sum(dim=1, keepdim=True))
                            / self.in_norm[self.u_trust[1]].reshape(-1, 1))

            guided_item_emb = final_item_emb[guided_item].unsqueeze(dim=0)
            neigh_prefer = (source_user_emb * guided_item_emb).sum(dim=1, keepdim=True) * self.neigh_scale

            reward1_per_edge = self.beta * att_per_edge * neigh_prefer


            return reward1_per_edge



class simulation_lightgcn(MessagePassing):
    def __init__(self, config, dataset):
        super().__init__(aggr='add')  # "Add" aggregation (Step 5).
        self.num_users = config['num_users']
        self.num_items = config['num_items']
        self.device = config['device']
        self.emb_dim = config['emb_dim']
        self.beta = config['beta']
        self.indi_scale = config['indi_scale']
        self.neigh_scale = config['neigh_scale']
        self.att_scale = config['att_scale']

        self.u_trust = torch.from_numpy(dataset.trust_data[:, [1, 0]]).long().t().contiguous().to(
            self.device)

        self.in_degree = degree(self.u_trust[1], self.num_users)
        self.in_norm = torch.pow(self.in_degree, config['power'])
        self.in_norm[self.in_norm == 0.0] = 1.0

        self.user_embedding = nn.Embedding(num_embeddings=self.num_users, embedding_dim=self.emb_dim)
        self.item_embedding = nn.Embedding(num_embeddings=self.num_items, embedding_dim=self.emb_dim)


        print('---------------------------------simulation_lightgcn---------------------------------------')
        print('simulation_lightgcn init')
        print('num_users num_items emb_dim beta', self.num_users, self.num_items,  self.emb_dim, self.beta)
        print('indi_scale', self.indi_scale)
        print('neigh_scale', self.neigh_scale)
        print('att_scale', self.att_scale)
        print('u_trust', self.u_trust)
        print('in_degree', self.in_degree)
        print('in_norm', self.in_norm)
        print('---------------------------------simulation_lightgcn---------------------------------------')


    def message(self, x_j, coeff):
        with torch.no_grad():
            return x_j * coeff

    def get_emb(self):
        return self.user_embedding.weight, self.item_embedding.weight


    def click_forward(self, oi, guided_users, guided_item):
        with torch.no_grad():
            final_user_emb, final_item_emb = self.get_emb()

            base_prefer = (final_user_emb
                           * final_item_emb[guided_item].unsqueeze(dim=0)).sum(dim=1, keepdim=True)
            indi_prefer = base_prefer * self.indi_scale
            neigh_prefer = base_prefer * self.neigh_scale
            prefer_factor = oi * neigh_prefer

            source_user_emb = final_user_emb[self.u_trust[0]]
            target_user_emb = final_user_emb[self.u_trust[1]]
            att_per_edge = (torch.sigmoid((source_user_emb * target_user_emb).sum(dim=1, keepdim=True))
                            / self.in_norm[self.u_trust[1]].reshape(-1, 1))

            neigh_effect = self.propagate(edge_index=self.u_trust, x=prefer_factor, coeff=att_per_edge)

            click_logit = indi_prefer + self.beta * neigh_effect
            guided_indi_click_prob = torch.sigmoid(indi_prefer[guided_users])
            guided_click_prob = torch.sigmoid(click_logit[guided_users])

            return click_logit, guided_indi_click_prob, guided_click_prob


    def get_linear_spill_reward1(self, guided_item):
        with torch.no_grad():
            final_user_emb, final_item_emb = self.get_emb()

            source_user_emb = final_user_emb[self.u_trust[0]]
            target_user_emb = final_user_emb[self.u_trust[1]]

            att_per_edge = (torch.sigmoid((source_user_emb * target_user_emb).sum(dim=1, keepdim=True))
                            / self.in_norm[self.u_trust[1]].reshape(-1, 1))

            guided_item_emb = final_item_emb[guided_item].unsqueeze(dim=0)
            neigh_prefer = (source_user_emb * guided_item_emb).sum(dim=1, keepdim=True) * self.neigh_scale

            reward1_per_edge = self.beta * att_per_edge * neigh_prefer


            return reward1_per_edge


class simulation_mf(MessagePassing):
    def __init__(self, config, dataset):
        super().__init__(aggr='add')  # "Add" aggregation (Step 5).
        self.num_users = config['num_users']
        self.num_items = config['num_items']
        self.device = config['device']
        self.emb_dim = config['emb_dim']
        self.beta = config['beta']
        self.indi_scale = config['indi_scale']
        self.neigh_scale = config['neigh_scale']
        self.att_scale = config['att_scale']

        self.u_trust = torch.from_numpy(dataset.trust_data[:, [1, 0]]).long().t().contiguous().to(
            self.device)

        self.in_degree = degree(self.u_trust[1], self.num_users)
        self.in_norm = torch.pow(self.in_degree, config['power'])
        self.in_norm[self.in_norm == 0.0] = 1.0

        self.user_embedding = nn.Embedding(num_embeddings=self.num_users, embedding_dim=self.emb_dim)
        self.item_embedding = nn.Embedding(num_embeddings=self.num_items, embedding_dim=self.emb_dim)


        print('---------------------------------simulation_mf---------------------------------------')
        print('simulation_mf init')
        print('num_users num_items emb_dim beta', self.num_users, self.num_items,  self.emb_dim, self.beta)
        print('indi_scale', self.indi_scale)
        print('neigh_scale', self.neigh_scale)
        print('att_scale', self.att_scale)
        print('u_trust', self.u_trust)
        print('in_degree', self.in_degree)
        print('in_norm', self.in_norm)

        print('---------------------------------simulation_mf---------------------------------------')


    def message(self, x_j, coeff):
        with torch.no_grad():
            return x_j * coeff

    def get_emb(self):
        return self.user_embedding.weight, self.item_embedding.weight


    def click_forward(self, oi, guided_users, guided_item):
        with torch.no_grad():
            final_user_emb, final_item_emb = self.get_emb()

            base_prefer = (final_user_emb
                           * final_item_emb[guided_item].unsqueeze(dim=0)).sum(dim=1, keepdim=True)
            indi_prefer = base_prefer * self.indi_scale
            neigh_prefer = base_prefer * self.neigh_scale
            prefer_factor = oi * neigh_prefer

            source_user_emb = final_user_emb[self.u_trust[0]]
            target_user_emb = final_user_emb[self.u_trust[1]]
            att_per_edge = (torch.sigmoid((source_user_emb * target_user_emb).sum(dim=1, keepdim=True))
                            / self.in_norm[self.u_trust[1]].reshape(-1, 1))

            neigh_effect = self.propagate(edge_index=self.u_trust, x=prefer_factor, coeff=att_per_edge)

            click_logit = indi_prefer + self.beta * neigh_effect
            guided_indi_click_prob = torch.sigmoid(indi_prefer[guided_users])
            guided_click_prob = torch.sigmoid(click_logit[guided_users])

            return click_logit, guided_indi_click_prob, guided_click_prob


    def get_linear_spill_reward1(self, guided_item):
        with torch.no_grad():
            final_user_emb, final_item_emb = self.get_emb()

            source_user_emb = final_user_emb[self.u_trust[0]]
            target_user_emb = final_user_emb[self.u_trust[1]]

            att_per_edge = (torch.sigmoid((source_user_emb * target_user_emb).sum(dim=1, keepdim=True))
                            / self.in_norm[self.u_trust[1]].reshape(-1, 1))

            guided_item_emb = final_item_emb[guided_item].unsqueeze(dim=0)
            neigh_prefer = (source_user_emb * guided_item_emb).sum(dim=1, keepdim=True) * self.neigh_scale

            reward1_per_edge = self.beta * att_per_edge * neigh_prefer


            return reward1_per_edge


class simulation_diffnet(MessagePassing):
    def __init__(self, config, dataset):
        super().__init__(aggr='add')  # "Add" aggregation (Step 5).
        self.num_users = config['num_users']
        self.num_items = config['num_items']
        self.beta = config['beta']
        self.indi_scale = config['indi_scale']
        self.neigh_scale = config['neigh_scale']
        self.att_scale = config['att_scale']
        self.u_trust = torch.from_numpy(dataset.trust_data[:, [1, 0]]).long().t().contiguous().to(config['device'])
        self.in_degree = torch_geometric.utils.degree(self.u_trust[1], self.num_users).float()
        self.in_norm = torch.pow(self.in_degree, config['power'])
        self.in_norm[self.in_norm == 0.0] = 1.0

        self.num_layers = config['num_layers']
        self.emb_dim = config['emb_dim']
        self.user_embedding = nn.Embedding(num_embeddings=self.num_users, embedding_dim=self.emb_dim)
        self.item_embedding = nn.Embedding(num_embeddings=self.num_items, embedding_dim=self.emb_dim)


        print('---------------------------------simulation_diffnet---------------------------------------')
        print('simulation_diffnet init')
        print('num_users num_items num_layers emb_dim beta', self.num_users, self.num_items, self.num_layers,
              self.emb_dim, self.beta)
        print('indi_scale', self.indi_scale)
        print('neigh_scale', self.neigh_scale)
        print('att_scale', self.att_scale)
        print('u_trust', self.u_trust)
        print('in_degree', self.in_degree)
        print('in_norm', self.in_norm)
        print('indi_scale', self.indi_scale)
        print('neigh_scale', self.neigh_scale)
        print('att_scale', self.att_scale)
        print('---------------------------------simulation_diffnet---------------------------------------')


    def message(self, x_j, coeff):
        with torch.no_grad():
            return x_j * coeff


    def get_emb(self):
        with torch.no_grad():
            node_emb_list = [self.user_embedding.weight]
            att_per_edge = 1.0 / self.in_degree[self.u_trust[1]].reshape(-1, 1)

            for i in range(self.num_layers):
                temp_node_emb = self.propagate(edge_index=self.u_trust, x=node_emb_list[i], coeff=att_per_edge)
                node_emb_list.append(temp_node_emb)

            node_emb_per_layer = torch.stack(node_emb_list, dim=1)
            final_user_emb = node_emb_per_layer.sum(dim=1)

            return final_user_emb, self.item_embedding.weight

    def click_forward(self, oi, guided_users, guided_item):
        with torch.no_grad():
            final_user_emb, final_item_emb = self.get_emb()

            base_prefer = (final_user_emb
                           * final_item_emb[guided_item].unsqueeze(dim=0)).sum(dim=1, keepdim=True)
            indi_prefer = base_prefer * self.indi_scale
            neigh_prefer = base_prefer * self.neigh_scale
            prefer_factor = oi * neigh_prefer

            source_user_emb = final_user_emb[self.u_trust[0]]
            target_user_emb = final_user_emb[self.u_trust[1]]
            att_per_edge = (torch.sigmoid((source_user_emb * target_user_emb).sum(dim=1, keepdim=True))
                            / self.in_norm[self.u_trust[1]].reshape(-1, 1))

            neigh_effect = self.propagate(edge_index=self.u_trust, x=prefer_factor, coeff=att_per_edge)

            click_logit = indi_prefer + self.beta * neigh_effect
            guided_indi_click_prob = torch.sigmoid(indi_prefer[guided_users])
            guided_click_prob = torch.sigmoid(click_logit[guided_users])

            return click_logit, guided_indi_click_prob, guided_click_prob


    def get_linear_spill_reward1(self, guided_item):
        with torch.no_grad():
            final_user_emb, final_item_emb = self.get_emb()

            source_user_emb = final_user_emb[self.u_trust[0]]
            target_user_emb = final_user_emb[self.u_trust[1]]

            att_per_edge = (torch.sigmoid((source_user_emb * target_user_emb).sum(dim=1, keepdim=True))
                            / self.in_norm[self.u_trust[1]].reshape(-1, 1))

            guided_item_emb = final_item_emb[guided_item].unsqueeze(dim=0)
            neigh_prefer = (source_user_emb * guided_item_emb).sum(dim=1, keepdim=True) * self.neigh_scale

            reward1_per_edge = self.beta * att_per_edge * neigh_prefer

            return reward1_per_edge


if __name__ == '__main__':
    print('=================simulation=====================')
    import data_loader
    from utils import set_seed
    from world import config
    set_seed(0)
    dataset = data_loader.load_data(config)
    config['num_users'] = dataset.num_users
    config['num_items'] = dataset.num_items
    test_gatv2_spill_var2 = gatv2_spill_var2(config, dataset)
    test_gatv2_spill_var2.load_state_dict(
        torch.load(
            '/data2/phang/projects/guided_rec/best_para_path/'
            'ciao100.10.20.70.250.251.010.00.75_p10_20gatv2_spill_var2_explicit_seed0_ex1_0.005_1e-05_0.001_0.01_0.001_0.001.pt',
            map_location=config['device']))
    test_gatv2_spill_var2.to(config['device'])
    att = np.loadtxt('att1.txt').reshape(-1, 1)
    source_neigh_index_wrt_uedge = np.loadtxt('/home/phang/projects/guided_rec/datasets/ciao/ciao10_full_mf_processed_ui_trust_data.txt')
    source_neigh_index_wrt_uedge = torch.from_numpy(source_neigh_index_wrt_uedge[:, 1]).long().to(config['device'])
    att = torch.from_numpy(att).float().to(config['device'])
    test_gatv2_spill_var2.get_candi_spill_reward1(1, source_neigh_index_wrt_uedge)
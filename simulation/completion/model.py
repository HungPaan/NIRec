import torch
from torch import nn

class mf(nn.Module):
    def __init__(self, config):
        super(mf, self).__init__()
        self.num_users = config['num_users']
        self.num_items = config['num_items']
        self.emb_dim = config['emb_dim']
        self.user_embedding = nn.Embedding(num_embeddings=self.num_users, embedding_dim=self.emb_dim)
        self.item_embedding = nn.Embedding(num_embeddings=self.num_items, embedding_dim=self.emb_dim)
        nn.init.xavier_normal_(self.user_embedding.weight)
        nn.init.xavier_normal_(self.item_embedding.weight)

        print('---------------------------------mf---------------------------------------')
        print('mf init')
        print('num_users num_items emb_dim', self.num_users, self.num_items,  self.emb_dim)
        for p in self.parameters():
            print('mf para', p, p.size())
        print('---------------------------------mf---------------------------------------')

    def out_forward(self, user_index, item_index):
        user_emb = self.user_embedding(user_index)
        item_emb = self.item_embedding(item_index)
        rating = (user_emb * item_emb).sum(dim=1)

        return rating

    def out_forward_mat(self):
        user_emb = self.user_embedding.weight
        item_emb = self.item_embedding.weight
        rating = torch.matmul(user_emb, item_emb.t())

        return rating

    def get_rating(self, user_index, item_index):
        with torch.no_grad():
            rating = self.out_forward(user_index, item_index)

            return rating





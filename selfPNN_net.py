import torch
from torch import nn
import numpy as np
import torch.nn.functional as F

np.random.seed(2020)
torch.manual_seed(2020)


'''
This code was originally implemented by user miracle8070 in csdn and uploaded to github, i modify some part of  it to
deal with this problem 
'''
class DNN(nn.Module):

    def __init__(self, hidden_units, dropout=0.):
        super(DNN, self).__init__()
        self.dnn_network = nn.ModuleList(
            [nn.Linear(layer[0], layer[1]) for layer in list(zip(hidden_units[:-1], hidden_units[1:]))])
        self.dropout = nn.Dropout(p=dropout)
    def forward(self, x):
        for linear in self.dnn_network:
            x = linear(x)
            relu = torch.nn.ReLU()
            x = relu(x)
        x = self.dropout(x)

        return x


class ProductLayer(nn.Module):
    def __init__(self, mode, embed_dim, field_num, hidden_units):
        super(ProductLayer, self).__init__()
        self.mode = mode
        self.w_z = nn.Parameter(torch.rand([field_num, embed_dim, hidden_units[0]]),requires_grad= True)

        if mode == 'in':
            self.w_p = nn.Parameter(torch.rand([field_num, field_num, hidden_units[0]]),requires_grad=True)
        else:
            self.w_p = nn.Parameter(torch.rand([embed_dim, embed_dim, hidden_units[0]]),requires_grad=True)

        self.l_b = torch.rand([hidden_units[0], ], requires_grad=True)

    def forward(self, z, sparse_embeds):
        # z: (z == sparse_embeds)[None, field_num, embed_dim]
        # lz part
        l_z = torch.mm(z.reshape(z.shape[0], -1),
                       self.w_z.permute((2, 0, 1)).reshape(self.w_z.shape[2], -1).T)  # (None, hidden_units[0])
        if self.mode == 'in':
            p = torch.matmul(sparse_embeds, sparse_embeds.permute((0, 2, 1)))  # [None, field_num, field_num]
        else:
            f_sum = torch.unsqueeze(torch.sum(sparse_embeds, dim=1), dim=1)  # [None, 1, embed_dim]
            p = torch.matmul(f_sum.permute((0, 2, 1)), f_sum)  # [None, embed_dim, embed_dim]

        l_p = torch.mm(p.reshape(p.shape[0], -1),
                       self.w_p.permute((2, 0, 1)).reshape(self.w_p.shape[2], -1).T)  # [None, hidden_units[0]]
        output = l_p + l_z + self.l_b
        return output


# 下面我们定义真正的PNN网络
# 这里的逻辑是底层输入（类别型特征) -> embedding层 -> product 层 -> DNN -> 输出
class PNN(nn.Module):

    def __init__(self, hidden_units, field_num = 4, mode='out', dnn_dropout=0.5, embed_dim=4, outdim=1):
        """
        DeepCrossing：
            feature_info: 特征信息（数值特征， 类别特征， 类别特征embedding映射)
            hidden_units: 列表， 全连接层的每一层神经单元个数， 这里注意一下， 第一层神经单元个数实际上是hidden_units[1]， 因为hidden_units[0]是输入层
            dropout: Dropout层的失活比例
            embed_dim: embedding的维度m
            outdim: 网络的输出维度
        """
        super(PNN, self).__init__()

        self.field_num = field_num
        self.mode = mode
        self.embed_dim = embed_dim
        self.user_emb = torch.nn.Embedding(num_embeddings=290,embedding_dim=embed_dim)
        self.user2_emb = torch.nn.Embedding(num_embeddings=290,embedding_dim=embed_dim)
        self.user3_emb = torch.nn.Embedding(num_embeddings=290,embedding_dim=embed_dim)
        self.item_emb = torch.nn.Embedding(num_embeddings=300,embedding_dim=embed_dim)
        self.item2_emb = torch.nn.Embedding(num_embeddings=300,embedding_dim=embed_dim)
        self.item3_emb = torch.nn.Embedding(num_embeddings=300,embedding_dim=embed_dim)

        # embedding层， 这里需要一个列表的形式， 因为每个类别特征都需要embedding

        # Product层
        self.product = ProductLayer(mode, embed_dim, self.field_num, hidden_units)
        # dnn 层
        self.dnn_network = DNN(hidden_units, dnn_dropout)
        self.dense_final = nn.Linear(hidden_units[-1], 1)

    def forward(self, x):
        x: list
        lis = []
        for i in x:
            lis.append(torch.cat((self.user_emb(torch.LongTensor([i[0]])),self.user2_emb(torch.LongTensor([i[0]])),
                                  self.item_emb(torch.LongTensor([i[1]])),self.item2_emb(torch.LongTensor([i[1]])),self.item3_emb(torch.LongTensor([i[1]])),),dim= 0))
        lis = tuple(lis)
        sparse_embeds = torch.stack(lis, dim=0)
        z = sparse_embeds

        # product layer
        sparse_inputs = self.product(z, sparse_embeds)
        l1 = F.relu(sparse_inputs)
        dnn_x = self.dnn_network(l1)
        sigmoid = torch.nn.Sigmoid()
        outputs = sigmoid(self.dense_final(dnn_x))
        return outputs


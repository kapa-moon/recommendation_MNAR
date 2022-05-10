# -*- coding: utf-8 -*-
import numpy as np
import torch
import pdb
from sklearn.metrics import roc_auc_score
np.random.seed(2020)
torch.manual_seed(2020)
import pdb

from dataset import load_data
from matrix_factorization import MF, MF_CVIB, MF_IPS,  MF_DR
from snips import MF_SNIPS
# from matrix_factorization import MF_SNIPS
from utils import gini_index, ndcg_func, get_user_wise_ctr, rating_mat_to_sample, binarize, shuffle, minU
mse_func = lambda x,y: np.mean((x-y)**2)
acc_func = lambda x,y: np.sum(x == y) / len(x)
from Logisticregression import LogisticRegressionModel
import torch.nn.functional as F

dataset_name = "coat"
separeter_times = 0



def separater():
    global separeter_times
    separeter_times += 1
    print(separeter_times,"----------------------------------------------------------------------")

if dataset_name == "coat":
    train_mat, test_mat = load_data("coat")

    separater()
    print(type(train_mat[0]),len(train_mat),len(train_mat[0]))

    separater()
    print(test_mat.shape)

    x_train, y_train = rating_mat_to_sample(train_mat)#数大于0的坐标矩阵,和值矩阵
    x_test, y_test = rating_mat_to_sample(test_mat)#数大于0的坐标矩阵,和值矩阵
    num_user = train_mat.shape[0]
    num_item = train_mat.shape[1]# shape[0] means how many columns shape[1] means how many rows
elif dataset_name == "yahoo":
    x_train, y_train, x_test, y_test = load_data("yahoo")
    x_train, y_train = shuffle(x_train, y_train)
    num_user = x_train[:,0].max() + 1
    num_item = x_train[:,1].max() + 1
separater()
print(f'before binarize{y_test}')

separater()
print(f"xtrain{type(train_mat)}")

separater()
print(num_item)
print(num_user)

separater()
print("# user: {}, # item: {}".format(num_user, num_item))
# binarize
y_train = binarize(y_train) # thres == 3

y_test = binarize(y_test)
print(f'after binarize{y_test}')

print("_____________",type(x_train))



# # 导入数据， 数据已经处理好了 preprocess/下
# train_set = pd.read_csv('preprocessed_data/train_set.csv')
# val_set = pd.read_csv('preprocessed_data/val_set.csv')
# test_set = pd.read_csv('preprocessed_data/test.csv')
# val_set.head()
# # 这里需要把特征分成数值型和离散型， 因为后面的模型里面离散型的特征需要embedding， 而数值型的特征直接进入了stacking层， 处理方式会不一样
# data_df = pd.concat((train_set, val_set, test_set))
#
# dense_feas = ['I'+str(i) for i in range(1, 14)]
# sparse_feas = ['C'+str(i) for i in range(1, 27)]
#
# # 定义一个稀疏特征的embedding映射， 字典{key: value}, key表示每个稀疏特征， value表示数据集data_df对应列的不同取值个数， 作为embedding输入维度
# sparse_feas_map = {}
# for key in sparse_feas:
#     sparse_feas_map[key] = data_df[key].nunique()
# feature_info = [dense_feas, sparse_feas, sparse_feas_map]  # 这里把特征信息进行封装， 建立模型的时候作为参数传入
#
# train_set.columns
# test_set.columns
#
# # 把数据构建成数据管道
# dl_train_dataset = TensorDataset(torch.tensor(train_set.drop(columns='Label').values).float(), torch.tensor(train_set['Label'].values).float())
# dl_val_dataset = TensorDataset(torch.tensor(val_set.drop(columns='Label').values).float(), torch.tensor(val_set['Label'].values).float())
#
# dl_train = DataLoader(dl_train_dataset, shuffle=True, batch_size=16)
# dl_vaild = DataLoader(dl_val_dataset, shuffle=True, batch_size=16)


# 首先， 自定义一个残差块
class Residual_block(nn.Module):
    """
    Define Residual_block
    """

    def __init__(self, hidden_unit, dim_stack):
        super(Residual_block, self).__init__()
        self.linear1 = nn.Linear(dim_stack, hidden_unit)
        self.linear2 = nn.Linear(hidden_unit, dim_stack)
        self.relu = nn.ReLU()

    def forward(self, x):
        orig_x = x.clone()
        x = self.linear1(x)
        x = self.linear2(x)
        outputs = self.relu(x + orig_x)
        return outputs


# 定义deep Crossing 网络
class DeepCrossing(nn.Module):

    def __init__(self, feature_info, hidden_units, dropout=0., embed_dim=10, output_dim=1):
        """
        DeepCrossing：
            feature_info: 特征信息（数值特征， 类别特征， 类别特征embedding映射)
            hidden_units: 列表， 隐藏单元的个数(多层残差那里的)
            dropout: Dropout层的失活比例
            embed_dim: embedding维度
        """
        super(DeepCrossing, self).__init__()
        # self.dense_feas, self.sparse_feas, self.sparse_feas_map = feature_info

        # # embedding层， 这里需要一个列表的形式， 因为每个类别特征都需要embedding
        self.embed_layers = nn.ModuleDict({
            'embed_' + str(key): nn.Embedding(num_embeddings=val, embedding_dim=embed_dim)
            for key, val in self.sparse_feas_map.items()
        })


   [1 0  0 0 0 0 0 0 0]
   [0 1 0  0 0 0 0 0 0]
   [1 4 6 7]pirce length
        # 统计embedding_dim的总维度
        embed_dim_sum = sum([embed_dim] * len(self.sparse_feas))

        # stack layers的总维度
        dim_stack = len(self.dense_feas) + embed_dim_sum

        # 残差层
        self.res_layers = nn.ModuleList([
            Residual_block(unit, dim_stack) for unit in hidden_units
        ])

        # dropout层
        self.res_dropout = nn.Dropout(dropout)

        # 线性层
        self.linear = nn.Linear(dim_stack, output_dim)

    def forward(self, x,is_training=False):
        user_idx = torch.LongTensor(x[:, 0])
        item_idx = torch.LongTensor(x[:, 1])
        U_emb = self.W(user_idx)
        V_emb = self.H(item_idx)
        out = torch.sum(U_emb.mul(V_emb), 1)

        if is_training:
            return out, U_emb, V_emb
        else:
            return out
        # dense_inputs, sparse_inputs = x[:, :13], x[:, 13:]
        # sparse_inputs = sparse_inputs.long()  # 需要转成长张量， 这个是embedding的输入要求格式
        # sparse_embeds = [self.embed_layers['embed_' + key](sparse_inputs[:, i]) for key, i in
        #                  zip(self.sparse_feas_map.keys(), range(sparse_inputs.shape[1]))]
        # sparse_embed = torch.cat(sparse_embeds, axis=-1)
        # stack = torch.cat([sparse_embed, dense_inputs], axis=-1)
        # r = stack
        # for res in self.res_layers:
        #     r = res(r)
        #
        # r = self.res_dropout(r)
        # outputs = F.sigmoid(self.linear(r))
        # return outputs

hidden_units = [256, 128, 64, 32]
net = DeepCrossing(feature_info, hidden_units)
summary(net, input_shape=(train_set.shape[1],))

# 模型的相关设置
def auc(y_pred, y_true):
    pred = y_pred.data
    y = y_true.data
    return roc_auc_score(y, pred)     # 计算AUC， 但要注意如果y只有一个类别的时候， 会报错

loss_func = nn.BCELoss()
optimizer = torch.optim.Adam(params=net.parameters(), lr=0.001)
metric_func = auc
metric_name = 'auc'

epochs = 4
log_step_freq = 10

dfhistory = pd.DataFrame(columns=["epoch", "loss", metric_name, "val_loss", "val_" + metric_name])
print('Start Training...')
nowtime = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
print('=========' * 8 + "%s" % nowtime)

for epoch in range(1, epochs + 1):
    # 训练阶段
    net.train()
    loss_sum = 0.0
    metric_sum = 0.0
    step = 1

    for step, (features, labels) in enumerate(dl_train, 1):

        # 梯度清零
        optimizer.zero_grad()

        # 正向传播
        predictions = net(features)
        loss = loss_func(predictions, labels)
        try:  # 这里就是如果当前批次里面的y只有一个类别， 跳过去
            metric = metric_func(predictions, labels)
        except ValueError:
            pass

        # 反向传播求梯度
        loss.backward()
        optimizer.step()

        # 打印batch级别日志
        loss_sum += loss.item()
        metric_sum += metric.item()
        if step % log_step_freq == 0:
            print(("[step = %d] loss: %.3f, " + metric_name + ": %.3f") %
                  (step, loss_sum / step, metric_sum / step))
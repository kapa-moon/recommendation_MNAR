# -*- coding: utf-8 -*-
import numpy as np
import torch
import pdb
from sklearn.metrics import roc_auc_score
np.random.seed(2020)
torch.manual_seed(2020)
import pdb

from dataset import load_data

# from matrix_factorization import MF_SNIPS
from utils import gini_index, ndcg_func, get_user_wise_ctr, rating_mat_to_sample, binarize, shuffle, minU
mse_func = lambda x,y: np.mean((x-y)**2)
acc_func = lambda x,y: np.sum(x == y) / len(x)
from Logisticregression import LogisticRegressionModel
import torch.nn.functional as F
from indicator_propensity import get_logis_propensity
from selfPNN_net import PNN

from  matrix_factorization import generate_total_sample
dataset_name = "coat"
separeter_times = 0
logis = True


def separater():
    global separeter_times
    separeter_times += 1
    print(separeter_times,"----------------------------------------------------------------------")

if dataset_name == "coat":
    train_mat, test_mat = load_data("coat")

    # separater()
    # print(type(train_mat[0]),len(train_mat),len(train_mat[0]))

    # separater()
    # print(test_mat.shape)

    x_train, y_train = rating_mat_to_sample(train_mat)#数大于0的坐标矩阵,和值矩阵
    x_test, y_test = rating_mat_to_sample(test_mat)#数大于0的坐标矩阵,和值矩阵
    num_user = train_mat.shape[0]
    num_item = train_mat.shape[1]# shape[0] means how many columns shape[1] means how many rows
elif dataset_name == "yahoo":
    x_train, y_train, x_test, y_test = load_data("yahoo")
    x_train, y_train = shuffle(x_train, y_train)
    num_user = x_train[:,0].max() + 1
    num_item = x_train[:,1].max() + 1
# separater()
# print(f'before binarize{y_test}')
#
# separater()
# print(f"xtrain{type(train_mat)}")
#
# separater()
# print(num_item)
# print(num_user)
#
# separater()
# print("# user: {}, # item: {}".format(num_user, num_item))
# binarize
y_train = binarize(y_train) # thres == 3

y_test = binarize(y_test)
# print(f'after binarize{y_test}')
#
# print("_____________",type(x_train))


# "MF naive"
# mf = MF(num_user, num_item)
# print(mf.W.weight,
#      mf.H.weight)
# mf.fit(x_train, y_train,
#     lr=0.01,
#     batch_size=128,
#     lamb=1e-4,
#     tol=1e-5,
#     verbose=False)
# test_pred = mf.predict(x_test)
# mse_mf = mse_func(y_test, test_pred)
# auc_mf = roc_auc_score(y_test, test_pred)
# ndcg_res = ndcg_func(mf, x_test, y_test)
#
# print("***"*5 + "[MF]" + "***"*5)
# print("[MF] test mse:", mse_func(y_test, test_pred))
# print("[MF] test auc:", auc_mf)
# print("[MF] ndcg@5:{:.6f}, ndcg@10:{:.6f}".format(
#         np.mean(ndcg_res["ndcg_5"]), np.mean(ndcg_res["ndcg_10"])))
# user_wise_ctr = get_user_wise_ctr(x_test,y_test,test_pred)
# gi,gu = gini_index(user_wise_ctr)
# print("***"*5 + "[MF]" + "***"*5)
#
# print("-------------------------------------------------------------------------------------------------")


if logis:
    y_logis = get_logis_propensity()
else:
    y_logis = 0

if not logis:
    from matrix_factorization import MF, MF_CVIB, MF_IPS,  MF_DR
else:
    from snips import MF_SNIPS,MF_DR,MF_IPS,   NCF_IPS, NCF_DR
#
#
#
#
#
# #
# #
# #
# #
"MF IPS"
mf_ips = MF_IPS(num_user, num_item)

ips_idxs = np.arange(len(y_test))
np.random.shuffle(ips_idxs)
y_ips = y_test[ips_idxs[:int(0.05 * len(ips_idxs))]]

mf_ips.fit(x_train, y_train,y_logis, y_ips=y_ips,
    lr=0.01,
    batch_size=128,
    lamb=1e-3,
    tol=1e-5,
    verbose=False,)
test_pred = mf_ips.predict(x_test)
print(f"this is *******((((((({test_pred}")
print(len(test_pred))
print(type(test_pred))
mse_mfips = mse_func(y_test, test_pred)
auc_mfips = roc_auc_score(y_test, test_pred)
ndcg_res = ndcg_func(mf_ips, x_test, y_test)

print("***"*5 + "[MF-IPS]" + "***"*5)
print("[MF-IPS] test mse:", mse_func(y_test, test_pred))
print("[MF-IPS] test auc:", auc_mfips)
print("[MF] ndcg@5:{:.6f}, ndcg@10:{:.6f}".format(
        np.mean(ndcg_res["ndcg_5"]), np.mean(ndcg_res["ndcg_10"])))
user_wise_ctr = get_user_wise_ctr(x_test,y_test,test_pred)
gi,gu = gini_index(user_wise_ctr)
print("***"*5 + "[MF-IPS]" + "***"*5)














# item embedding
# item_path = r".\data\coat\user_item_features\item_features.ascii"
# item_accumulate_dict = {}
# with open(item_path, "r") as f:
#     for line in f.readlines():
#         if line not in item_accumulate_dict.keys():
#             item_accumulate_dict[line] = [True,1]
#         else: item_accumulate_dict[line][1] += 1
# n_i_f = len(item_accumulate_dict)
# print(n_i_f)
# print(item_accumulate_dict)
#
# user_accumulate_dict = {}
# user_feature_path = r".\data\coat\user_item_features\user_features.ascii"
# with open(user_feature_path,"r") as f:
#     for line in f.readlines():
#         if line not in item_accumulate_dict.keys():
#             user_accumulate_dict[line] = [True,1]
#         else: user_accumulate_dict[line][1] +=1
# n_u_f = len(user_accumulate_dict)
# print(f"this is use feature vector numbers{n_u_f}")
#
# item_embedding_K = 2
# item_feature_embedding = torch.nn.Embedding(n_i_f,item_embedding_K )
# user_embedding_k = 2
# user_feature_embedding = torch.nn.Embedding(n_u_f,user_embedding_k )

# with open(item_path, "r") as f:
#     idx = 0
#     i_emb_list = []
#     for line in f.readlines():
#         if item_accumulate_dict[line][0]:
#             item_accumulate_dict[line][0] = False
#             item_accumulate_dict[line][1] = idx
#             i_emb_list.append(item_accumulate_dict[line][1] )
#             idx += 1
#         else:
#             i_emb_list.append(item_accumulate_dict[line][1])
#
# print(len(i_emb_list), "\n",i_emb_list)
#
#
#
# with open(user_feature_path, "r") as f:
#     idx = 0
#     u_emb_list = []
#     for line in f.readlines():
#         if user_accumulate_dict[line][0]:
#             user_accumulate_dict[line][0] = False
#             user_accumulate_dict[line][1] = idx
#             u_emb_list.append(user_accumulate_dict[line][1])
#             idx += 1
#         else:
#             u_emb_list.append(user_accumulate_dict[line][1])
# print(len(u_emb_list), "\n",u_emb_list)
#
# u_i_p_coord = []
# for ele in x_train:
#     u_i_p_coord.append([u_emb_list[ele[0]],i_emb_list[ele[1]]])
#
# u_i_p_coord = np.array(u_i_p_coord)
# u_emb_idx = torch.LongTensor(u_i_p_coord[:,0])
# i_emb_idx = torch.LongTensor(u_i_p_coord[:,1])
# U_emb = user_feature_embedding(u_emb_idx)
# I_emb = item_feature_embedding(i_emb_idx)
#
# print(U_emb)
# print(I_emb)
#
# U_I_emb = torch.cat((U_emb,I_emb), 1)
# print(U_I_emb.shape)
#
# model = LogisticRegressionModel()
# optimizer = torch.optim.Adam(model.parameters(), lr=0.05)
# y_train = torch.tensor(y_train)
#
#
#
# for epoch in range(1000):
#     y_pred = model(U_I_emb)
#     # y_pred = y_pred.detach().numpy()
#     y_pred = torch.squeeze(y_pred)
#     if epoch%10 == 0:
#         print(y_pred)
#     y_pred = y_pred.to(torch.float32)
#     y_train = y_train.to(torch.float32)
#     print(f"this is type of pred{type(y_pred)}\nshape{y_pred.shape}, train{type(y_train)}\nshape{y_train.shape}")
#     # print(y_pred)
#     # print(y_train)
#     print(sum(1/y_pred))
#     # UI = torch.tensor(UI)
#
#     # loss1 = F.mse_loss(UI,torch.tensor(87000),reduction= "mean")
#     loss= F.binary_cross_entropy(y_pred, y_train,reduction="mean")
#
#     print(epoch, loss.item())
#     optimizer.zero_grad()
#     loss.backward(retain_graph=True)
#     optimizer.step()


# y_propensity = model(U_I_emb)
# print((290*300)/(sum(1/y_propensity))*(1/y_propensity))
# print(len(u_emb_list))
# n_i_f = len(item_accumulate_dict)
# print(n_i_f)
# print(item_accumulate_dict)





# "MF-SNIPS"
# mf_snips = MF_SNIPS(num_user, num_item)
#
# ips_idxs = np.arange(len(y_test))
# np.random.shuffle(ips_idxs)
# y_ips = y_test[ips_idxs[:int(0.05 * len(ips_idxs))]]
#
# mf_snips.fit(x_train, y_train,  y_ips=y_ips,
#     lr=0.01,
#     batch_size=128,
#     lamb=1e-4,
#     tol=1e-5,
#     verbose=False)
# test_pred = mf_snips.predict(x_test)
# mse_mfsnips = mse_func(y_test, test_pred)
# auc_mfsnips = roc_auc_score(y_test, test_pred)
# ndcg_res = ndcg_func(mf_snips, x_test, y_test)
#
# print("***"*5 + "[MF-SNIPS]" + "***"*5)
# print("[MF-SNIPS] test mse:", mse_mfsnips)
# print("[MF-SNIPS] test auc:", auc_mfsnips)
# print("[MF] ndcg@5:{:.6f}, ndcg@10:{:.6f}".format(
#         np.mean(ndcg_res["ndcg_5"]), np.mean(ndcg_res["ndcg_10"])))
# user_wise_ctr = get_user_wise_ctr(x_test,y_test,test_pred)
# gi,gu = gini_index(user_wise_ctr)
# print("***"*5 + "[MF-SNIPS]" + "***"*5)








"MF DR"
mf_dr = MF_DR(num_user, num_item)

ips_idxs = np.arange(len(y_test))
np.random.shuffle(ips_idxs)
print(f"this is ips_idxs{ips_idxs}")
y_ips = y_test[ips_idxs[:int(0.05 * len(ips_idxs))]]

print(f"this is x_train in DR{x_train}, len{len(x_train)}")

mf_dr.fit(x_train, y_train, y_logis, y_ips=y_ips,
    lr=0.05,
    batch_size=128,
    lamb=1e-3,
    tol=1e-5,
    verbose=False, )
# mf_dr.fit(x_train, y_train,  y_ips=y_propensity,
#     lr=0.05,
#     batch_size=128,
#     lamb=1e-4,
#     tol=1e-5,
#     verbose=False)
test_pred = mf_dr.predict(x_test)
mse_mfdr = mse_func(y_test, test_pred)
auc_mfdr = roc_auc_score(y_test, test_pred)
ndcg_res = ndcg_func(mf_dr, x_test, y_test)

print("***"*5 + "[MF-DR]" + "***"*5)
print("[MF-DR] test mse:", mse_mfdr)
print("[MF-DR] test auc:", auc_mfdr)
print("[MF] ndcg@5:{:.6f}, ndcg@10:{:.6f}".format(
        np.mean(ndcg_res["ndcg_5"]), np.mean(ndcg_res["ndcg_10"])))
user_wise_ctr = get_user_wise_ctr(x_test,y_test,test_pred)
gi,gu = gini_index(user_wise_ctr)
print("***"*5 + "[MF-DR]" + "***"*5)


# "NCF IPS"
# ncf_ips = NCF_IPS(num_user, num_item)
#
# ips_idxs = np.arange(len(y_test))
# np.random.shuffle(ips_idxs)
# y_ips = y_test[ips_idxs[:int(0.05 * len(ips_idxs))]]
#
# ncf_ips.fit(x_train, y_train,y_logis,
#     y_ips=y_ips,
#     lr=0.01,
#     batch_size=512,
#     lamb=1e-3,tol=1e-6, verbose=1)
#
# ndcg_res = ndcg_func(ncf_ips, x_test, y_test)
# test_pred = ncf_ips.predict(x_test)
# mse_ncfips = mse_func(y_test, test_pred)
# auc_ncfips = roc_auc_score(y_test, test_pred)
# print("***"*5 + "[NCF-IPS]" + "***"*5)
# print("[NCF-IPS] test mse:", mse_ncfips)
# print("[NCF-IPS] test auc:", auc_ncfips)
# print("ndcg@5:{:.6f}, ndcg@10:{:.6f}".format(
#     np.mean(ndcg_res["ndcg_5"]), np.mean(ndcg_res["ndcg_10"])))
# user_wise_ctr = get_user_wise_ctr(x_test,y_test,test_pred)
# gi,gu = gini_index(user_wise_ctr)
# print("***"*5 + "[NCF-IPS]" + "***"*5)
#
#
#
# "NCF DR"
# ncf_dr = NCF_DR(num_user, num_item)
#
# ips_idxs = np.arange(len(y_test))
# np.random.shuffle(ips_idxs)
# y_ips = y_test[ips_idxs[:int(0.05 * len(ips_idxs))]]
#
# ncf_dr.fit(x_train, y_train,y_logis, y_ips=y_ips, batch_size=512, lr=0.01, lamb=1e-3,verbose=0)
# ndcg_res = ndcg_func(ncf_dr, x_test, y_test)
#
# test_pred = ncf_dr.predict(x_test)
# mse_mfdr = mse_func(y_test, test_pred)
# auc_mfdr = roc_auc_score(y_test, test_pred)
# print("***"*5 + "[NCF-DR]" + "***"*5)
# print("[NCF-DR] test mse:", mse_mfdr)
# print("[NCF-DR] test auc:", auc_mfdr)
# print("ndcg@5:{:.6f}, ndcg@10:{:.6f}".format(
#     np.mean(ndcg_res["ndcg_5"]), np.mean(ndcg_res["ndcg_10"])))
# user_wise_ctr = get_user_wise_ctr(x_test,y_test,test_pred)
# gi,gu = gini_index(user_wise_ctr)
# print("***"*5 + "[NCF-DR]" + "***"*5)












def _compute_IPS(x, y, y_ips=None):
    if y_ips is None:
        one_over_zl = np.ones(len(y))
    else:
        py1 = y_ips.sum() / len(y_ips)
        py0 = 1 - py1
        po1 = len(x) / (x[:, 0].max() * x[:, 1].max())
        py1o1 = y.sum() / len(y)
        py0o1 = 1 - py1o1

        propensity = np.zeros(len(y))

        propensity[y == 0] = (py0o1 * po1) / py0
        propensity[y == 1] = (py1o1 * po1) / py1
        one_over_zl = 1 / propensity
    one_over_zl = torch.Tensor(one_over_zl)

    return one_over_zl

ips_idxs = np.arange(len(y_test))
np.random.shuffle(ips_idxs)
y_ips = y_test[ips_idxs[:int(0.05 * len(ips_idxs))]]
if not logis:
    one_over_zl = _compute_IPS(x_train,y_train,y_ips)




embedding_K = 5
hidden_units = [4,5,3]

pnn = PNN(hidden_units,field_num=5,embed_dim=embedding_K)



batch_size = 256
num_epoch = 60
optimizer = torch.optim.Adam(params=pnn.parameters(), lr=0.002, weight_decay=0.008)

x_all = generate_total_sample(num_user, num_item)

# idx1 = [(i[0] * 300) + i[1] for i in x_test]
# feature = U_I_emb[idx1]
# print(U_I_emb.shape)
num_sample = len(x_train)
total_batch = num_sample // batch_size
if logis:
    one_over_zl = torch.tensor(1 / y_logis)
prior_y = y_ips.mean()

for epoch in range(num_epoch):
    pnn.train()
    all_idx = np.arange(num_sample)  # 6960
    np.random.shuffle(all_idx)
    ul_idxs = np.arange(x_all.shape[0])  # 87000
    # print("*************************")
    # print(ul_idxs)
    np.random.shuffle(ul_idxs)

    epoch_loss = 0
    if epoch > 35:
        optimizer = torch.optim.Adam(params=pnn.parameters(), lr=0.0008, weight_decay=0.008)

    if epoch > 45:
        optimizer = torch.optim.Adam(params=pnn.parameters(), lr=0.0002, weight_decay=0.008)
    if epoch > 50:
        optimizer = torch.optim.Adam(params=pnn.parameters(), lr=0.0001, weight_decay=0.008)
    if epoch > 55:
        optimizer = torch.optim.Adam(params=pnn.parameters(), lr=0.00005, weight_decay=0.008)
    for idx in range(total_batch):
        # mini-batch training
        selected_idx = all_idx[batch_size * idx:(idx + 1) * batch_size]  # 6960
        sub_x = x_train[selected_idx]
        sub_y = y_train[selected_idx]

        all_idx2 = [(i[0] * 300) + i[1] for i in sub_x]
        if logis:
            inv_prop = torch.squeeze(one_over_zl[all_idx2])
        else:
            inv_prop = one_over_zl[selected_idx]
        sub_y = torch.Tensor(sub_y)

        pred = torch.squeeze(pnn(sub_x))

        x_sampled = x_all[ul_idxs[idx * batch_size:(idx + 1) * batch_size]]

        # x_sampled= [(i[0] * 300) + i[1] for i in x_sampled]
        pred_ul = torch.squeeze(pnn(x_sampled))

        xent_loss = F.binary_cross_entropy(pred, sub_y, weight=inv_prop, reduction="sum")

        imputation_y = torch.Tensor([prior_y] * selected_idx.shape[0])
        # if control == 0:
        #     print(f"this is imputation_y{imputation_y}")
        imputation_loss = F.binary_cross_entropy(pred, imputation_y, reduction="sum")

        ips_loss = (xent_loss - imputation_loss) / selected_idx.shape[0]

        # direct loss
        direct_loss = F.binary_cross_entropy(pred_ul, imputation_y, reduction="mean")

        loss = ips_loss + direct_loss

        optimizer.zero_grad()
        loss.backward(retain_graph = True)
        optimizer.step()

        epoch_loss += xent_loss.detach().numpy()
    print(epoch)
    pred = torch.squeeze(pnn(x_test))
    test_pred = pred.detach().numpy()
    print(test_pred)


    mse_mfips = mse_func(y_test, test_pred)
    auc_mfips = roc_auc_score(y_test, test_pred)
    print("***"*5 + "[PNN-DR]" + "***"*5)
    print("[PNN-DR] test mse:", mse_func(y_test, test_pred))
    print("[PNN-DR] test auc:", auc_mfips)
    user_wise_ctr = get_user_wise_ctr(x_test,y_test,test_pred)
    gi,gu = gini_index(user_wise_ctr)
    print("***"*5 + "[PNN-DR]" + "***"*5)















embedding_K = 4
hidden_units = [88,55,33]

pnn = PNN(hidden_units,field_num=5,embed_dim=embedding_K)

ips_idxs = np.arange(len(y_test))
np.random.shuffle(ips_idxs)
# y_ips = y_test[ips_idxs[:int(0.05 * len(ips_idxs))]]

batch_size = 256
num_epoch = 75
optimizer = torch.optim.Adam(params=pnn.parameters(), lr=0.002, weight_decay=0.008)

x_all = generate_total_sample(num_user, num_item)

for epoch in range(num_epoch):
    pnn.train()
    all_idx = np.arange(num_sample)  # 6960
    np.random.shuffle(all_idx)
    ul_idxs = np.arange(x_all.shape[0])  # 87000
    # print("*************************")
    # print(ul_idxs)
    np.random.shuffle(ul_idxs)

    epoch_loss = 0
    if epoch > 38:
        optimizer = torch.optim.Adam(params=pnn.parameters(), lr=0.0008, weight_decay=0.008)

    if epoch > 48:
        optimizer = torch.optim.Adam(params=pnn.parameters(), lr=0.0002, weight_decay=0.008)
    if epoch > 53:
        optimizer = torch.optim.Adam(params=pnn.parameters(), lr=0.0001, weight_decay=0.008)
    if epoch > 58:
        optimizer = torch.optim.Adam(params=pnn.parameters(), lr=0.00005, weight_decay=0.008)
    for idx in range(total_batch):
        # mini-batch training
        selected_idx = all_idx[batch_size * idx:(idx + 1) * batch_size]  # 6960
        sub_x = x_train[selected_idx]
        sub_y = y_train[selected_idx]

        all_idx2 = [(i[0] * 300) + i[1] for i in sub_x]
        if logis:
            inv_prop = torch.squeeze(one_over_zl[[(i[0] * 300) + i[1] for i in sub_x]])
        else:
            inv_prop = one_over_zl[selected_idx]
        sub_y = torch.Tensor(sub_y)

        pred = torch.squeeze(pnn(sub_x))


        xent_loss = F.binary_cross_entropy(pred, sub_y,
                                           weight=inv_prop)

        loss = xent_loss


        optimizer.zero_grad()
        loss.backward(retain_graph = True)
        optimizer.step()

        epoch_loss += xent_loss.detach().numpy()
    print(epoch)
    pred = torch.squeeze(pnn(x_test))
    test_pred = pred.detach().numpy()
    print(test_pred)


    mse_mfips = mse_func(y_test, test_pred)
    auc_mfips = roc_auc_score(y_test, test_pred)
    print("***"*5 + "[PNN-IPS]" + "***"*5)
    print("[PNN-IPS] test mse:", mse_func(y_test, test_pred))
    print("[PNN-IPS] test auc:", auc_mfips)
    user_wise_ctr = get_user_wise_ctr(x_test,y_test,test_pred)
    gi,gu = gini_index(user_wise_ctr)
    print("***"*5 + "[PNN-IPS]" + "***"*5)

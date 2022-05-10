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










#
#
#
#
# "MF IPS"
# mf_ips = MF_IPS(num_user, num_item)
#
# ips_idxs = np.arange(len(y_test))
# np.random.shuffle(ips_idxs)
# y_ips = y_test[ips_idxs[:int(0.05 * len(ips_idxs))]]
#
# mf_ips.fit(x_train, y_train,  y_ips=y_ips,
#     lr=0.01,
#     batch_size=128,
#     lamb=1e-4,
#     tol=1e-5,
#     verbose=False)
# test_pred = mf_ips.predict(x_test)
# mse_mfips = mse_func(y_test, test_pred)
# auc_mfips = roc_auc_score(y_test, test_pred)
# ndcg_res = ndcg_func(mf_ips, x_test, y_test)
#
# print("***"*5 + "[MF-IPS]" + "***"*5)
# print("[MF-IPS] test mse:", mse_func(y_test, test_pred))
# print("[MF-IPS] test auc:", auc_mfips)
# print("[MF] ndcg@5:{:.6f}, ndcg@10:{:.6f}".format(
#         np.mean(ndcg_res["ndcg_5"]), np.mean(ndcg_res["ndcg_10"])))
# user_wise_ctr = get_user_wise_ctr(x_test,y_test,test_pred)
# gi,gu = gini_index(user_wise_ctr)
# print("***"*5 + "[MF-IPS]" + "***"*5)














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





"MF-SNIPS"
mf_snips = MF_SNIPS(num_user, num_item)

ips_idxs = np.arange(len(y_test))
np.random.shuffle(ips_idxs)
y_ips = y_test[ips_idxs[:int(0.05 * len(ips_idxs))]]

mf_snips.fit(x_train, y_train,  y_ips=y_ips,
    lr=0.01,
    batch_size=128,
    lamb=1e-3,
    tol=1e-5,
    verbose=False)
test_pred = mf_snips.predict(x_test)
mse_mfsnips = mse_func(y_test, test_pred)
auc_mfsnips = roc_auc_score(y_test, test_pred)
ndcg_res = ndcg_func(mf_snips, x_test, y_test)

print("***"*5 + "[MF-SNIPS]" + "***"*5)
print("[MF-SNIPS] test mse:", mse_mfsnips)
print("[MF-SNIPS] test auc:", auc_mfsnips)
print("[MF] ndcg@5:{:.6f}, ndcg@10:{:.6f}".format(
        np.mean(ndcg_res["ndcg_5"]), np.mean(ndcg_res["ndcg_10"])))
user_wise_ctr = get_user_wise_ctr(x_test,y_test,test_pred)
gi,gu = gini_index(user_wise_ctr)
print("***"*5 + "[MF-SNIPS]" + "***"*5)








# "MF DR"
# mf_dr = MF_DR(num_user, num_item)
#
# ips_idxs = np.arange(len(y_test))
# np.random.shuffle(ips_idxs)
# print(f"this is ips_idxs{ips_idxs}")
# y_ips = y_test[ips_idxs[:int(0.05 * len(ips_idxs))]]
#
# print(f"this is x_train in DR{x_train}, len{len(x_train)}")
#
# # mf_dr.fit(x_train, y_train,  y_ips=y_ips,
# #     lr=0.05,
# #     batch_size=128,
# #     lamb=1e-4,
# #     tol=1e-5,
# #     verbose=False)
# mf_dr.fit(x_train, y_train,  y_ips=y_propensity,
#     lr=0.05,
#     batch_size=128,
#     lamb=1e-4,
#     tol=1e-5,
#     verbose=False)
# test_pred = mf_dr.predict(x_test)
# mse_mfdr = mse_func(y_test, test_pred)
# auc_mfdr = roc_auc_score(y_test, test_pred)
# ndcg_res = ndcg_func(mf_dr, x_test, y_test)
#
# print("***"*5 + "[MF-DR]" + "***"*5)
# print("[MF-DR] test mse:", mse_mfdr)
# print("[MF-DR] test auc:", auc_mfdr)
# print("[MF] ndcg@5:{:.6f}, ndcg@10:{:.6f}".format(
#         np.mean(ndcg_res["ndcg_5"]), np.mean(ndcg_res["ndcg_10"])))
# user_wise_ctr = get_user_wise_ctr(x_test,y_test,test_pred)
# gi,gu = gini_index(user_wise_ctr)
# print("***"*5 + "[MF-DR]" + "***"*5)
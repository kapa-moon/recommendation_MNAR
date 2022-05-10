# -*- coding: utf-8 -*-
import numpy as np
import torch

np.random.seed(2020)
torch.manual_seed(2020)


from dataset import load_data
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



def get_logis_propensity():
    if dataset_name == "coat":
        train_mat, test_mat = load_data("coat")
        x_train, y_train = rating_mat_to_sample(train_mat)
        x_test, y_test = rating_mat_to_sample(test_mat)#show the user—item pair's cordinate which we can observed
        num_user = train_mat.shape[0]
        num_item = train_mat.shape[1]# shape[0] means how many columns shape[1] means how many rows

    indicator_train_mat = binarize(train_mat)
    test_mat_low = [i for j in test_mat for i in j]
    indicator_test_mat = binarize(np.array(test_mat_low),thres=1)

    # item embedding
    item_path = r"./data/coat/user_item_features/item_features.ascii"
    item_accumulate_dict = {}
    with open(item_path, "r") as f:
        for line in f.readlines():
            if line not in item_accumulate_dict.keys():
                item_accumulate_dict[line] = [True,1]
            else: item_accumulate_dict[line][1] += 1
    n_i_f = len(item_accumulate_dict)#the number of different items

    user_accumulate_dict = {}
    user_feature_path = r"./data/coat/user_item_features/user_features.ascii"
    with open(user_feature_path,"r") as f:
        for line in f.readlines():
            if line not in item_accumulate_dict.keys():
                user_accumulate_dict[line] = [True,1]
            else: user_accumulate_dict[line][1] +=1
    n_u_f = len(user_accumulate_dict)#number of different users
    # print(f"this is use feature vector numbers{n_u_f}")

    item_embedding_K = 4
    item_feature_embedding = torch.nn.Embedding(n_i_f,item_embedding_K )
    user_embedding_k = 3
    user_feature_embedding = torch.nn.Embedding(n_u_f,user_embedding_k )


    with open(item_path, "r") as f:
        idx = 0
        i_emb_list = []
        for line in f.readlines():
            if item_accumulate_dict[line][0]:
                item_accumulate_dict[line][0] = False
                item_accumulate_dict[line][1] = idx
                i_emb_list.append(item_accumulate_dict[line][1] )
                idx += 1
            else:
                i_emb_list.append(item_accumulate_dict[line][1])

    # print(len(i_emb_list), "\n",i_emb_list)



    with open(user_feature_path, "r") as f:
        idx = 0
        u_emb_list = []
        for line in f.readlines():
            if user_accumulate_dict[line][0]:
                user_accumulate_dict[line][0] = False
                user_accumulate_dict[line][1] = idx
                u_emb_list.append(user_accumulate_dict[line][1])
                idx += 1
            else:
                u_emb_list.append(user_accumulate_dict[line][1])
    # print(len(u_emb_list), "\n",u_emb_list)

    u_i_p_coord = []
    all_u_i_embedding = []
    for i in u_emb_list:
        for j in i_emb_list:
            all_u_i_embedding.append([i,j])

    all_u_i_embedding = np.array(all_u_i_embedding)
    u_emb_idx = torch.LongTensor(all_u_i_embedding[:,0])
    i_emb_idx = torch.LongTensor(all_u_i_embedding[:,1])
    U_emb = user_feature_embedding(u_emb_idx)
    I_emb = item_feature_embedding(i_emb_idx)
    # print("**********************************************************")
    # print(f"this is U_emb{U_emb}")
    # print(f"this is I_emb{I_emb}")


    U_I_emb = torch.cat((U_emb,I_emb), 1)
    # print(U_I_emb.shape)




    model = LogisticRegressionModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.05, weight_decay= 0.05)
    indicator_test_tensor = torch.squeeze(torch.tensor(indicator_test_mat).to(torch.float32))

    for epoch in range(7):
        y_pred = model(U_I_emb)
        y_pred = torch.squeeze(y_pred)
        if epoch == 4:
            optimizer = torch.optim.Adam(model.parameters(), lr=0.04, weight_decay= 0.05)
        if epoch%7 == 0:
            optimizer = torch.optim.Adam(model.parameters(), lr=0.02, weight_decay= 0.05)
        if epoch%10 == 0:
            optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay= 0.05)
        y_pred = y_pred.to(torch.float32)
        loss_regularization2 = sum(1/y_pred)/87000/2
        loss_pre_y= F.binary_cross_entropy(y_pred, indicator_test_tensor,reduction="mean")
        loss = loss_regularization2 + loss_pre_y
        print(epoch, loss.item())
        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        optimizer.step()
    return model(U_I_emb).detach().numpy()


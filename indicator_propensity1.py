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
        train_mat, test_mat = load_data("coat")# trian_maat is the original rating matrix

        separater()
        print(type(train_mat[0]),len(train_mat),len(train_mat[0]))

        separater()
        print(test_mat.shape)

        x_train, y_train = rating_mat_to_sample(train_mat)#show the user—item par's cordinate which we can observed
        x_test, y_test = rating_mat_to_sample(test_mat)
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
    print(num_item)#still a little bit confused cuz t
    print(num_user)

    separater()
    print("# user: {}, # item: {}".format(num_user, num_item))
    # binarize
    y_train = binarize(y_train) # y->1,2,we consider it is small,in this function we just sperate the score into 2 parts, small one and the big one
    y_test = binarize(y_test)

    indicator_train_mat = binarize(train_mat)
    test_mat_low = [i for j in test_mat for i in j]#confused!是不是这个就是表示的是一个test rating matrix
    # print("pppppppppppppppppppp",test_mat_low)
    indicator_test_mat = binarize(np.array(test_mat_low),thres=1)
    # print("___-------------------------------------")
    # print(sum(test_mat))




    # item embedding
    item_path = r"./data/coat/user_item_features/item_features.ascii"
    item_accumulate_dict = {}
    with open(item_path, "r") as f:
        for line in f.readlines():
            if line not in item_accumulate_dict.keys():
                item_accumulate_dict[line] = [True,1]
            else: item_accumulate_dict[line][1] += 1
    n_i_f = len(item_accumulate_dict)#the number of different items
    print(n_i_f)
    print(item_accumulate_dict)

    user_accumulate_dict = {}
    user_feature_path = r"./data/coat/user_item_features/user_features.ascii"
    with open(user_feature_path,"r") as f:
        for line in f.readlines():
            if line not in item_accumulate_dict.keys():
                user_accumulate_dict[line] = [True,1]
            else: user_accumulate_dict[line][1] +=1
    n_u_f = len(user_accumulate_dict)#number of different users
    # print(f"this is use feature vector numbers{n_u_f}")

    item_embedding_K = 5
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

    print(len(i_emb_list), "\n",i_emb_list)



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
    print(len(u_emb_list), "\n",u_emb_list)

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
    print("**********************************************************")
    print(f"this is U_emb{U_emb}")
    print(f"this is I_emb{I_emb}")


    U_I_emb = torch.cat((U_emb,I_emb), 1)
    print(U_I_emb.shape)




    model = LogisticRegressionModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.05)
    y_train = torch.tensor(y_train)

    # indicator_test_mat = np.squeeze(indicator_train_mat)
    indicator_test_tensor = torch.squeeze(torch.tensor(indicator_test_mat).to(torch.float32))

    print(indicator_test_tensor)
    print(indicator_test_tensor.shape)
    for epoch in range(39):
        y_pred = model(U_I_emb)
        # y_pred = y_pred.detach().numpy()
        y_pred = torch.squeeze(y_pred)
        if epoch%10 == 0:
            print(y_pred)
        y_pred = y_pred.to(torch.float32)
        # print(f"this is type of pred{type(y_pred)}\nshape{y_pred.shape}, train{type(y_train)}\nshape{y_train.shape}")
        # print(y_pred)
        # print(y_train)
        print(sum(1/y_pred))
        # UI = torch.tensor(UI)

        # loss1 = F.mse_loss(UI,torch.tensor(87000),reduction= "mean")
        loss= F.binary_cross_entropy(y_pred, indicator_test_tensor,reduction="mean")

        print(epoch, loss.item())
        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        optimizer.step()
    return model(U_I_emb).detach().numpy()

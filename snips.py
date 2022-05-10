import numpy as np
import torch
torch.manual_seed(2020)
from torch import nn
import torch.nn.functional as F
from indicator_propensity import get_logis_propensity
control = 0
import pdb


def generate_total_sample(num_user, num_item):
    sample = []
    for i in range(num_user):
        sample.extend([[i,j] for j in range(num_item)])
    return np.array(sample)


class MF_SNIPS(nn.Module):
    def __init__(self, num_users, num_items, embedding_k=4, *args, **kwargs):
        super(MF_SNIPS, self).__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_k = embedding_k
        self.W = torch.nn.Embedding(self.num_users, self.embedding_k)
        self.H = torch.nn.Embedding(self.num_items, self.embedding_k)

        self.sigmoid = torch.nn.Sigmoid()
        self.xent_func = torch.nn.BCELoss()

    def forward(self, x, is_training=False):
        user_idx = torch.LongTensor(x[:, 0])
        item_idx = torch.LongTensor(x[:, 1])
        U_emb = self.W(user_idx)
        V_emb = self.H(item_idx)

        out = torch.sum(U_emb.mul(V_emb), 1)

        if is_training:
            return out, U_emb, V_emb
        else:
            return out

    def fit(self, x, y,y_logis, y_ips=None,
            num_epoch=1000, batch_size=128, lr=0.05, lamb=0,
            tol=1e-4, verbose=True):

        optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=lamb)
        last_loss = 1e9

        num_sample = len(x)
        total_batch = num_sample // batch_size

        # if y_ips is None:
        #     one_over_zl = self._compute_IPS(x, y)
        # else:
        #     one_over_zl = self._compute_IPS(x, y, y_ips)
        one_over_zl = torch.tensor(1 / y_logis)
        early_stop = 0
        for epoch in range(num_epoch):
            all_idx = np.arange(num_sample)
            np.random.shuffle(all_idx)
            epoch_loss = 0

            for idx in range(total_batch):
                # mini-batch training
                selected_idx = all_idx[batch_size * idx:(idx + 1) * batch_size]
                sub_x = x[selected_idx]
                sub_y = y[selected_idx]

                # propensity score
                inv_prop = torch.squeeze(one_over_zl[[(i[0]*300)+i[1] for i in sub_x]])
                sum_inv_prop = torch.sum(inv_prop)

                sub_y = torch.Tensor(sub_y)

                pred, u_emb, v_emb = self.forward(sub_x, True)
                pred = self.sigmoid(pred)

                xent_loss = F.binary_cross_entropy(pred, sub_y,
                                                   weight=inv_prop, reduction="sum")

                xent_loss = xent_loss / sum_inv_prop

                loss = xent_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += xent_loss.detach().numpy()

            relative_loss_div = (last_loss - epoch_loss) / (last_loss + 1e-10)
            if relative_loss_div < tol:
                if early_stop > 5:
                    print("[MF-SNIPS] epoch:{}, xent:{}".format(epoch, epoch_loss))
                    break
                early_stop += 1

            last_loss = epoch_loss

            if epoch % 10 == 0 and verbose:
                print("[MF-SNIPS] epoch:{}, xent:{}".format(epoch, epoch_loss))

            if epoch == num_epoch - 1:
                print("[MF-SNIPS] Reach preset epochs, it seems does not converge.")
    def predict(self, x):
        pred = self.forward(x)
        pred = self.sigmoid(pred)
        return pred.detach().numpy()


class MF_IPS(nn.Module):
    def __init__(self, num_users, num_items, embedding_k=4, *args, **kwargs):
        super(MF_IPS, self).__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_k = embedding_k
        self.W = torch.nn.Embedding(self.num_users, self.embedding_k)
        self.H = torch.nn.Embedding(self.num_items, self.embedding_k)

        self.sigmoid = torch.nn.Sigmoid()
        self.xent_func = torch.nn.BCELoss()

    def forward(self, x, is_training=False):
        global control
        user_idx = torch.LongTensor(x[:, 0])
        item_idx = torch.LongTensor(x[:, 1])
        U_emb = self.W(user_idx)
        V_emb = self.H(item_idx)

        out = torch.sum(U_emb.mul(V_emb), 1)
        # if control == 0:
        #     print(f"this is the result of embedding mutiplication:\n{out}")
        if is_training:
            return out, U_emb, V_emb
        else:
            return out

    def fit(self, x, y, y_logis,y_ips=None,
            num_epoch=1000, batch_size=128, lr=0.05, lamb=0,
            tol=1e-4, verbose=True):

        optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=lamb)
        last_loss = 1e9

        num_sample = len(x)
        total_batch = num_sample // batch_size

        early_stop = 0
        # if y_ips is None:
        #     one_over_zl = self._compute_IPS(x, y)
        # else:
        #     one_over_zl = self._compute_IPS(x, y, y_ips)
        # print(f"this is one over zl {one_over_zl}")
        one_over_zl = torch.tensor(1 / y_logis)

        control = 0
        for epoch in range(num_epoch):
            all_idx = np.arange(num_sample)
            np.random.shuffle(all_idx)
            epoch_loss = 0

            for idx in range(total_batch):
                # mini-batch training
                selected_idx = all_idx[batch_size * idx:(idx + 1) * batch_size]
                sub_x = x[selected_idx]

                inv_prop = torch.squeeze(one_over_zl[[(i[0] * 300) + i[1] for i in sub_x]])

                sub_y = y[selected_idx]
                if control == 0:
                    print("this is what i want to test")
                    print(sub_x)
                    print(x)
                    print(selected_idx)
                # # propensity score
                # inv_prop = one_over_zl[selected_idx]
                if control == 0:
                    print(inv_prop)
                sub_y = torch.Tensor(sub_y)

                pred, u_emb, v_emb = self.forward(sub_x, True)
                pred = self.sigmoid(pred)

                xent_loss = F.binary_cross_entropy(pred, sub_y,
                                                   weight=inv_prop)

                loss = xent_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += xent_loss.detach().numpy()
                control += 1
            relative_loss_div = (last_loss - epoch_loss) / (last_loss + 1e-10)
            if relative_loss_div < tol:
                if early_stop > 5:
                    print("[MF-IPS] epoch:{}, xent:{}".format(epoch, epoch_loss))
                    break
                early_stop += 1

            last_loss = epoch_loss

            if epoch % 10 == 0 and verbose:
                print("[MF-IPS] epoch:{}, xent:{}".format(epoch, epoch_loss))

            if epoch == num_epoch - 1:
                print("[MF-IPS] Reach preset epochs, it seems does not converge.")

    def predict(self, x):
        pred = self.forward(x)
        pred = self.sigmoid(pred)
        return pred.detach().numpy()

    def _compute_IPS(self, x, y, y_ips=None):
        if y_ips is None:
            one_over_zl = np.ones(len(y))
        else:
            py1 = y_ips.sum() / len(y_ips)  # y_ips中1的比例
            py0 = 1 - py1  # 0的比例
            po1 = len(x) / (x[:, 0].max() * x[:, 1].max())  # 可以近似看为观测到值的比例
            # print(f'below is x\n,{x}')
            # print(f'this is x[:,0].max:{x[:,0].max()},and{x[:,1].max()}')
            py1o1 = y.sum() / len(y)  # 训练集 挑选出带值的集合中 值大于2 的比例
            py0o1 = 1 - py1o1  # 值小于3的比例

            propensity = np.zeros(len(y))
            # print(f"this is first propensity{propensity},length:{len(propensity)}")

            propensity[y == 0] = (py0o1 * po1) / py0  # 应该是让y=0的地方值变动
            # 值集中小于2的比例乘以观测到值的比例, 等于观测到小于2的值的比例(挑选)/ 测试集中观测到值小于2的比例(随机)

            propensity[y == 1] = (py1o1 * po1) / py1
            one_over_zl = 1 / propensity

        one_over_zl = torch.Tensor(one_over_zl)
        # print(f'this is what returned by propensity method {one_over_zl}')
        return one_over_zl  # 返回的是inversed


class MF_DR(nn.Module):
    def __init__(self, num_users, num_items, embedding_k=4, *args, **kwargs):
        super(MF_DR, self).__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_k = embedding_k
        self.W = torch.nn.Embedding(self.num_users, self.embedding_k)
        self.H = torch.nn.Embedding(self.num_items, self.embedding_k)

        self.sigmoid = torch.nn.Sigmoid()
        self.xent_func = torch.nn.BCELoss()

    def forward(self, x, is_training=False):
        global control
        user_idx = torch.LongTensor(x[:, 0])
        item_idx = torch.LongTensor(x[:, 1])
        U_emb = self.W(user_idx)
        V_emb = self.H(item_idx)

        out = torch.sum(U_emb.mul(V_emb), 1)
        # if control == 0:
        #     print('+++++++++++++++++++++++++++++++++++++++',type(user_idx),user_idx)
        #     print('+++++++++++++++++++++++++++++++++++++++xxxx',type(x),x)
        #     print(f"this is the result of embedding mutiplication:\n{out}")
        if is_training:
            return out, U_emb, V_emb
        else:
            return out

    def fit(self, x, y,y_logis, y_ips,
            num_epoch=1000, batch_size=128, lr=0.05, lamb=0,
            tol=1e-4, verbose=True):

        global control
        optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=lamb)
        last_loss = 1e9

        # generate all counterfactuals and factuals
        x_all = generate_total_sample(self.num_users, self.num_items)
        print(f"this is what returned by generate_total_sample:\n{x_all}")
        # 总的training matrix 坐标被填满(既有带平分的也有不带的)

        num_sample = len(x)
        # 带有平分的点的个数
        total_batch = num_sample // batch_size

        # if y_ips is None:
        #     one_over_zl = self._compute_IPS(x, y)
        # else:
        #     one_over_zl = self._compute_IPS(x, y, y_ips)
        #     print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
        #     print(one_over_zl.shape)
        #     print(sum(one_over_zl))
        one_over_zl = torch.tensor(1 / y_logis)
        # one_over_zl = 1 / y_ips

        prior_y = y_ips.mean()
        if control == 0:
            print(f"this is prior_y{prior_y}")
        # 暂时没搞明白这里的作用
        # 得到的是有评分中评分大于等于3的比例

        early_stop = 0
        for epoch in range(num_epoch):
            all_idx = np.arange(num_sample)  # 6960
            np.random.shuffle(all_idx)

            # sampling counterfactuals
            ul_idxs = np.arange(x_all.shape[0])  # 87000
            np.random.shuffle(ul_idxs)

            epoch_loss = 0

            for idx in range(total_batch):
                # mini-batch training
                selected_idx = all_idx[batch_size * idx:(idx + 1) * batch_size]  # 6960
                sub_x = x[selected_idx]
                sub_y = y[selected_idx]

                # # propensity score
                # inv_prop = one_over_zl[selected_idx]
                inv_prop = torch.squeeze(one_over_zl[[(i[0] * 300) + i[1] for i in sub_x]])
                sub_y = torch.Tensor(sub_y)

                pred, u_emb, v_emb = self.forward(sub_x, True)
                pred = self.sigmoid(pred)

                x_sampled = x_all[ul_idxs[idx * batch_size:(idx + 1) * batch_size]]
                # if control == 0:
                #
                #     print(f"this is ul_indx:{ul_idxs},x_all's shape: {x_all.shape}")
                #     print(f"this is what x sampled looked like{x_sampled}\nlen:{len(x_sampled)}")
                #     print(f"this is what x_sub looked like{sub_x}\nlen:{len(sub_x)}")
                #     print(f"this is pred\n{pred},{pred.shape}")
                #     print(f"this is sub_y\n{sub_y},{sub_y.shape}")
                pred_ul, _, _ = self.forward(x_sampled, True)
                pred_ul = self.sigmoid(pred_ul)

                xent_loss = F.binary_cross_entropy(pred, sub_y, weight=inv_prop, reduction="sum")

                imputation_y = torch.Tensor([prior_y] * selected_idx.shape[0])
                if control == 0:
                    print(f"this is imputation_y{imputation_y}")
                imputation_loss = F.binary_cross_entropy(pred, imputation_y, reduction="sum")

                ips_loss = (xent_loss - imputation_loss) / selected_idx.shape[0]

                # direct loss
                direct_loss = F.binary_cross_entropy(pred_ul, imputation_y, reduction="mean")

                loss = ips_loss + direct_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += xent_loss.detach().numpy()
                # if epoch == 0 and idx > 40:
                # print(f"this is prediction model{self.W.weight}\n this is imputation:{imputation_y}")
                if control == 0:
                    control += 1
            relative_loss_div = (last_loss - epoch_loss) / (last_loss + 1e-10)
            if relative_loss_div < tol:

                if early_stop > 5:
                    print("[MF-DR] epoch:{}, xent:{}".format(epoch, epoch_loss))
                    break
                early_stop += 1

            last_loss = epoch_loss

            if epoch % 10 == 0 and verbose:
                print("[MF-DR] epoch:{}, xent:{}".format(epoch, epoch_loss))

            if epoch == num_epoch - 1:
                print("[MF-DR] Reach preset epochs, it seems does not converge.")

    def predict(self, x):
        pred = self.forward(x)
        pred = self.sigmoid(pred)
        return pred.detach().numpy()

    def _compute_IPS(self, x, y, y_ips=None):

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




class NCF_IPS(nn.Module):
    """The neural collaborative filtering method.
    """

    def __init__(self, num_users, num_items, embedding_k=4):
        super(NCF_IPS, self).__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_k = embedding_k
        self.W = torch.nn.Embedding(self.num_users, self.embedding_k)
        self.H = torch.nn.Embedding(self.num_items, self.embedding_k)
        self.linear_1 = torch.nn.Linear(self.embedding_k * 2, self.embedding_k)
        self.relu = torch.nn.ReLU()
        self.linear_2 = torch.nn.Linear(self.embedding_k, 1, bias=False)
        self.sigmoid = torch.nn.Sigmoid()

        self.xent_func = torch.nn.BCELoss()

    def forward(self, x, is_training=False):
        user_idx = torch.LongTensor(x[:, 0])
        item_idx = torch.LongTensor(x[:, 1])
        U_emb = self.W(user_idx)
        V_emb = self.H(item_idx)

        # concat
        z_emb = torch.cat([U_emb, V_emb], axis=1)

        h1 = self.linear_1(z_emb)
        h1 = self.relu(h1)

        out = self.linear_2(h1)

        # out = torch.sum(U_emb.mul(V_emb), 1)

        if is_training:
            return out, U_emb, V_emb
        else:
            return out

    def fit(self, x, y,y_logis, y_ips=None,
            num_epoch=1000, batch_size=128,
            lr=0.05, lamb=0, tol=1e-4, verbose=0):
        optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=lamb)
        last_loss = 1e9

        num_sample = len(x)
        total_batch = num_sample // batch_size

        # if y_ips is None:
        #     one_over_zl = self._compute_IPS(x, y)
        # else:
        #     one_over_zl = self._compute_IPS(x, y, y_ips)
        one_over_zl = torch.tensor(1 / y_logis)
        early_stop = 0
        for epoch in range(num_epoch):
            all_idx = np.arange(num_sample)
            np.random.shuffle(all_idx)
            epoch_loss = 0
            for idx in range(total_batch):
                # mini-batch training
                selected_idx = all_idx[batch_size * idx:(idx + 1) * batch_size]
                sub_x = x[selected_idx]
                sub_y = torch.Tensor(y[selected_idx])

                # propensity score
                inv_prop = torch.squeeze(one_over_zl[[(i[0] * 300) + i[1] for i in sub_x]])
                sub_y = torch.Tensor(sub_y)

                pred, u_emb, v_emb = self.forward(sub_x, True)
                pred = self.sigmoid(pred)

                # x_sampled = x_all[ul_idxs[idx * batch_size:(idx + 1) * batch_size]]

                xent_loss = F.binary_cross_entropy(torch.squeeze(pred), sub_y,
                                                   weight=inv_prop)

                loss = xent_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += xent_loss.detach().numpy()

            relative_loss_div = (last_loss - epoch_loss) / (last_loss + 1e-10)
            if relative_loss_div < tol:
                if early_stop > 5:
                    print("[NCF-IPS] epoch:{}, xent:{}".format(epoch, epoch_loss))
                    break
                early_stop += 1

            last_loss = epoch_loss

            if epoch % 10 == 0 and verbose:
                print("[NCF-IPS] epoch:{}, xent:{}".format(epoch, epoch_loss))

            if epoch == num_epoch - 1:
                print("[NCF-IPS] Reach preset epochs, it seems does not converge.")

    def predict(self, x):
        pred = self.forward(x)
        pred = self.sigmoid(pred)
        return pred.detach().numpy().flatten()

    def predict_proba(self, x):
        pred = self.forward(x)
        pred = pred.reshape(-1, 1)
        pred = self.sigmoid(pred)
        return np.concatenate([1 - pred, pred], axis=1)

    def _compute_IPS(self, x, y, y_ips=None):
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


class NCF_DR(nn.Module):
    """The neural collaborative filtering method.
    """

    def __init__(self, num_users, num_items, embedding_k=4):
        super(NCF_DR, self).__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_k = embedding_k
        self.W = torch.nn.Embedding(self.num_users, self.embedding_k)
        self.H = torch.nn.Embedding(self.num_items, self.embedding_k)
        self.linear_1 = torch.nn.Linear(self.embedding_k * 2, self.embedding_k)
        self.relu = torch.nn.ReLU()
        self.linear_2 = torch.nn.Linear(self.embedding_k, 1, bias=False)
        self.sigmoid = torch.nn.Sigmoid()

        self.xent_func = torch.nn.BCELoss()

    def forward(self, x, is_training=False):
        user_idx = torch.LongTensor(x[:, 0])
        item_idx = torch.LongTensor(x[:, 1])
        U_emb = self.W(user_idx)
        V_emb = self.H(item_idx)

        # concat
        z_emb = torch.cat([U_emb, V_emb], axis=1)

        h1 = self.linear_1(z_emb)
        h1 = self.relu(h1)

        out = self.linear_2(h1)

        # out = torch.sum(U_emb.mul(V_emb), 1)

        if is_training:
            return out, U_emb, V_emb
        else:
            return out

    def fit(self, x, y,y_logis, y_ips=None,
            num_epoch=1000, batch_size=128,
            lr=0.05, lamb=0, tol=1e-4, verbose=0):
        optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=lamb)
        last_loss = 1e9

        # generate all counterfactuals and factuals
        x_all = generate_total_sample(self.num_users, self.num_items)

        num_sample = len(x)
        total_batch = num_sample // batch_size

        # if y_ips is None:
        #     one_over_zl = self._compute_IPS(x, y)
        # else:
        #     one_over_zl = self._compute_IPS(x, y, y_ips)
        one_over_zl = torch.tensor(1 / y_logis)
        prior_y = y_ips.mean()
        early_stop = 0
        for epoch in range(num_epoch):
            all_idx = np.arange(num_sample)
            np.random.shuffle(all_idx)

            # sampling counterfactuals
            ul_idxs = np.arange(x_all.shape[0])
            np.random.shuffle(ul_idxs)

            epoch_loss = 0

            for idx in range(total_batch):
                # mini-batch training
                selected_idx = all_idx[batch_size * idx:(idx + 1) * batch_size]
                sub_x = x[selected_idx]
                sub_y = y[selected_idx]
                # sub_y = [[x] for x in sub_y]
                # propensity score

                inv_prop = torch.squeeze(one_over_zl[[(i[0] * 300) + i[1] for i in sub_x]])
                sub_y = torch.Tensor(sub_y)

                pred, u_emb, v_emb = self.forward(sub_x, True)
                pred = self.sigmoid(pred)

                x_sampled = x_all[ul_idxs[idx * batch_size:(idx + 1) * batch_size]]

                pred_ul, _, _ = self.forward(x_sampled, True)
                pred_ul = self.sigmoid(pred_ul)
                # if(idx == 0 ):
                #     print("#######################################################################################")
                #     print(f"this is pred\n{pred},{pred.shape}")
                #     print(f"this is pred\n{sub_y},{sub_y.shape}")
                xent_loss = F.binary_cross_entropy(torch.squeeze(pred), sub_y, weight=inv_prop, reduction="sum")

                imputation_y = torch.unsqueeze(torch.Tensor([prior_y] * selected_idx.shape[0]), 1)
                imputation_loss = F.binary_cross_entropy(pred, imputation_y, reduction="sum")

                ips_loss = (xent_loss - imputation_loss) / selected_idx.shape[0]

                # direct loss
                direct_loss = F.binary_cross_entropy(pred_ul, imputation_y, reduction="mean")

                loss = ips_loss + direct_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += xent_loss.detach().numpy()

            relative_loss_div = (last_loss - epoch_loss) / (last_loss + 1e-10)
            if relative_loss_div < tol:
                if early_stop > 5:
                    print("[NCF-DR] epoch:{}, xent:{}".format(epoch, epoch_loss))
                    break
                early_stop += 1

            last_loss = epoch_loss

            if epoch % 10 == 0 and verbose:
                print("[NCF-DR] epoch:{}, xent:{}".format(epoch, epoch_loss))

            if epoch == num_epoch - 1:
                print("[NCF-DR] Reach preset epochs, it seems does not converge.")

    def predict(self, x):
        pred = self.forward(x)
        pred = self.sigmoid(pred)
        return pred.detach().numpy().flatten()

    def predict_proba(self, x):
        pred = self.forward(x)
        pred = pred.reshape(-1, 1)
        pred = self.sigmoid(pred)
        return np.concatenate([1 - pred, pred], axis=1)

    def _compute_IPS(self, x, y, y_ips=None):
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
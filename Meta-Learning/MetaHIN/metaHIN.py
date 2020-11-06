# Import packages
import numpy as np
import torch
from torch.nn import functional as F

# Import utility scripts
from evaluation import Evaluation
from meta_learner import MetapathLearner, MetaLearner
from embedding_init import UserEmbedding, ItemEmbedding


class HML(torch.nn.Module):
    """
    Construct MetaHIN model class
    """
    def __init__(self, config, model_name):
        """
        Initialize model class
        :param config: experiment configuration
        :param model_name: model name
        """
        super(HML, self).__init__()
        self.config = config
        self.device = torch.device("cpu")
        self.model_name = model_name

        self.item_emb = ItemEmbedding(config)
        self.user_emb = UserEmbedding(config)

        self.mp_learner = MetapathLearner(config)
        self.meta_learner = MetaLearner(config)

        self.mp_lr = config['mp_lr']
        self.local_lr = config['local_lr']
        self.emb_dim = self.config['embedding_dim']

        self.cal_metrics = Evaluation()

        self.ml_weight_len = len(self.meta_learner.update_parameters())
        self.ml_weight_name = list(self.meta_learner.update_parameters().keys())
        self.mp_weight_len = len(self.mp_learner.update_parameters())
        self.mp_weight_name = list(self.mp_learner.update_parameters().keys())

        self.transformer_liners = self.transform_mp2task()

        self.meta_optimizer = torch.optim.Adam(self.parameters(), lr=config['lr'])

    def transform_mp2task(self):
        """
        Transformation function in task-wide adaptation
        """
        liners = {}
        ml_parameters = self.meta_learner.update_parameters()
        # output_dim_of_mp = self.config['user_embedding_dim']
        output_dim_of_mp = 32  # movielens: lr=0.001, avg mp, 0.8081
        for w in self.ml_weight_name:
            liners[w.replace('.', '-')] = torch.nn.Linear(output_dim_of_mp, np.prod(ml_parameters[w].shape))
        return torch.nn.ModuleDict(liners)

    def forward(self, support_user_emb, support_item_emb, support_set_y, support_mp_user_emb, vars_dict=None):
        """
        Perform a forward pass
        """
        if vars_dict is None:
            vars_dict = self.meta_learner.update_parameters()

        support_set_y_pred = self.meta_learner(support_user_emb, support_item_emb, support_mp_user_emb, vars_dict)
        loss = F.mse_loss(support_set_y_pred, support_set_y)
        grad = torch.autograd.grad(loss, vars_dict.values(), create_graph=True)

        fast_weights = {}
        for i, w in enumerate(vars_dict.keys()):
            fast_weights[w] = vars_dict[w] - self.local_lr * grad[i]

        for idx in range(1, self.config['local_update']):  # for the current task, locally update
            support_set_y_pred = self.meta_learner(support_user_emb, support_item_emb,
                                                   support_mp_user_emb, vars_dict=fast_weights)
            loss = F.mse_loss(support_set_y_pred, support_set_y)  # calculate loss on support set
            grad = torch.autograd.grad(loss, fast_weights.values(),
                                       create_graph=True)  # calculate gradients w.r.t. model parameters

            for i, w in enumerate(fast_weights.keys()):
                fast_weights[w] = fast_weights[w] - self.local_lr * grad[i]

        return fast_weights

    def mp_update(self, support_set_x, support_set_y, support_set_mps, query_set_x, query_set_y, query_set_mps):
        """
        Mete-update the parameters of MetaPathLearner, AggLearner and MetaLearner.
        """
        # each mp
        support_mp_enhanced_user_emb_s, query_mp_enhanced_user_emb_s = [], []
        mp_task_fast_weights_s = {}
        mp_task_loss_s = {}

        mp_initial_weights = self.mp_learner.update_parameters()
        ml_initial_weights = self.meta_learner.update_parameters()

        support_user_emb = self.user_emb(support_set_x[:, self.config['item_fea_len']:])
        support_item_emb = self.item_emb(support_set_x[:, 0:self.config['item_fea_len']])
        query_user_emb = self.user_emb(query_set_x[:, self.config['item_fea_len']:])
        query_item_emb = self.item_emb(query_set_x[:, 0:self.config['item_fea_len']])

        for mp in self.config['mp']:
            support_set_mp = list(support_set_mps[mp])
            query_set_mp = list(query_set_mps[mp])
            support_neighs_emb = self.item_emb(torch.cat(support_set_mp))
            support_index_list = list(map(lambda _: _.shape[0], support_set_mp))
            query_neighs_emb = self.item_emb(torch.cat(query_set_mp))
            query_index_list = list(map(lambda _: _.shape[0], query_set_mp))

            support_mp_enhanced_user_emb = self.mp_learner(support_user_emb, support_item_emb, support_neighs_emb,
                                                           mp, support_index_list)
            support_set_y_pred = self.meta_learner(support_user_emb, support_item_emb, support_mp_enhanced_user_emb)
            loss = F.mse_loss(support_set_y_pred, support_set_y)
            grad = torch.autograd.grad(loss, mp_initial_weights.values(), create_graph=True)

            fast_weights = {}
            for i in range(self.mp_weight_len):
                weight_name = self.mp_weight_name[i]
                fast_weights[weight_name] = mp_initial_weights[weight_name] - self.mp_lr * grad[i]

            for idx in range(1, self.config['mp_update']):
                support_mp_enhanced_user_emb = self.mp_learner(support_user_emb, support_item_emb, support_neighs_emb,
                                                               mp, support_index_list, vars_dict=fast_weights)
                support_set_y_pred = self.meta_learner(support_user_emb, support_item_emb, support_mp_enhanced_user_emb)
                loss = F.mse_loss(support_set_y_pred, support_set_y)
                grad = torch.autograd.grad(loss, fast_weights.values(), create_graph=True)

                for i in range(self.mp_weight_len):
                    weight_name = self.mp_weight_name[i]
                    fast_weights[weight_name] = fast_weights[weight_name] - self.mp_lr * grad[i]

            support_mp_enhanced_user_emb = self.mp_learner(support_user_emb, support_item_emb, support_neighs_emb, mp,
                                                           support_index_list, vars_dict=fast_weights)
            support_mp_enhanced_user_emb_s.append(support_mp_enhanced_user_emb)
            query_mp_enhanced_user_emb = self.mp_learner(query_user_emb, query_item_emb, query_neighs_emb, mp,
                                                         query_index_list, vars_dict=fast_weights)
            query_mp_enhanced_user_emb_s.append(query_mp_enhanced_user_emb)

            f_fast_weights = {}
            for w, liner in self.transformer_liners.items():
                w = w.replace('-', '.')
                f_fast_weights[w] = ml_initial_weights[w] * \
                                    torch.sigmoid(liner(support_mp_enhanced_user_emb.mean(0))). \
                                        view(ml_initial_weights[w].shape)
            # f_fast_weights = None
            # # the current mp ---> task update
            mp_task_fast_weights = self.forward(support_user_emb, support_item_emb, support_set_y,
                                                support_mp_enhanced_user_emb,vars_dict=f_fast_weights)
            mp_task_fast_weights_s[mp] = mp_task_fast_weights

            query_set_y_pred = self.meta_learner(query_user_emb, query_item_emb, query_mp_enhanced_user_emb,
                                                 vars_dict=mp_task_fast_weights)
            q_loss = F.mse_loss(query_set_y_pred, query_set_y)
            mp_task_loss_s[mp] = q_loss.data  # movielens: 0.8126 dbook 0.6084
            # mp_task_loss_s[mp] = loss.data  # dbook 0.6144

        # mp_att = torch.FloatTensor([l/sum(mp_task_loss_s.values())
        #                             for l in mp_task_loss_s.values()]).to(self.device)  # movielens: 0.81
        mp_att = F.softmax(-torch.stack(list(mp_task_loss_s.values())), dim=0)  # movielens: 0.80781 lr0.001
        # mp_att = torch.FloatTensor([1.0 / len(self.config['mp'])] * len(self.config['mp'])).to(self.device)

        agg_task_fast_weights = self.aggregator(mp_task_fast_weights_s, mp_att)
        agg_mp_emb = torch.stack(query_mp_enhanced_user_emb_s, 1)
        # agg_mp_emb = torch.stack(support_mp_enhanced_user_emb_s, 1)
        query_agg_enhanced_user_emb = torch.sum(agg_mp_emb * mp_att.unsqueeze(1), 1)
        query_y_pred = self.meta_learner(query_user_emb, query_item_emb, query_agg_enhanced_user_emb,
                                         vars_dict=agg_task_fast_weights)

        loss = F.mse_loss(query_y_pred, query_set_y)
        query_y_real = query_set_y.data.cpu().numpy()
        query_y_pred = query_y_pred.data.cpu().numpy()
        mae, rmse = self.cal_metrics.prediction(query_y_real, query_y_pred)
        ndcg_5 = self.cal_metrics.ranking(query_y_real, query_y_pred, k=5)
        return loss, mae, rmse, ndcg_5

    def mp_update_mp_MAML(self, support_set_x, support_set_y, support_set_mps, query_set_x, query_set_y, query_set_mps):
        """
        MeLU + multiple meta-paths aggregation
        """
        support_mp_enhanced_user_emb_s, query_mp_enhanced_user_emb_s = [], []
        support_user_emb = self.user_emb(support_set_x[:, self.config['item_fea_len']:])
        support_item_emb = self.item_emb(support_set_x[:, 0:self.config['item_fea_len']])
        query_user_emb = self.user_emb(query_set_x[:, self.config['item_fea_len']:])
        query_item_emb = self.item_emb(query_set_x[:, 0:self.config['item_fea_len']])
        mp_task_loss_s = {}

        for mp in self.config['mp']:
            support_set_mp = list(support_set_mps[mp])
            query_set_mp = list(query_set_mps[mp])
            support_neighs_emb = self.item_emb(torch.cat(support_set_mp))
            support_index_list = map(lambda _: _.shape[0], support_set_mp)
            query_neighs_emb = self.item_emb(torch.cat(query_set_mp))
            query_index_list = map(lambda _: _.shape[0], query_set_mp)

            support_mp_enhanced_user_emb = self.mp_learner(support_user_emb, support_item_emb, support_neighs_emb, mp,
                                                           support_index_list)
            support_mp_enhanced_user_emb_s.append(support_mp_enhanced_user_emb)
            query_mp_enhanced_user_emb = self.mp_learner(query_user_emb, query_item_emb, query_neighs_emb, mp,
                                                         query_index_list)
            query_mp_enhanced_user_emb_s.append(query_mp_enhanced_user_emb)

            # query_set_y_pred = self.meta_learner(query_user_emb, query_item_emb, query_mp_enhanced_user_embs)
            # q_loss = F.mse_loss(query_set_y_pred, query_set_y)
            # mp_task_loss_s[mp] = q_loss.data

        # mp_att = F.softmax(-torch.stack(list(mp_task_loss_s.values())), dim=0)
        mp_att = torch.FloatTensor([1.0 / len(self.config['mp'])] * len(self.config['mp'])).to(self.device)  # mean

        agg_mp_emb = torch.stack(support_mp_enhanced_user_emb_s, 1)
        support_agg_enhanced_user_emb = torch.sum(agg_mp_emb * mp_att.unsqueeze(1), 1)
        agg_mp_emb = torch.stack(query_mp_enhanced_user_emb_s, 1)
        query_agg_enhanced_user_emb = torch.sum(agg_mp_emb * mp_att.unsqueeze(1), 1)

        task_fast_weights = self.forward(support_user_emb, support_item_emb, support_set_y,
                                         support_agg_enhanced_user_emb)
        query_y_pred = self.meta_learner(query_user_emb, query_item_emb, query_agg_enhanced_user_emb,
                                         vars_dict=task_fast_weights)
        loss = F.mse_loss(query_y_pred, query_set_y)
        query_y_real = query_set_y.data.cpu().numpy()
        query_y_pred = query_y_pred.data.cpu().numpy()
        mae, rmse = self.cal_metrics.prediction(query_y_real, query_y_pred)
        ndcg_5 = self.cal_metrics.ranking(query_y_real, query_y_pred, k=5)

        return loss, mae, rmse, ndcg_5

    def mp_update_multi_MAML(self, support_set_x, support_set_y, support_set_mps, query_set_x, query_set_y, query_set_mps):
        """
        multiple MAML for multiple meta-paths
        """
        loss_s = []
        mae_s, rmse_s = [], []
        ndcg_at_5 = []
        support_user_emb = self.user_emb(support_set_x[:, self.config['item_fea_len']:])
        support_item_emb = self.item_emb(support_set_x[:, 0:self.config['item_fea_len']])
        query_user_emb = self.user_emb(query_set_x[:, self.config['item_fea_len']:])
        query_item_emb = self.item_emb(query_set_x[:, 0:self.config['item_fea_len']])

        for mp in self.config['mp']:
            support_set_mp = list(support_set_mps[mp])
            query_set_mp = list(query_set_mps[mp])
            support_neighs_emb = self.item_emb(torch.cat(support_set_mp))
            support_index_list = map(lambda _: _.shape[0], support_set_mp)
            query_neighs_emb = self.item_emb(torch.cat(query_set_mp))
            query_index_list = map(lambda _: _.shape[0], query_set_mp)

            support_mp_enhanced_user_emb = self.mp_learner(support_user_emb, support_item_emb, support_neighs_emb, mp,
                                                           support_index_list)
            query_mp_enhanced_user_emb = self.mp_learner(query_user_emb, query_item_emb, query_neighs_emb, mp,
                                                         query_index_list)

            task_fast_weights = self.forward(support_user_emb, support_item_emb, support_set_y,
                                             support_mp_enhanced_user_emb)
            query_y_pred = self.meta_learner(query_user_emb, query_item_emb, query_mp_enhanced_user_emb,
                                             vars_dict=task_fast_weights)
            loss = F.mse_loss(query_y_pred, query_set_y)
            mae, rmse = self.cal_metrics.prediction(query_set_y.data.cpu().numpy(),
                                                    query_y_pred.data.cpu().numpy())
            ndcg_5 = self.cal_metrics.ranking(query_set_y.data.cpu().numpy(),
                                              query_y_pred.data.cpu().numpy(), 5)

            loss_s.append(loss)
            mae_s.append(mae)
            rmse_s.append(rmse)
            ndcg_at_5.append(ndcg_5)

        return torch.stack(loss_s).mean(0), np.mean(mae_s), np.mean(rmse_s), np.mean(ndcg_5)

    def no_MAML(self, support_set_x, support_set_y, support_set_mps, query_set_x, query_set_y, query_set_mps):
        # each mp
        support_mp_enhanced_user_emb_s, query_mp_enhanced_user_emb_s = [], []
        support_user_emb = self.user_emb(support_set_x[:, self.config['item_fea_len']:])
        support_item_emb = self.item_emb(support_set_x[:, 0:self.config['item_fea_len']])
        query_user_emb = self.user_emb(query_set_x[:, self.config['item_fea_len']:])
        query_item_emb = self.item_emb(query_set_x[:, 0:self.config['item_fea_len']])
        for mp in self.config['mp']:
            support_set_mp = list(support_set_mps[mp])
            query_set_mp = list(query_set_mps[mp])
            support_neighs_emb = self.item_emb(torch.cat(support_set_mp))
            support_index_list = map(lambda _: _.shape[0], support_set_mp)
            query_neighs_emb = self.item_emb(torch.cat(query_set_mp))
            query_index_list = map(lambda _: _.shape[0], query_set_mp)

            support_mp_enhanced_user_emb = self.mp_learner(support_user_emb, support_item_emb, support_neighs_emb, mp,
                                                           support_index_list)
            support_mp_enhanced_user_emb_s.append(support_mp_enhanced_user_emb)
            query_mp_enhanced_user_emb = self.mp_learner(query_user_emb, query_item_emb, query_neighs_emb, mp,
                                                         query_index_list)
            query_mp_enhanced_user_emb_s.append(query_mp_enhanced_user_emb)

        mp_att = torch.FloatTensor([1.0 / len(self.config['mp'])] * len(self.config['mp'])).to(self.device)  # mean
        agg_mp_emb = torch.stack(support_mp_enhanced_user_emb_s, 1)
        support_agg_enhanced_user_emb = torch.sum(agg_mp_emb * mp_att.unsqueeze(1), 1)
        agg_mp_emb = torch.stack(query_mp_enhanced_user_emb_s, 1)
        query_agg_enhanced_user_emb = torch.sum(agg_mp_emb * mp_att.unsqueeze(1), 1)

        support_y_pred = self.meta_learner(support_user_emb, support_item_emb, support_agg_enhanced_user_emb)
        support_loss = F.mse_loss(support_y_pred, support_set_y)
        support_mae, support_rmse = self.cal_metrics.prediction(support_set_y.data.cpu().numpy(),
                                                                support_y_pred.data.cpu().numpy())
        support_ndcg_5 = self.cal_metrics.ranking(support_set_y.data.cpu().numpy(),
                                                  support_y_pred.data.cpu().numpy(), 5)

        query_y_pred = self.meta_learner(query_user_emb, query_item_emb, query_agg_enhanced_user_emb)
        query_loss = F.mse_loss(query_y_pred, query_set_y)
        query_mae, query_rmse = self.cal_metrics.prediction(query_set_y.data.cpu().numpy(),
                                                            query_y_pred.data.cpu().numpy())
        query_ndcg_5 = self.cal_metrics.ranking(query_set_y.data.cpu().numpy(),
                                                query_y_pred.data.cpu().numpy(), 5)

        return (support_loss + query_loss) / 2.0, (support_mae + query_mae) / 2.0, (support_rmse + query_rmse) / 2.0, \
               (support_ndcg_5 + query_ndcg_5) / 2.0

    def global_update(self, support_xs, support_ys, support_mps, query_xs, query_ys, query_mps, device='cpu'):
        """
        Perform global update
        """
        batch_sz = len(support_xs)
        loss_s = []
        mae_s = []
        rmse_s = []
        ndcg_at_5_s = []

        for i in range(batch_sz):  # each task in a batch
            support_mp = dict(support_mps[i])  # must be dict!!!
            query_mp = dict(query_mps[i])

            for mp in self.config['mp']:
                support_mp[mp] = map(lambda x: x.to(device), support_mp[mp])
                query_mp[mp] = map(lambda x: x.to(device), query_mp[mp])

            _loss, _mae, _rmse, _ndcg_5 = self.mp_update(support_xs[i].to(device), support_ys[i].to(device), support_mp,
                                                         query_xs[i].to(device), query_ys[i].to(device), query_mp)
            # _loss, _mae, _rmse, _ndcg_5 = self.mp_update_mp_MAML(support_xs[i].to(device), support_ys[i].to(device),
            #                                                      support_mp, query_xs[i].to(device),
            #                                                      query_ys[i].to(device), query_mp)
            # _loss, _mae, _rmse, _ndcg_5 = self.mp_update_multi_MAML(support_xs[i].to(device), support_ys[i].to(device),
            #                                                         support_mp, query_xs[i].to(device),
            #                                                         query_ys[i].to(device), query_mp)
            # _loss, _mae, _rmse, _ndcg_5 = self.no_MAML(support_xs[i].to(device), support_ys[i].to(device), support_mp,
            #                                            query_xs[i].to(device), query_ys[i].to(device), query_mp)
            loss_s.append(_loss)
            mae_s.append(_mae)
            rmse_s.append(_rmse)
            ndcg_at_5_s.append(_ndcg_5)

        loss = torch.stack(loss_s).mean(0)
        mae = np.mean(mae_s)
        rmse = np.mean(rmse_s)
        ndcg_at_5 = np.mean(ndcg_at_5_s)

        self.meta_optimizer.zero_grad()
        loss.backward()
        self.meta_optimizer.step()

        return loss.cpu().data.numpy(), mae, rmse, ndcg_at_5

    def evaluation(self, support_x, support_y, support_mp, query_x, query_y, query_mp, device='cpu'):
        """
        Perform evaluation
        """
        support_mp = dict(support_mp)  # must be dict!!!
        query_mp = dict(query_mp)
        for mp in self.config['mp']:
            support_mp[mp] = map(lambda x: x.to(device), support_mp[mp])
            query_mp[mp] = map(lambda x: x.to(device), query_mp[mp])

        _, mae, rmse, ndcg_5 = self.mp_update(support_x.to(device), support_y.to(device), support_mp,
                                              query_x.to(device), query_y.to(device), query_mp)
        # _, mae, rmse, ndcg_5 = self.mp_update_mp_MAML(support_x.to(device), support_y.to(device), support_mp,
        #                                               query_x.to(device), query_y.to(device), query_mp)
        # _, mae, rmse, ndcg_5 = self.mp_update_multi_MAML(support_x.to(device), support_y.to(device), support_mp,
        #                                                  query_x.to(device), query_y.to(device), query_mp)
        # mae, rmse, ndcg_5 = self.eval_no_MAML(query_x.to(device), query_y.to(device), query_mp)

        return mae, rmse, ndcg_5

    def aggregator(self, task_weights_s, att):
        for idx, mp in enumerate(self.config['mp']):
            if idx == 0:
                att_task_weights = dict({k: v * att[idx] for k, v in task_weights_s[mp].items()})
                continue
            tmp_att_task_weights = dict({k: v * att[idx] for k, v in task_weights_s[mp].items()})
            att_task_weights = dict(zip(att_task_weights.keys(),
                                        list(map(lambda x: x[0] + x[1],
                                                 zip(att_task_weights.values(), tmp_att_task_weights.values())))))

        return att_task_weights

    def eval_no_MAML(self, query_set_x, query_set_y, query_set_mps):
        # each mp
        query_mp_enhanced_user_emb_s = []
        query_user_emb = self.user_emb(query_set_x[:, self.config['item_fea_len']:])
        query_item_emb = self.item_emb(query_set_x[:, 0:self.config['item_fea_len']])

        for mp in self.config['mp']:
            query_set_mp = list(query_set_mps[mp])
            query_neighs_emb = self.item_emb(torch.cat(query_set_mp))
            query_index_list = map(lambda _: _.shape[0], query_set_mp)
            query_mp_enhanced_user_emb = self.mp_learner(query_user_emb, query_item_emb, query_neighs_emb, mp,
                                                         query_index_list)
            query_mp_enhanced_user_emb_s.append(query_mp_enhanced_user_emb)

        mp_att = torch.FloatTensor([1.0 / len(self.config['mp'])] * len(self.config['mp'])).to(self.device)  # mean
        agg_mp_emb = torch.stack(query_mp_enhanced_user_emb_s, 1)
        query_agg_enhanced_user_emb = torch.sum(agg_mp_emb * mp_att.unsqueeze(1), 1)

        query_y_pred = self.meta_learner(query_user_emb, query_item_emb, query_agg_enhanced_user_emb)
        query_mae, query_rmse = self.cal_metrics.prediction(query_set_y.data.cpu().numpy(),
                                                            query_y_pred.data.cpu().numpy())
        query_ndcg_5 = self.cal_metrics.ranking(query_set_y.data.cpu().numpy(),
                                                query_y_pred.data.cpu().numpy(), 5)

        return query_mae, query_rmse, query_ndcg_5

    def fine_tune(self, support_x,support_y,support_mp):
        if self.cuda():
            support_x = support_x.cuda()
            support_y = support_y.cuda()
            support_mp = dict(support_mp)  # must be dict!!!

            for mp, mp_data in support_mp.items():
                support_mp[mp] = list(map(lambda x: x.cuda(), mp_data))
        support_mp_enhanced_user_emb_s = []
        support_user_emb = self.user_emb(support_x[:, self.config['item_fea_len']:])
        support_item_emb = self.item_emb(support_x[:, 0:self.config['item_fea_len']])

        for mp in self.config['mp']:
            support_set_mp = support_mp[mp]
            support_neighs_emb = self.item_emb(torch.cat(support_set_mp))
            support_index_list = map(lambda _: _.shape[0], support_set_mp)

            support_mp_enhanced_user_emb = self.mp_learner(support_user_emb, support_item_emb, support_neighs_emb, mp,
                                                           support_index_list)
            support_mp_enhanced_user_emb_s.append(support_mp_enhanced_user_emb)

        mp_att = torch.FloatTensor([1.0 / len(self.config['mp'])] * len(self.config['mp'])).to(self.device)  # mean
        agg_mp_emb = torch.stack(support_mp_enhanced_user_emb_s, 1)
        support_agg_enhanced_user_emb = torch.sum(agg_mp_emb * mp_att.unsqueeze(1), 1)

        support_y_pred = self.meta_learner(support_user_emb, support_item_emb, support_agg_enhanced_user_emb)
        support_loss = F.mse_loss(support_y_pred, support_y)

        # fine-tune
        self.meta_optimizer.zero_grad()
        support_loss.backward()
        self.meta_optimizer.step()

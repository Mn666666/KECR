import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.sparse as sp

class CKAN(nn.Module):


    def __init__(self, args, n_entity, n_relation,n_users ,n_items,ui_adj,ui_adjdency,u_adjdency,n_user,n_item,ui_adj_orign):
        super(CKAN, self).__init__()
        self._parse_args(args, n_entity, n_relation, n_users ,n_items)
        self.entity_emb = nn.Embedding(self.n_entity, self.dim)
        self.relation_emb = nn.Embedding(self.n_relation, self.dim)
        self.adj_mat = ui_adj
        self.ui_adj_orign = ui_adj_orign
        self.ui_adjdency = ui_adjdency
        self.u_adjdency = u_adjdency
        self.lightgcn_layer = 3
        self.n_item_layer = 1
        self.num_layers =2
        self.batch_size = args.batch_size
        self.n_users = n_users
        self.n_items = n_items
        self.c_temp = args.c_temp
        self.device = torch.device("cuda:" + str(args.gpu_id)) if args.use_cuda \
            else torch.device("cpu")
        self.topk = 10
        self.lambda_coeff = 0.5
        self.attention = nn.Sequential(
                nn.Linear(self.dim*2, self.dim, bias=False),
                nn.ReLU(),
                nn.Linear(self.dim, self.dim, bias=False),
                nn.ReLU(),
                nn.Linear(self.dim, 1, bias=False),
                nn.Sigmoid(),
                )
        self.get_item_level_graph_1()
        self.get_item_level_graph_2()

        self._init_weight()

                
    def forward(self,items: torch.LongTensor,user_triple_set: list,item_triple_set: list,batch = None):

        user = batch['users']  # 4096
        item = batch['items']
        labels = batch['labels']
        self.ui_embeddings = batch['ui_adjdency']
        self.u_embeddings = batch['u_adjdency']

        global user_emb,item_emb
        user_emb = self.all_embed[:self.n_users, :]
        item_emb = self.all_embed[self.n_users:, :]



        users_feature_1, item_feature_1, users_feature_2,item_feature_2 = self.propagate()


        user_loss = self.cal_loss(users_feature_1, users_feature_2)
        item_loss = self.cal_loss(item_feature_1, item_feature_2)

        bpr_loss_u = self.cal_loss_1(users_feature_1, users_feature_2)
        bpr_loss_i = self.cal_loss_1(item_feature_1, item_feature_2)

        c_losses = [user_loss, item_loss]
        global c_loss
        c_loss = sum(c_losses) / len(c_losses) + bpr_loss_u + bpr_loss_i



        interact_mat_new = self.interact_mat
        indice_old = interact_mat_new._indices()
        value_old = interact_mat_new._values()

        interact_graph = torch.sparse.FloatTensor(indice_old, value_old, torch.Size(
            [self.n_users + self.n_items, self.n_users + self.n_items]))
        user_lightgcn_emb, item_lightgcn_emb = self.light_gcn(user_emb, item_emb, interact_graph)

        global u_e_2, i_e_2
        u_e_2 = user_lightgcn_emb[user].to(self.device)
        i_e_2 = item_lightgcn_emb[item].to(self.device)

        user_embeddings = []

        user_emb_0 = self.entity_emb(user_triple_set[0][0])

        user_embeddings.append(user_emb_0.mean(dim=1))

        for i in range(self.n_layer):

            h_emb = self.entity_emb(user_triple_set[0][i])

            r_emb = self.relation_emb(user_triple_set[1][i])

            t_emb = self.entity_emb(user_triple_set[2][i])

            user_emb_i = self._knowledge_attention(h_emb, r_emb, t_emb)
            user_embeddings.append(user_emb_i)
            
        item_embeddings = []
        

        item_emb_origin = self.entity_emb(items)
        item_embeddings.append(item_emb_origin)

        item_embedding = item_emb_origin
        
        for i in range(self.n_layer):

            h_emb = self.entity_emb(item_triple_set[0][i])

            r_emb = self.relation_emb(item_triple_set[1][i])

            t_emb = self.entity_emb(item_triple_set[2][i])

            item_emb_i = self._knowledge_attention(h_emb, r_emb, t_emb)
            item_embeddings.append(item_emb_i)
        
        if self.n_layer > 0 and (self.agg == 'sum' or self.agg == 'pool'):

            item_emb_0 = self.entity_emb(item_triple_set[0][0])

            item_embeddings.append(item_emb_0.mean(dim=1))



        scores = self.predict(user_embeddings, item_embeddings)
        return scores, c_loss

    def us_aggrigate(self,u_adjdency,user_embedding):
        n = u_adjdency.shape[0]
        shape = (n,self.dim)
        Weight = torch.zeros(shape).to(self.device)
        torch.nn.init.xavier_normal_(torch.tensor(Weight).to(self.device) , gain=1)
        u1 = torch.mm(torch.tensor(u_adjdency).to(self.device),torch.tensor(Weight))
        u2 = torch.mm(user_embedding,u1.t())
        u2 = torch.mm(u2,torch.tensor(Weight))
        u2 = torch.tanh(u2)

        return u2



    def light_gcn(self, user_embedding, item_embedding, adj):
        ego_embeddings = torch.cat((user_embedding, item_embedding), dim=0)

        all_embeddings = [ego_embeddings]

        for i in range(self.lightgcn_layer):
            side_embeddings = torch.mm(adj, ego_embeddings)
            ego_embeddings = side_embeddings
            all_embeddings += [ego_embeddings]

        all_embeddings = torch.stack(all_embeddings, dim=1)
        all_embeddings = all_embeddings.mean(dim=1, keepdim=False)
        u_g_embeddings, i_g_embeddings = torch.split(all_embeddings, [self.n_users, self.n_items], dim=0)

        return u_g_embeddings, i_g_embeddings

    def predict(self, user_embeddings, item_embeddings):
        e_u = user_embeddings[0]
        e_v = item_embeddings[0]
    
        if self.agg == 'concat':
            if len(user_embeddings) != len(item_embeddings):
                raise Exception("Concat aggregator needs same length for user and item embedding")
            for i in range(1, len(user_embeddings)):
                e_u = torch.cat((self.us_aggrigate(self.u_adjdency,user_embeddings[i]), e_u),dim=-1)
            for i in range(1, len(item_embeddings)):
                e_v = torch.cat((item_embeddings[i], e_v),dim=-1)
        elif self.agg == 'sum':
            for i in range(1, len(user_embeddings)):
                e_u += self.us_aggrigate(self.u_adjdency,user_embeddings[i])
            for i in range(1, len(item_embeddings)):
                e_v += item_embeddings[i]
        elif self.agg == 'pool':
            for i in range(1, len(user_embeddings)):
                e_u = torch.max(e_u, self.us_aggrigate(self.u_adjdency,user_embeddings[i]))
            for i in range(1, len(item_embeddings)):
                e_v = torch.max(e_v,item_embeddings[i])
        else:
            raise Exception("Unknown aggregator: " + self.agg)

        e_u = torch.add(e_u ,u_e_2)
        e_v = torch.add(e_v, i_e_2)
        scores = (e_v * e_u).sum(dim=1)
        scores = torch.sigmoid(scores)
        return scores
    

    def _parse_args(self, args, n_entity, n_relation,n_users,n_items):
        self.n_entity = n_entity
        self.n_relation = n_relation
        self.n_users = n_users
        self.n_items = n_items
        self.dim = args.dim
        self.n_layer = args.n_layer
        self.agg = args.agg
        
        
    def _init_weight(self):
        # init embedding
        nn.init.xavier_uniform_(self.entity_emb.weight)
        nn.init.xavier_uniform_(self.relation_emb.weight)

        for layer in self.attention:
            if isinstance(layer,nn.Linear):
                nn.init.xavier_uniform_(layer.weight)

        initializer = nn.init.xavier_uniform_

        self.all_embed = initializer(torch.empty(self.n_users + self.n_items, self.dim))
        self.interact_mat = self._convert_sp_mat_to_sp_tensor(self.adj_mat).to()
        self.ui_adj_orign = self._convert_sp_mat_to_sp_tensor(self.ui_adj_orign).to()


    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo()
        i = torch.LongTensor([coo.row, coo.col])
        v = torch.from_numpy(coo.data).float()
        return torch.sparse.FloatTensor(i, v, coo.shape)
    
    
    def _knowledge_attention(self, h_emb, r_emb, t_emb):

        att_weights = self.attention(torch.cat((h_emb, r_emb),dim=-1)).squeeze(-1)

        att_weights_norm = F.softmax(att_weights,dim=-1)

        emb_i = torch.mul(att_weights_norm.unsqueeze(-1), t_emb)

        emb_i = emb_i.sum(dim=1)
        return emb_i




    def get_item_level_graph_1(self):
        ui_graph = self.ui_adj_orign
        modification_ratio = 0.2

        if modification_ratio != 0:
            graph = ui_graph.tocoo()
            values = self.np_edge_dropout(graph.data, modification_ratio)
            item_level_graph = sp.coo_matrix((values, (graph.row, graph.col)), shape=graph.shape).tocsr()

        self.item_level_graph_1 = self.to_tensor(self.laplace_transform(item_level_graph)).to(self.device)


    def get_item_level_graph_2(self):
        ui_graph = self.ui_adj_orign
        modification_ratio = 0.2

        if modification_ratio != 0:
            graph = ui_graph.tocoo()
            values = self.np_edge_dropout(graph.data, modification_ratio)
            item_level_graph = sp.coo_matrix((values, (graph.row, graph.col)), shape=graph.shape).tocsr()

        self.item_level_graph_2 = self.to_tensor(self.laplace_transform(item_level_graph)).to(self.device)

    def np_edge_dropout(self, value, dropout_ratio):
        mask = np.random.choice([0, 1], size=(len(value),), p=[dropout_ratio, 1 - dropout_ratio])
        value = mask * value
        return value

    def to_tensor(self , graph):
        graph = graph.tocoo()
        values = graph.data
        indices = np.vstack((graph.row, graph.col))
        graph = torch.sparse.FloatTensor(torch.LongTensor(indices), torch.FloatTensor(values),
                                        torch.Size(graph.shape))
        return graph

    def laplace_transform(self, graph):
        rowsum_sqrt = sp.diags(1 / (np.sqrt(graph.sum(axis=1).A.ravel()) + 1e-8))
        colsum_sqrt = sp.diags(1 / (np.sqrt(graph.sum(axis=0).A.ravel()) + 1e-8))
        graph = rowsum_sqrt @ graph @ colsum_sqrt

        return graph

    def one_propagate(self, graph, A_feature, B_feature):
        features = torch.cat((A_feature, B_feature), 0).to(self.device)
        all_features = [features]

        for i in range(self.num_layers):
            features = torch.spmm(graph, features)

            features = features / (i+2)
            all_features.append(F.normalize(features, p=2, dim=1))

        all_features = torch.stack(all_features, 1)
        all_features = torch.sum(all_features, dim=1).squeeze(1)

        A_feature, B_feature = torch.split(all_features, (A_feature.shape[0], B_feature.shape[0]), 0)

        return A_feature, B_feature


    def propagate(self):
        users_feature_1, items_feature_1 = self.one_propagate(self.item_level_graph_1, user_emb,
                                                                item_emb)
        users_feature_2, items_feature_2 = self.one_propagate(self.item_level_graph_2, user_emb,
                                                                item_emb)

        return users_feature_1, items_feature_1,users_feature_2,items_feature_2


    def cal_loss_1(self, users_feature, bundles_feature):

        IL_users_feature = users_feature.unsqueeze(0)
        BL_users_feature = users_feature.unsqueeze(0)

        IL_bundles_feature = bundles_feature.unsqueeze(0)
        BL_bundles_feature = bundles_feature.unsqueeze(0)


        pred = torch.sum(IL_users_feature * IL_bundles_feature, 2) + torch.sum(BL_users_feature * BL_bundles_feature, 2)
        bpr_loss = self.cal_bpr_loss(pred)

        return bpr_loss

    def cal_bpr_loss(self, pred):
        if pred.shape[1] > 2:
            negs = pred[:, 1:]
            pos = pred[:, 0].unsqueeze(1).expand_as(negs)
        else:
            negs = pred[:, 1].unsqueeze(1)
            pos = pred[:, 0].unsqueeze(1)

        loss = - torch.log(torch.sigmoid(pos - negs))
        loss = torch.mean(loss)

        return loss

    def cal_loss(self, pos, aug):

        pos = pos[:, :]
        aug = aug[:, :]

        pos = F.normalize(pos, p=2, dim=1)
        aug = F.normalize(aug, p=2, dim=1)
        pos_score = torch.sum(pos * aug, dim=1)
        ttl_score = torch.matmul(pos, aug.permute(1, 0))

        pos_score = torch.exp(pos_score / self.c_temp)
        ttl_score = torch.sum(torch.exp(ttl_score / self.c_temp), axis=1)

        c_loss = - torch.mean(torch.log(pos_score / ttl_score))

        return c_loss



    def build_adj(self, context, topk):

        n_entity = context.shape[0]
        context_norm = context.div(torch.norm(context, p=2, dim=-1, keepdim=True)).cpu()
        sim = torch.mm(context_norm, context_norm.transpose(1, 0))

        knn_val, knn_ind = torch.topk(sim, topk, dim=-1)
        knn_val, knn_ind = knn_val.to(self.device), knn_ind.to(self.device)

        y = knn_ind.reshape(-1)

        x = torch.arange(0, n_entity).unsqueeze(dim=-1).to(self.device)

        x = x.expand(n_entity, topk).reshape(-1)

        indice = torch.cat((x.unsqueeze(dim=0), y.unsqueeze(dim=0)), dim=0)
        value = knn_val.reshape(-1)
        adj_sparsity = torch.sparse.FloatTensor(indice.data, value.data, torch.Size([n_entity, n_entity])).to(self.device)


        rowsum = torch.sparse.sum(adj_sparsity, dim=1)
        d_inv_sqrt = torch.pow(rowsum, -0.5)
        d_mat_inv_sqrt_value = d_inv_sqrt._values()
        x = torch.arange(0, n_entity).unsqueeze(dim=0).to(self.device)
        x = x.expand(2, n_entity)
        d_mat_inv_sqrt_indice = x
        d_mat_inv_sqrt = torch.sparse.FloatTensor(d_mat_inv_sqrt_indice, d_mat_inv_sqrt_value, torch.Size([n_entity, n_entity]))
        L_norm = torch.sparse.mm(torch.sparse.mm(d_mat_inv_sqrt, adj_sparsity.to_dense()), d_mat_inv_sqrt.to_dense())
        return L_norm


import torch
import torch.nn as nn
from torch_geometric.nn import TopKPooling, SAGEConv, GATConv
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp




class GMKT(nn.Module):
    """
    Extension of Memory-Augmented Neural Network (MANN)
    """

    def __init__(self, config, q_q_neighbors, q_l_neighbors, l_q_neighbors, l_l_neighbors):
        super(GMKT, self).__init__()
        self.is_cuda = torch.cuda.is_available()
        self.cuda = self.is_cuda & config.cuda
        if self.cuda:
            torch.cuda.manual_seed(config.seed)
            self.device = torch.device("cuda")
            # torch.cuda.set_device(config.gpu_device)
        else:
            torch.cuda.manual_seed(config.seed)
            self.device = torch.device("cpu")
        self.metric = config.metric

        # initialize the parameters
        self.num_questions = config.num_items
        self.num_nongradable_items = config.num_nongradable_items
        self.lambda_q = config.lambda_q
        self.lambda_l = config.lambda_l
        self.embeding_size_q = config.embedding_size_q
        self.embeding_size_a = config.embedding_size_a
        self.embeding_size_l = config.embedding_size_l

        self.num_concepts = config.num_concepts
        self.key_dim = config.key_dim
        self.value_dim = config.value_dim
        self.summary_dim = config.summary_dim
        self.key_matrix = torch.Tensor(self.num_concepts, self.key_dim).to(self.device)
        self.init_std = config.init_std
        nn.init.normal_(self.key_matrix, mean=0, std=self.init_std)

        self.value_matrix_init = torch.Tensor(self.num_concepts, self.value_dim).to(self.device)
        nn.init.normal_(self.value_matrix_init, mean=0., std=self.init_std)

        self.q_q_neighbors = q_q_neighbors.to(self.device)
        self.q_l_neighbors = q_l_neighbors.to(self.device)
        self.l_q_neighbors = l_q_neighbors.to(self.device)
        self.l_l_neighbors = l_l_neighbors.to(self.device)

        self.q_q_neighbors_size = (self.q_q_neighbors != 0).sum(dim=1)
        self.q_l_neighbors_size = (self.q_l_neighbors != 0).sum(dim=1)
        self.l_q_neighbors_size = (self.l_q_neighbors != 0).sum(dim=1)
        self.l_l_neighbors_size = (self.l_l_neighbors != 0).sum(dim=1)

        # initialize the layers
        self.q_embed_matrix = nn.Embedding(num_embeddings=self.num_questions + 1,
                                           embedding_dim=self.embeding_size_q,
                                           padding_idx=0)

        self.l_embed_matrix = nn.Embedding(num_embeddings=self.num_nongradable_items + 1,
                                           embedding_dim=self.embeding_size_l,
                                           padding_idx=0)

        if self.metric == "rmse":
            self.a_embed_matrix = nn.Linear(1, self.embeding_size_a)
        else:
            self.a_embed_matrix = nn.Embedding(3, self.embeding_size_a, padding_idx=2)

        self.GNN_Q_Q = nn.Linear(self.embeding_size_q, self.embeding_size_q, bias=False)
        self.GNN_Q_L = nn.Linear(self.embeding_size_l, self.embeding_size_q,
                                 bias=False)  # Q's GNN embedding from L: L embed_size to Q
        self.GNN_L_L = nn.Linear(self.embeding_size_l, self.embeding_size_l, bias=False)
        self.GNN_L_Q = nn.Linear(self.embeding_size_q, self.embeding_size_l,
                                 bias=False)  # L's GNN embedding from Q: Q embed_size to L

        self.GNN_Q = nn.Linear(self.embeding_size_q, self.embeding_size_q, bias=True)
        self.GNN_L = nn.Linear(self.embeding_size_l, self.embeding_size_l, bias=True)

        self.mapQ_value = nn.Linear(self.embeding_size_q + self.embeding_size_a, self.value_dim)
        self.mapL_value = nn.Linear(self.embeding_size_l, self.value_dim)

        self.mapQ_key = nn.Linear(self.embeding_size_q, self.key_dim)
        self.mapL_key = nn.Linear(self.embeding_size_l, self.key_dim)

        self.mapQ_key_type = nn.Linear(self.embeding_size_q, self.key_dim)
        self.mapL_key_type = nn.Linear(self.embeding_size_l, self.key_dim)

        self.erase_E_Q = nn.Linear(self.embeding_size_q + self.embeding_size_a, self.value_dim, bias=True)
        self.erase_E_L = nn.Linear(self.embeding_size_l, self.value_dim, bias=True)
        self.add_D_Q = nn.Linear(self.embeding_size_q + self.embeding_size_a, self.value_dim, bias=True)
        self.add_D_L = nn.Linear(self.embeding_size_l, self.value_dim, bias=True)

        self.T_QQ = nn.Linear(self.value_dim, self.value_dim, bias=False)
        self.T_QL = nn.Linear(self.value_dim, self.value_dim, bias=False)
        self.T_LQ = nn.Linear(self.value_dim, self.value_dim, bias=False)
        self.T_LL = nn.Linear(self.value_dim, self.value_dim, bias=False)

        self.summary_fc = nn.Linear(self.embeding_size_q + self.value_dim, self.summary_dim)
        self.linear_out = nn.Linear(self.summary_dim, 1)
        self.linear_out_type_q = nn.Linear(self.value_dim, 1)
        self.linear_out_type_l = nn.Linear(self.value_dim, 1)

        # initialize the activiate functions
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax()

    def forward(self, q_data, a_data, l_data, d_data):
        '''
           get output of the model
           :param q_data: (batch_size, seq_len) question indexes/ids of each learning interaction, 0 represent paddings
           :param a_data: (batch_size, seq_len) student performance of each learning interaction, 2 represent paddings
           :param l_data: (batch_size, seq_len) non-assessed material indexes/ids of each learning interaction, 0 represent paddings
           :param d_data: (batch_size, seq_len) material type of each learning interaction, 0: question 1:non-assessed material
           :return:
       '''

        # inintialize M^v
        batch_size, seq_len = q_data.size(0), q_data.size(1)
        self.value_matrix = self.value_matrix_init.clone().repeat(batch_size, 1, 1)

        # get embedings of learning material and response
        q_embed_data = self.q_embed_matrix(q_data.long())
        if self.metric == 'rmse':
            a_data = torch.unsqueeze(a_data, dim=2)
            a_embed_data = self.a_embed_matrix(a_data)
        else:
            a_embed_data = self.a_embed_matrix(a_data)
        l_embed_data = self.l_embed_matrix(l_data)

        # get the neighbors of each learning material (q_t/l_t) in sequence
        q_q_neigbors_embed_data = self.q_embed_matrix(self.q_q_neighbors[q_data])
        q_l_neigbors_embed_data = self.l_embed_matrix(self.q_l_neighbors[q_data])
        l_l_neigbors_embed_data = self.l_embed_matrix(self.l_l_neighbors[l_data])
        l_q_neigbors_embed_data = self.q_embed_matrix(self.l_q_neighbors[l_data])

        # get how many neighbors for each learning material (q_t/l_t) in sequence
        q_q_neigbors_size_data = self.q_q_neighbors_size[q_data]
        q_l_neigbors_size_data = self.q_l_neighbors_size[q_data]
        l_l_neigbors_size_data = self.l_l_neighbors_size[l_data]
        l_q_neigbors_size_data = self.l_q_neighbors_size[l_data]

        # split the data seq into chunk and process sequentially
        sliced_q_embed_data = torch.chunk(q_embed_data, seq_len, dim=1)
        sliced_a_embed_data = torch.chunk(a_embed_data, seq_len, dim=1)
        sliced_l_embed_data = torch.chunk(l_embed_data, seq_len, dim=1)
        sliced_d_data = torch.chunk(d_data, seq_len, dim=1)

        sliced_q_q_neigbors_embed_data = torch.chunk(q_q_neigbors_embed_data, seq_len, dim=1)
        sliced_q_l_neigbors_embed_data = torch.chunk(q_l_neigbors_embed_data, seq_len, dim=1)
        sliced_l_l_neigbors_embed_data = torch.chunk(l_l_neigbors_embed_data, seq_len, dim=1)
        sliced_l_q_neigbors_embed_data = torch.chunk(l_q_neigbors_embed_data, seq_len, dim=1)

        sliced_q_q_neigbors_size_data = torch.chunk(q_q_neigbors_size_data, seq_len, dim=1)
        sliced_q_l_neigbors_size_data = torch.chunk(q_l_neigbors_size_data, seq_len, dim=1)
        sliced_l_q_neigbors_size_data = torch.chunk(l_q_neigbors_size_data, seq_len, dim=1)
        sliced_l_l_neigbors_size_data = torch.chunk(l_l_neigbors_size_data, seq_len, dim=1)

        batch_pred, batch_pred_type = [], []

        for i in range(1, seq_len - 1):
            # embedding layer, get material embeddings and neighbors embeddings for each time step t
            q = sliced_q_embed_data[i].squeeze(1)  # (batch_size, emebeding_size_q)
            a = sliced_a_embed_data[i].squeeze(1)
            l = sliced_l_embed_data[i].squeeze(1)
            d_t = sliced_d_data[i]
            d_t_1 = sliced_d_data[i - 1]

            q_q_nbors = sliced_q_q_neigbors_embed_data[i].squeeze(1)
            q_l_nbors = sliced_q_l_neigbors_embed_data[i].squeeze(1)
            l_q_nbors = sliced_l_q_neigbors_embed_data[i].squeeze(1)
            l_l_nbors = sliced_l_l_neigbors_embed_data[i].squeeze(1)

            q_q_nbors_size = sliced_q_q_neigbors_size_data[i]
            q_l_nbors_size = sliced_q_l_neigbors_size_data[i]
            l_q_nbors_size = sliced_l_q_neigbors_size_data[i]
            l_l_nbors_size = sliced_l_l_neigbors_size_data[i]

            # multi-activity transition graph layer, perform propagation,
            q_q_agg = (q_q_nbors.sum(dim=1) / q_q_nbors_size)
            q_l_agg = (q_l_nbors.sum(dim=1) / q_l_nbors_size)
            l_q_agg = (l_q_nbors.sum(dim=1) / l_q_nbors_size)
            l_l_agg = (l_l_nbors.sum(dim=1) / l_l_nbors_size)

            q_q_agg = self.GNN_Q_Q(torch.nan_to_num(q_q_agg))
            q_l_agg = self.GNN_Q_L(torch.nan_to_num(q_l_agg))
            l_q_agg = self.GNN_L_Q(torch.nan_to_num(l_q_agg))
            l_l_agg = self.GNN_L_L(torch.nan_to_num(l_l_agg))

            q = self.GNN_Q(q + self.lambda_q * q_q_agg + (1 - self.lambda_q) * q_l_agg)
            l = self.GNN_L(l + self.lambda_q * l_l_agg + (1 - self.lambda_l) * l_q_agg)

            # hidden knowledge transfer
            qa = torch.cat([q, a], dim=1)

            q_read_key = self.mapQ_key(q)
            l_read_key = self.mapL_key(l)

            lnmt_embedded = (
                                        1 - d_t) * q_read_key + d_t * l_read_key  # learning material embedding mapped to concept for getting knowledge

            correlation_weight = self.compute_correlation_weight(lnmt_embedded)

            self.value_matrix = self.write(correlation_weight, qa, l, d_t, d_t_1)

            # prediction layer
            q_next = sliced_q_embed_data[i + 1].squeeze(1)  # (batch_size, key_dim)
            q_read_key_next = self.mapQ_key(q_next)
            correlation_weight_next = self.compute_correlation_weight(q_read_key_next)
            read_content_next = self.read(correlation_weight_next, d_t)

            mastery_level = torch.cat([read_content_next, q_next], dim=1)
            summary_output = self.tanh(self.summary_fc(mastery_level))
            batch_sliced_pred = self.sigmoid(self.linear_out(summary_output))
            batch_pred.append(batch_sliced_pred)

            # modeling for the activity-type object
            q_read_key_type = self.mapQ_key_type(q)
            l_read_key_type = self.mapL_key_type(l)

            lnmt_embedded_type = (
                                             1 - d_t) * q_read_key_type + d_t * l_read_key_type  # learning material embedding mapped to concept for getting the type
            correlation_weight_type = self.compute_correlation_weight(lnmt_embedded_type)
            read_content_type = self.read_type(correlation_weight_type)

            batch_sliced_pred_type = self.sigmoid(
                d_t * self.linear_out_type_q(read_content_type) + (1 - d_t) * self.linear_out_type_l(read_content_type))
            batch_pred_type.append(batch_sliced_pred_type)

        batch_pred = torch.cat(batch_pred, dim=-1)
        batch_pred_type = torch.cat(batch_pred_type, dim=-1)
        return batch_pred, batch_pred_type

    def compute_correlation_weight(self, query_embedded):
        """
        use dot product to find the similarity between question embedding and each concept
        embedding stored as key_matrix
        where key-matrix could be understood as all concept embedding covered by the course.
        query_embeded : (batch_size, concept_embedding_dim)
        key_matrix : (num_concepts, concept_embedding_dim)
        output: is the correlation distribution between question and all concepts
        """

        similarity = query_embedded @ self.key_matrix.t()
        correlation_weight = torch.softmax(similarity, dim=1)
        return correlation_weight

    def read(self, correlation_weight, d_t):
        """
        read function is to read a student's knowledge level on part of concepts covered by a
        target question.
        we could view value_matrix as the latent representation of a student's knowledge
        in terms of all possible concepts.

        value_matrix: (batch_size, num_concepts, concept_embedding_dim)
        correlation_weight: (batch_size, num_concepts)
        """
        batch_size = self.value_matrix.size(0)
        value_matrix_reshaped = torch.transpose(
            d_t * torch.transpose(self.T_QQ(self.value_matrix), 0, 1) + (1 - d_t) * torch.transpose(self.T_LQ(
                self.value_matrix), 0, 1), 0, 1)
        value_matrix_reshaped = value_matrix_reshaped.reshape(
            batch_size * self.num_concepts, self.value_dim
        )
        correlation_weight_reshaped = correlation_weight.reshape(batch_size * self.num_concepts, 1)
        # a (10,3) * b (10,1) = c (10, 3)is every row vector of a multiplies the row scalar of b
        # the multiplication below is to scale the memory embedding by the correlation weight
        rc = value_matrix_reshaped * correlation_weight_reshaped
        read_content = rc.reshape(batch_size, self.num_concepts, self.value_dim)
        read_content = torch.sum(read_content, dim=1)  # sum over all concepts

        return read_content

    def write(self, correlation_weight, qa_embed, l_embed, d_t, d_t_1):
        """
                write function is to update memory based on the interaction
                value_matrix: (batch_size, memory_size, memory_state_dim)
                correlation_weight: (batch_size, memory_size)
                qa_embedded: (batch_size, memory_state_dim)
                """
        batch_size = self.value_matrix.size(0)
        erase_vector = self.sigmoid(
            (1 - d_t) * self.erase_E_Q(qa_embed) + d_t * self.erase_E_L(l_embed))  # (batch_size, value_dim)

        add_vector = self.tanh(
            (1 - d_t) * self.add_D_Q(qa_embed) + d_t * self.add_D_L(l_embed))  # (batch_size, value_dim)

        erase_reshaped = erase_vector.reshape(batch_size, 1, self.value_dim)
        cw_reshaped = correlation_weight.reshape(batch_size, self.num_concepts,
                                                 1)  # the multiplication is to generate weighted erase vector for each memory cell, therefore, the size is (batch_size, num_concepts, value_dim)
        erase_mul = erase_reshaped * cw_reshaped
        # memory_after_erase = self.value_matrix * (1 - erase_mul)

        memory_after_erase = torch.transpose(
            (((1 - d_t) * (1 - d_t_1)) * torch.transpose(self.T_QQ(self.value_matrix), 0, 1)) + (
                        d_t * d_t_1) * torch.transpose(self.T_LL(
                self.value_matrix), 0, 1) + ((1 - d_t_1) * d_t) * torch.transpose(self.T_QL(self.value_matrix), 0,
                                                                                  1) + (
                        d_t_1 * (1 - d_t)) * torch.transpose(self.T_LQ(
                self.value_matrix), 0, 1), 0, 1) * (1 - erase_mul)

        add_reshaped = add_vector.reshape(batch_size, 1,
                                          self.value_dim)  # the multiplication is to generate weighted add vector for each memory cell therefore, the size is (batch_size, num_concepts, value_dim)
        add_memory = add_reshaped * cw_reshaped
        updated_memory = memory_after_erase + add_memory

        return updated_memory

    def read_type(self, correlation_weight):
        """
        read function is to read a student's knowledge level on part of concepts covered by a
        target question.
        we could view value_matrix as the latent representation of a student's knowledge
        in terms of all possible concepts.
        value_matrix: (batch_size, num_concepts, concept_embedding_dim)
        correlation_weight: (batch_size, num_concepts)
        """
        batch_size = self.value_matrix.size(0)
        value_matrix_reshaped = self.value_matrix.reshape(
            batch_size * self.num_concepts, self.value_dim
        )
        correlation_weight_reshaped = correlation_weight.reshape(batch_size * self.num_concepts, 1)
        # a (10,3) * b (10,1) = c (10, 3)is every row vector of a multiplies the row scalar of b
        # the multiplication below is to scale the memory embedding by the correlation weight
        rc = value_matrix_reshaped * correlation_weight_reshaped
        read_content = rc.reshape(batch_size, self.num_concepts, self.value_dim)
        read_content = torch.sum(read_content, dim=1)  # sum over all concepts

        return read_content

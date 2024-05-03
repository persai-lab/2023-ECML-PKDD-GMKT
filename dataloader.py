import numpy as np
import random
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
from torch.utils.data import TensorDataset
import more_itertools as miter

class GMKT_DataLoader:
    def __init__(self, config, data):
        self.data_name = config['data_name']
        self.batch_size = config["batch_size"]
        self.shuffle = config["shuffle"]
        self.collate_fn = default_collate
        self.metric = config["metric"]

        self.num_questions = config.num_items
        self.num_nongradable_items = config.num_nongradable_items

        self.seed = config['seed']

        self.validation_split = config["validation_split"]
        self.mode = config["mode"]

        self.min_seq_len = config["min_seq_len"] if "min_seq_len" in config else None
        self.max_seq_len = config["max_seq_len"] if "max_seq_len" in config else None
        self.stride = config["max_seq_len"] if "max_seq_len" in config else None

        self.init_kwargs = {
            'batch_size': self.batch_size,
            'shuffle': self.shuffle,
            'collate_fn': self.collate_fn,
        }

        self.generate_train_test_data(data)
        self.read_graph_neighbor()

        if self.metric == 'rmse':
            self.train_data = TensorDataset(torch.Tensor(self.train_data_q).long(),
                                            torch.Tensor(self.train_data_a).float(),
                                            torch.Tensor(self.train_data_l).long(),
                                            torch.Tensor(self.train_data_d).long(),
                                            torch.Tensor(self.train_target_answers).float(),
                                            torch.Tensor(self.train_target_masks).bool(),
                                            torch.Tensor(self.train_target_masks_l).bool())

            self.test_data = TensorDataset(torch.Tensor(self.test_data_q).long(), torch.Tensor(self.test_data_a).float(),
                                           torch.Tensor(self.test_data_l).long(), torch.Tensor(self.test_data_d).long(),
                                           torch.Tensor(self.test_target_answers).float(),
                                           torch.Tensor(self.test_target_masks).bool(),
                                           torch.Tensor(self.test_target_masks_l).bool())

        else:
            self.train_data = TensorDataset(torch.Tensor(self.train_data_q).long(),
                                            torch.Tensor(self.train_data_a).long(),
                                            torch.Tensor(self.train_data_l).long(),
                                            torch.Tensor(self.train_data_d).long(),
                                            torch.Tensor(self.train_target_answers).long(),
                                            torch.Tensor(self.train_target_masks).bool(),
                                            torch.Tensor(self.train_target_masks_l).bool())

            self.test_data = TensorDataset(torch.Tensor(self.test_data_q).long(), torch.Tensor(self.test_data_a).long(),
                                            torch.Tensor(self.test_data_l).long(), torch.Tensor(self.test_data_d).long(),
                                           torch.Tensor(self.test_target_answers).long(),
                                            torch.Tensor(self.test_target_masks).bool(),
                                           torch.Tensor(self.test_target_masks_l).bool())

        self.train_loader = DataLoader(self.train_data, batch_size=self.batch_size)

        self.test_loader = DataLoader(self.test_data, batch_size=self.test_data_a.shape[0])


    def generate_train_test_data(self, data):
        q_records = data["traindata"]["q_data"]
        a_records = data["traindata"]["a_data"]
        l_records = data["traindata"]["l_data"]
        d_records = data["traindata"]["d_data"]

        self.train_data_q, self.train_data_a, self.train_data_l, self.train_data_d = self.GMKT_ExtDataset(q_records,
                                                                                                       a_records,
                                                                                                       l_records,
                                                                                                       d_records,
                                                                                                       self.max_seq_len,
                                                                                                       stride=self.stride)

        self.train_target_answers = np.copy(self.train_data_a)
        self.train_target_masks = (self.train_data_q != 0)
        self.train_target_masks_l = (self.train_data_l != 0)

        if self.mode == "train":
            # n_samples = len(self.train_data_q)
            # split the train data into train and val sets based on the self.n_samples

            self.train_data_q, self.test_data_q, self.train_data_a, self.test_data_a, self.train_data_l, \
            self.test_data_l, self.train_data_d, \
            self.test_data_d, self.train_target_answers, self.test_target_answers, \
            self.train_target_masks, self.test_target_masks, self.train_target_masks_l, self.test_target_masks_l = train_test_split(
                self.train_data_q, self.train_data_a, self.train_data_l, self.train_data_d, self.train_target_answers,
                self.train_target_masks, self.train_target_masks_l)


        elif self.mode == 'test':
            q_records = data["testdata"]["q_data"]
            a_records = data["testdata"]["a_data"]
            l_records = data["testdata"]["l_data"]
            d_records = data["testdata"]["d_data"]


            self.test_data_q, self.test_data_a, self.test_data_l, self.test_data_d = self.GMKT_ExtDataset(q_records,
                                                                                                          a_records,
                                                                                                          l_records,
                                                                                                          d_records,
                                                                                                          self.max_seq_len,
                                                                                                          stride=self.stride)

            self.test_target_answers = np.copy(self.test_data_a)
            self.test_target_masks = (self.test_data_q != 0)
            self.test_target_masks_l = (self.test_data_l != 0)



    def read_graph_neighbor(self):
        q_records = self.train_data_q
        l_records = self.train_data_l
        d_records = self.train_data_d


        q_q_neighbors = [set() for i in range(self.num_questions+1)]
        q_l_neighbors = [set() for i in range(self.num_questions+1)]
        l_q_neighbors = [set() for i in range(self.num_nongradable_items+1)]
        l_l_neighbors = [set() for i in range(self.num_nongradable_items+1)]
        for index in range(len(q_records)):
            q_list = q_records[index]
            l_list = l_records[index]
            d_list = d_records[index]

            for att in range(3, len(d_list)-1):
                if not ((q_list[att] == 0) and (l_list[att] == 0)):
                    if d_list[att] == 0 and d_list[att-1] == 0:
                        q_q_neighbors[q_list[att]].add(q_list[att-1])
                        q_q_neighbors[q_list[att-1]].add(q_list[att])
                    elif d_list[att] == 0 and d_list[att-1] == 1:
                        q_l_neighbors[q_list[att]].add(l_list[att - 1])
                        l_q_neighbors[l_list[att-1]].add(q_list[att])
                    elif d_list[att] == 1 and d_list[att-1] == 0:
                        l_q_neighbors[l_list[att]].add(q_list[att-1])
                        q_l_neighbors[q_list[att-1]].add(l_list[att])
                    elif d_list[att] == 1 and d_list[att-1] == 1:
                        l_l_neighbors[l_list[att]].add(l_list[att - 1])
                        l_l_neighbors[l_list[att - 1]].add(l_list[att])

        for i in range(self.num_questions):
            if i in q_q_neighbors[i]:
                q_q_neighbors[i].remove(i)

        for i in range(self.num_nongradable_items):
            if i in l_l_neighbors[i]:
                l_l_neighbors[i].remove(i)

        q_q_neighbors = [list(i) for i in q_q_neighbors]
        q_l_neighbors = [list(i) for i in q_l_neighbors]
        l_q_neighbors = [list(i) for i in l_q_neighbors]
        l_l_neighbors = [list(i) for i in l_l_neighbors]

        max_q_q_neighbors = max([len(i) for i in q_q_neighbors])
        max_q_l_neighbors = max([len(i) for i in q_l_neighbors])
        max_l_q_neighbors = max([len(i) for i in l_q_neighbors])
        max_l_l_neighbors = max([len(i) for i in l_l_neighbors])

        max_q = max(max_q_q_neighbors, max_l_q_neighbors)
        max_l = max(max_l_l_neighbors, max_q_l_neighbors)

        self.q_q_neighbors = torch.Tensor(
            np.array([list(miter.padded(i, fillvalue=0, n=max_q, next_multiple=False)) for i in q_q_neighbors])).long()
        self.q_l_neighbors = torch.Tensor(
            np.array([list(miter.padded(i, fillvalue=0, n=max_l, next_multiple=False)) for i in q_l_neighbors])).long()
        self.l_q_neighbors = torch.Tensor(
            np.array([list(miter.padded(i, fillvalue=0, n=max_q, next_multiple=False)) for i in l_q_neighbors])).long()
        self.l_l_neighbors = torch.Tensor(
            np.array([list(miter.padded(i, fillvalue=0, n=max_l, next_multiple=False)) for i in l_l_neighbors])).long()


    def generate_graph_data(self, q_q_neighbors, q_l_neighbors, l_q_neighbors, l_l_neighbors, max_q_q_neighbors,
                            max_q_l_neighbors, max_l_q_neighbors, max_l_l_neighbors):

        max_q = max(max_q_q_neighbors, max_l_q_neighbors)
        max_l = max(max_l_l_neighbors, max_q_l_neighbors)

        q_q_neighbors = [list(miter.padded(i, fillvalue=0, n=max_q, next_multiple=False)) for i in q_q_neighbors]
        q_l_neighbors = [list(miter.padded(i, fillvalue=0, n=max_l, next_multiple=False)) for i in q_l_neighbors]
        l_q_neighbors = [list(miter.padded(i, fillvalue=0, n=max_q, next_multiple=False)) for i in l_q_neighbors]
        l_l_neighbors = [list(miter.padded(i, fillvalue=0, n=max_l, next_multiple=False)) for i in l_l_neighbors]

        q_records_train = self.train_data_q
        l_records_train = self.train_data_l
        d_records_train = self.train_data_d

        neighbors_q_train_data = []
        neighbors_l_train_data = []
        for index in range(len(q_records_train)):
            q_list = q_records_train[index]
            l_list = l_records_train[index]
            d_list = d_records_train[index]

            neigbors_q_list = []
            neigbors_l_list = []
            for att in range(len(d_list)):
                if d_list[att] == 0:
                    q_neighbor = q_q_neighbors[q_list[att]]
                    l_neighbor = q_l_neighbors[q_list[att]]

                if d_list[att] == 1:
                    q_neighbor = l_q_neighbors[l_list[att]]
                    l_neighbor = l_l_neighbors[l_list[att]]
                neigbors_q_list.append(q_neighbor)
                neigbors_l_list.append(l_neighbor)
            neighbors_q_train_data.append(neigbors_q_list)
            neighbors_l_train_data.append(neigbors_l_list)


        q_records_test = self.test_data_q
        l_records_test = self.test_data_l
        d_records_test = self.test_data_d

        neighbors_q_test_data = []
        neighbors_l_test_data = []
        for index in range(len(q_records_test)):
            q_list = q_records_test[index]
            l_list = l_records_test[index]
            d_list = d_records_test[index]

            neigbors_q_list = []
            neigbors_l_list = []
            for att in range(len(d_list)):
                if d_list[att] == 0:
                    q_neighbor = q_q_neighbors[q_list[att]]
                    l_neighbor = q_l_neighbors[q_list[att]]

                if d_list[att] == 1:
                    q_neighbor = l_q_neighbors[l_list[att]]
                    l_neighbor = l_l_neighbors[l_list[att]]
                neigbors_q_list.append(q_neighbor)
                neigbors_l_list.append(l_neighbor)
            neighbors_q_test_data.append(neigbors_q_list)
            neighbors_l_test_data.append(neigbors_l_list)

        return np.array(neighbors_q_train_data), np.array(neighbors_l_train_data),\
               np.array(neighbors_q_test_data), np.array(neighbors_l_test_data)


    def GMKT_ExtDataset(self, q_records, a_records, l_records, d_records,
                                           max_seq_len,
                                           stride):


        q_data = []
        a_data = []
        l_data = []
        d_data = []
        for index in range(len(q_records)):
            q_list = q_records[index]
            a_list = a_records[index]
            l_list = l_records[index]
            d_list = d_records[index]
            q_list.insert(0, 0)
            a_list.insert(0, 2)
            l_list.insert(0, 0)
            d_list.insert(0, 0)

            q_list.insert(0, 0)
            a_list.insert(0, 2)
            l_list.insert(0, 0)
            d_list.insert(0, 0)
            q_patches = list(miter.windowed(q_list, max_seq_len, fillvalue=0, step=stride-2))
            a_patches = list(miter.windowed(a_list, max_seq_len, fillvalue=2, step=stride-2))
            l_patches = list(miter.windowed(l_list, max_seq_len, fillvalue=0, step=stride-2))
            d_patches = list(miter.windowed(d_list, max_seq_len, fillvalue=0, step=stride-2))

            q_data.extend(q_patches)
            a_data.extend(a_patches)
            l_data.extend(l_patches)
            d_data.extend(d_patches)

        return np.array(q_data), np.array(a_data), np.array(l_data), np.array(d_data)




import logging
import pickle

import numpy as np
from sklearn import metrics
from torch.backends import cudnn
import torch
from torch import nn
import warnings
from model.GMKT import GMKT
from dataloader import GMKT_DataLoader

warnings.filterwarnings("ignore")
cudnn.benchmark = True


class GMKT_trainer(object):
    def __init__(self, config, data):
        super(GMKT_trainer, self).__init__()
        self.config = config
        self.logger = logging.getLogger("trainer")
        self.metric = config.metric

        self.mode = config.mode
        self.manual_seed = config.seed
        self.device = torch.device("cpu")

        self.current_epoch = 1
        self.current_iteration = 1

        if self.metric == "rmse":
            self.best_val_perf = 1.
        elif self.metric == "auc":
            self.best_val_perf = 0.
        self.best_val_perf_type = 0.
        self.train_loss_list = []
        self.train_loss_type_list = []
        self.test_loss_list = []
        self.test_loss_type_list = []
        self.test_roc_auc_list = []
        self.test_rmse_list = []

        self.data_loader = GMKT_DataLoader(config, data)
        self.model = GMKT(config, self.data_loader.q_q_neighbors, self.data_loader.q_l_neighbors,
                                        self.data_loader.l_q_neighbors, self.data_loader.l_l_neighbors)

        # self.criterion = nn.MSELoss(reduction='sum')
        self.criterion = nn.BCELoss(reduction='sum')
        if config.optimizer == "sgd":
            self.optimizer = torch.optim.SGD(self.model.parameters(),
                                             lr=self.config.learning_rate,
                                             momentum=self.config.momentum,
                                             weight_decay=self.config.weight_decay)
        elif config.optimizer == "adam":
            self.optimizer = torch.optim.Adam(self.model.parameters(),
                                              lr=self.config.learning_rate,
                                              betas=(config.beta1, config.beta2),
                                              eps=self.config.epsilon,
                                              weight_decay=self.config.weight_decay)

        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            patience=0,
            min_lr=1e-5,
            factor=0.5,
            verbose=True
        )

        self.is_cuda = torch.cuda.is_available()
        if self.is_cuda and not self.config.cuda:
            self.logger.info("WARNING: You have a CUDA device, so you should probably enable CUDA")

        self.cuda = self.is_cuda & self.config.cuda

        if self.cuda:
            torch.cuda.manual_seed(self.manual_seed)
            self.device = torch.device("cuda")
            # torch.cuda.set_device(self.config.gpu_device)
            self.model = self.model.to(self.device)
            self.criterion = self.criterion.to(self.device)

            self.logger.info("Program will run on *****GPU-CUDA***** ")
            print("Program will run on *****GPU-CUDA***** ")
        else:
            self.device = torch.device("cpu")
            torch.manual_seed(self.manual_seed)
            self.logger.info("Program will run on *****CPU*****\n")
            print("Program will run on *****CPU*****\n")

    def train(self):

        for epoch in range(1, self.config.max_epoch + 1):
            print("=" * 50 + "Epoch {}".format(epoch) + "=" * 50)
            self.train_one_epoch()
            self.validate()
            self.current_epoch += 1

    def train_one_epoch(self):
        self.model.train()
        self.logger.info("\n")
        self.logger.info("Train Epoch: {}".format(self.current_epoch))
        self.logger.info("learning rate: {}".format(self.optimizer.param_groups[0]['lr']))
        self.train_loss = 0
        self.train_loss_type = 0
        train_elements = 0
        train_elements_type = 0

        for batch_idx, data in enumerate(self.data_loader.train_loader):
            q_list, a_list, l_list, d_list, target_answers_list, target_masks_list, target_masks_l_list = data
            q_list = q_list.to(self.device)
            a_list = a_list.to(self.device)
            l_list = l_list.to(self.device)
            d_list = d_list.to(self.device)
            target_answers_list = target_answers_list.to(self.device)
            target_masks_list = target_masks_list.to(self.device)

            self.optimizer.zero_grad()
            output, output_type = self.model(q_list, a_list, l_list, d_list)

            label = torch.masked_select(target_answers_list[:, 2:], target_masks_list[:, 2:])
            label_type = torch.masked_select(d_list[:, 2:], (target_masks_list + target_masks_l_list)[:, 2:])

            output = torch.masked_select(output, target_masks_list[:, 2:])
            loss_q = self.criterion(output.float(), label.float())

            output_type = torch.masked_select(output_type, (target_masks_list + target_masks_l_list)[:, 2:])
            loss_type = self.criterion(output_type.float(), label_type.float())

            loss = loss_q + self.config.weight_type * loss_type

            self.train_loss += loss_q.item()
            train_elements += target_masks_list[:, 2:].int().sum()
            self.train_loss_type += loss_type.item()
            train_elements_type += (target_masks_list + target_masks_l_list)[:, 2:].int().sum()
            loss.backward(retain_graph=True)  # compute the gradient

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
            self.optimizer.step()  # update the model
            # self.scheduler.step()  # for CycleLR Scheduler or MultiStepLR
            self.current_iteration += 1

        self.train_loss = self.train_loss / train_elements
        self.train_loss_type = self.train_loss_type / train_elements_type
        self.train_loss_all = self.train_loss + self.config.weight_type * self.train_loss_type
        self.scheduler.step( self.train_loss_all)
        self.train_loss_list.append(self.train_loss.data.cpu())
        self.train_loss_type_list.append((self.config.weight_type * self.train_loss_type).data.cpu())
        self.logger.info("Train Loss: {:.6f}, Train Loss type: {:.6f}".format(self.train_loss, self.train_loss_type))
        print("Train Loss: {:.6f}, Train Loss type: {:.6f}".format(self.train_loss, self.train_loss_type))

    def validate(self):
        """
        One cycle of model validation
        :return:
        """
        self.model.eval()
        if self.mode == "train":
            self.logger.info("Validation Result at Epoch: {}".format(self.current_epoch))
            # print("Validation Result at Epoch: {}".format(self.current_epoch))
        else:
            self.logger.info("Test Result at Epoch: {}".format(self.current_epoch))
            # print("Test Result at Epoch: {}".format(self.current_epoch))
        test_loss = 0
        pred_labels = []
        true_labels = []
        pred_labels_type = []
        true_labels_type = []
        with torch.no_grad():
            for data in self.data_loader.test_loader:
                q_list, a_list, l_list, d_list, target_answers_list, target_masks_list, target_masks_list_l = data
                q_list = q_list.to(self.device)
                a_list = a_list.to(self.device)
                l_list = l_list.to(self.device)
                d_list = d_list.to(self.device)
                target_answers_list = target_answers_list.to(self.device)
                target_masks_list = target_masks_list.to(self.device)
                target_masks_l_list = target_masks_list_l.to(self.device)

                output, output_type = self.model(q_list, a_list, l_list, d_list)

                self.test_output_save = output
                self.test_label_save = target_answers_list
                self.test_mask_save = target_masks_list

                label = torch.masked_select(target_answers_list[:, 2:], target_masks_list[:, 2:])
                label_type = torch.masked_select(d_list[:, 2:], (target_masks_list + target_masks_l_list)[:, 2:])

                output = torch.masked_select(output, target_masks_list[:, 2:])
                test_loss += self.criterion(output.float(), label.float()).item()

                output_type = torch.masked_select(output_type, (target_masks_list + target_masks_l_list)[:, 2:])
                test_loss_type = self.criterion(output_type.float(), label_type.float())

                pred_labels.extend(output.tolist())
                true_labels.extend(label.tolist())
                test_elements = target_masks_list[:, 2:].int().sum()
                self.test_loss = test_loss / test_elements

                pred_labels_type.extend(output_type.tolist())
                true_labels_type.extend(label_type.tolist())
                test_elements_type = (target_masks_list + target_masks_l_list)[:, 2:].int().sum()
                self.test_loss_type = test_loss_type / test_elements_type
                self.test_loss_all = self.test_loss + self.config.weight_type * self.test_loss_type

                print("Test Loss: {:.6f}, Test Type Loss: {:.6f}".format(self.test_loss, self.test_loss_type))
                self.test_loss_list.append(self.test_loss.data.cpu())
                self.test_loss_type_list.append((self.config.weight_type * self.test_loss_type).data.cpu())
        self.track_best(true_labels, pred_labels)

    def track_best(self, true_labels, pred_labels):
        self.pred_labels = np.array(pred_labels).squeeze()
        self.true_labels = np.array(true_labels).squeeze()
        self.logger.info(
            "pred size: {} true size {}".format(self.pred_labels.shape, self.true_labels.shape))
        if self.metric == "rmse":
            perf = np.sqrt(metrics.mean_squared_error(self.true_labels, self.pred_labels))
            self.logger.info('RMSE: {:.05}'.format(perf))
            print('RMSE: {:.05}'.format(perf))
            if perf < self.best_val_perf:
                self.best_val_perf = perf
                self.best_train_loss = self.train_loss.item()
                self.best_test_loss = self.test_loss.item()
                self.best_epoch = self.current_epoch
                torch.save(self.model.state_dict(),
                           'saved_model/{}/{}/sl_{}_eq_{}_ea_{}_el_{}_nc_{}_kd_{}_vd_{}_sd_{}_wt_{}_lq_{}_ll_{}_wd_{}_fold_{}.pkl'.format(
                               self.config.data_name,
                               self.config.model_name,
                               self.config.max_seq_len,
                               self.config.embedding_size_q,
                               self.config.embedding_size_a,
                               self.config.embedding_size_l,
                               self.config.num_concepts,
                               self.config.key_dim,
                               self.config.value_dim,
                               self.config.summary_dim,
                               self.config.weight_type,
                               self.config.lambda_q,
                               self.config.lambda_l,
                               self.config.weight_decay,
                               self.config.fold))

            self.test_roc_auc_list.append(perf)
        elif self.metric == "auc":
            perf = metrics.roc_auc_score(self.true_labels, self.pred_labels)
            prec, rec, _ = metrics.precision_recall_curve(self.true_labels, self.pred_labels)
            self.logger.info('ROC-AUC: {:.05}'.format(perf))
            print('ROC-AUC: {:.05}'.format(perf))
            if perf > self.best_val_perf:
                self.best_val_perf = perf
                self.best_train_loss = self.train_loss.item()
                self.best_test_loss = self.test_loss.item()
                self.best_epoch = self.current_epoch
                torch.save(self.model.state_dict(),
                           'saved_model/{}/{}/sl_{}_eq_{}_ea_{}_el_{}_nc_{}_kd_{}_vd_{}_sd_{}_wt_{}_lq_{}_ll_{}_wd_{}_fold_{}.pkl'.format(
                               self.config.data_name,
                               self.config.model_name,
                               self.config.max_seq_len,
                               self.config.embedding_size_q,
                               self.config.embedding_size_a,
                               self.config.embedding_size_l,
                               self.config.num_concepts,
                               self.config.key_dim,
                               self.config.value_dim,
                               self.config.summary_dim,
                               self.config.weight_type,
                               self.config.lambda_q,
                               self.config.lambda_l,
                               self.config.weight_decay,
                               self.config.fold))
            self.test_roc_auc_list.append(perf)
        else:
            raise AttributeError

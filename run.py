import json
from easydict import EasyDict
import pickle
from trainer import GMKT_trainer
import matplotlib.pyplot as plt


def single_exp(config):
    config = EasyDict(config)
    print(config)

    data = pickle.load(open('data/{}/train_val_test_{}.pkl'.format(config.data_name, config.fold), 'rb'))

    config.num_items = data['num_items_Q']
    config.num_nongradable_items = data['num_items_L']
    config.num_users = data['num_users']

    exp_trainner = GMKT_trainer(config, data)
    exp_trainner.train()

    print("best ROC AUC: {}".format(exp_trainner.best_val_perf))

    plt.figure()
    plt.plot(exp_trainner.train_loss_list, label='train')
    plt.plot(exp_trainner.train_loss_type_list, label='train')
    plt.plot(exp_trainner.test_loss_list, label='test_type')
    plt.plot(exp_trainner.test_loss_type_list, label='test_type')
    plt.legend()
    plt.title('losses')
    plt.show()

    plt.figure()
    plt.plot(exp_trainner.test_roc_auc_list, label = 'test roc auc')
    plt.legend()
    plt.title('evaluation')
    plt.show()

    config["best_epoch"] = exp_trainner.best_epoch
    config["best_train_loss"] = exp_trainner.best_train_loss
    config["best_test_loss"] = exp_trainner.best_test_loss
    config["best_val_perf"] = exp_trainner.best_val_perf

    print(config)


def ednet():
    config = {
                "data_name": 'ednet',
                "model_name": "GMKT",

                "mode": 'test',
                "fold": 1,
                "metric": 'auc',
                "shuffle": True,

                "cuda": True,
                "gpu_device": 0,
                "seed": 1024,

                "min_seq_len": 2,
                "max_seq_len": 100,  # the max step of RNN model
                "batch_size": 32,
                "learning_rate": 0.01,
                "max_epoch": 70,
                "validation_split": 0.2,

                "embedding_size_q": 32,
                "embedding_size_a": 32,
                "embedding_size_l": 32,
                "num_concepts": 8,
                "key_dim": 32,
                "value_dim": 32,
                "summary_dim": 32,
                'weight_type': 0.1,
                "lambda_q": 0.5,
                "lambda_l": 0.5,

                "init_std": 0.2,
                "max_grad_norm": 10,

                "optimizer": 'adam',
                "epsilon": 0.1,
                "beta1": 0.9,
                "beta2": 0.999,
                "weight_decay": 0.05,
            }
    single_exp(config)

def Junyi2063():
    config = {
                "data_name": 'Junyi2063',
                "model_name": 'GMKT',

                "mode": 'test',
                "fold": 2,
                "metric": 'auc',
                "shuffle": True,

                "cuda": True,
                "gpu_device": 0,
                "seed": 1024,

                "min_seq_len": 2,
                "max_seq_len": 100,  # the max seq len of model
                "batch_size": 32,
                "learning_rate": 0.01,
                "max_epoch": 60,
                "validation_split": 0.2,

                "embedding_size_q": 32,
                "embedding_size_a": 32,
                "embedding_size_l": 32,
                "num_concepts": 32,
                "key_dim": 64,
                "value_dim": 64,
                "summary_dim": 32,
                'weight_type': 0.01,
                "lambda_q": 0.5,
                "lambda_l": 0.5,

                "init_std": 0.2,
                "max_grad_norm": 50,

                "optimizer": 'adam',
                "epsilon": 0.1,
                "beta1": 0.9,
                "beta2": 0.999,
                "weight_decay": 0.05,
            }
    single_exp(config)


def morf():
    config = {
        "data_name": 'MORF686',
        "model_name": 'GMKT',

        "mode": 'test',
        "fold": 1,
        "metric": 'rmse',
        "shuffle": True,

        "cuda": True,
        "gpu_device": 0,
        "seed": 1024,

        "min_seq_len": 2,
        "max_seq_len": 100,  # the max step of RNN model
        "batch_size": 32,
        "learning_rate": 0.01,
        "max_epoch": 200,
        "validation_split": 0.2,

        "embedding_size_q": 16,
        "embedding_size_a": 16,
        "embedding_size_l": 32,
        "num_concepts": 8,
        "key_dim": 16,
        "value_dim": 16,
        "summary_dim": 16,
        'weight_type': 0.1,
        "lambda_q": 0.5,
        "lambda_l": 0.5,

        "init_std": 0.2,
        "max_grad_norm": 5,

        "optimizer": 'adam',
        "epsilon": 0.1,
        # "epsilon": 1e-8,
        "beta1": 0.9,
        "beta2": 0.999,
        "weight_decay": 0.01,
    }
    single_exp(config)



if __name__== '__main__':
    # ednet()
    Junyi2063()
    # morf()
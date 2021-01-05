# Import packages
import copy
import os
import sys
import numpy as np
from torch.nn import functional as F
from torch.utils.data import DataLoader
import argparse
import torch
import time
from tqdm import tqdm

# Import utility scripts
from process_data import DataPrep
from log_outputs import Logger
from MetaRec import user_preference_estimator
import utils as utils

sys.path.append('..')


def parse_args():
    """
    Parse argument parameters
    """
    parser = argparse.ArgumentParser([], description='Fast Context Adaptation via Meta-Learning (CAVIA),'
                                                     'Clasification experiments.')

    parser.add_argument('--seed', type=int, default=53)
    parser.add_argument('--task', type=str, default='multi', help='problem setting: sine or celeba')
    parser.add_argument('--tasks_per_metaupdate', type=int, default=32,
                        help='number of tasks in each batch per meta-update')

    parser.add_argument('--lr_inner', type=float, default=0.01, help='inner-loop learning rate (per task)')
    parser.add_argument('--lr_meta', type=float, default=1e-3,
                        help='outer-loop learning rate (used with Adam optimiser)')
    parser.add_argument('--lr_meta_decay', type=float, default=0.9, help='decay factor for meta learning rate')

    parser.add_argument('--num_grad_steps_inner', type=int, default=5,
                        help='number of gradient steps in inner loop (during training)')
    parser.add_argument('--num_grad_steps_eval', type=int, default=1,
                        help='number of gradient updates at test time (for evaluation)')

    parser.add_argument('--first_order', action='store_true', default=False,
                        help='run first order approximation of CAVIA')

    parser.add_argument('--data_root', type=str, default="../../ml-1m", help='path to data root')
    parser.add_argument('--num_workers', type=int, default=4, help='num of workers to use')
    parser.add_argument('--test', action='store_true', default=False, help='num of workers to use')

    parser.add_argument('--embedding_dim', type=int, default=32, help='num of workers to use')
    parser.add_argument('--first_fc_hidden_dim', type=int, default=64, help='num of workers to use')
    parser.add_argument('--second_fc_hidden_dim', type=int, default=64, help='num of workers to use')
    parser.add_argument('--num_epoch', type=int, default=30, help='num of workers to use')
    parser.add_argument('--num_genre', type=int, default=25, help='num of workers to use')
    parser.add_argument('--num_director', type=int, default=2186, help='num of workers to use')
    parser.add_argument('--num_actor', type=int, default=8030, help='num of workers to use')
    parser.add_argument('--num_rate', type=int, default=6, help='num of workers to use')
    parser.add_argument('--num_gender', type=int, default=2, help='num of workers to use')
    parser.add_argument('--num_age', type=int, default=7, help='num of workers to use')
    parser.add_argument('--num_occupation', type=int, default=21, help='num of workers to use')
    parser.add_argument('--num_zipcode', type=int, default=3402, help='num of workers to use')

    parser.add_argument('--rerun', action='store_true', default=False,
                        help='Re-run experiment (will override previously saved results)')

    args = parser.parse_args()
    return args


def run(args, num_workers=1, log_interval=100, verbose=True, save_path=None):
    """
    Run the model
    :param args: set of arguments
    :param num_workers: number of workers
    :param log_interval: logging interval
    """
    # Current root path
    code_root = os.path.dirname(os.path.realpath(__file__))
    if not os.path.isdir('{}/{}_result_files/'.format(code_root, args.task)):
        os.mkdir('{}/{}_result_files/'.format(code_root, args.task))

    # Result path
    path = '{}/{}_result_files/'.format(code_root, args.task) + utils.get_path_from_args(args)
    print('File saved in {}'.format(path))

    if os.path.exists(path + '.pkl') and not args.rerun:
        print('File has already existed. Try --rerun')
        return utils.load_obj(path)

    start_time = time.time()
    utils.set_seed(args.seed)

    # ---------------------------------------------------------
    # -------------------- Training ---------------------------

    # Initialize the model
    model = user_preference_estimator(args)
    # Train the model
    model.train()
    print(sum([param.nelement() for param in model.parameters()]))

    # Set up the meta-optimiser for model parameters
    meta_optimiser = torch.optim.Adam(model.parameters(), args.lr_meta)
    # Set up a scheduler for the meta-optimizer
    scheduler = torch.optim.lr_scheduler.StepLR(meta_optimiser, 5000, args.lr_meta_decay)

    # Initialize logger
    logger = Logger()
    logger.args = args

    # Initialize the starting point for the meta gradient (it's faster to copy this than to create new object)
    meta_grad_init = [0 for _ in range(len(model.state_dict()))]

    # Create a PyTorch data loader
    dataloader_train = DataLoader(DataPrep(args), batch_size=1, num_workers=args.num_workers)

    # Loop through all epochs
    for epoch in range(args.num_epoch):

        x_spt, y_spt, x_qry, y_qry = [], [], [], []
        iter_counter = 0
        for step, batch in enumerate(dataloader_train):
            if len(x_spt) < args.tasks_per_metaupdate:
                x_spt.append(batch[0][0])
                y_spt.append(batch[1][0])
                x_qry.append(batch[2][0])
                y_qry.append(batch[3][0])
                if not len(x_spt) == args.tasks_per_metaupdate:
                    continue

            if len(x_spt) != args.tasks_per_metaupdate:
                continue

            # Initialize the meta-gradient
            meta_grad = copy.deepcopy(meta_grad_init)
            loss_pre = []
            loss_after = []
            for i in range(args.tasks_per_metaupdate):
                loss_pre.append(F.mse_loss(model(x_qry[i]), y_qry[i]).item())
                fast_parameters = model.final_part.parameters()
                for weight in model.final_part.parameters():
                    weight.fast = None
                for k in range(args.num_grad_steps_inner):
                    logits = model(x_spt[i])
                    loss = F.mse_loss(logits, y_spt[i])
                    grad = torch.autograd.grad(loss, fast_parameters, create_graph=True)
                    fast_parameters = []
                    for k, weight in enumerate(model.final_part.parameters()):
                        if weight.fast is None:
                            weight.fast = weight - args.lr_inner * grad[k]  # create weight.fast
                        else:
                            weight.fast = weight.fast - args.lr_inner * grad[k]
                        fast_parameters.append(weight.fast)

                logits_q = model(x_qry[i])
                # loss_q will be overwritten and just keep the loss_q on last update step.
                loss_q = F.mse_loss(logits_q, y_qry[i])
                loss_after.append(loss_q.item())
                task_grad_test = torch.autograd.grad(loss_q, model.parameters())

                for g in range(len(task_grad_test)):
                    meta_grad[g] += task_grad_test[g].detach()

            # -------------- Meta Update --------------
            # Zero the gradients of meta-optimiser
            meta_optimiser.zero_grad()

            # Set gradients of model parameters manually
            for c, param in enumerate(model.parameters()):
                param.grad = meta_grad[c] / float(args.tasks_per_metaupdate)
                param.grad.data.clamp_(-10, 10)

            # The meta-optimiser only operates on the shared parameters, not the context parameters
            meta_optimiser.step()
            scheduler.step()
            x_spt, y_spt, x_qry, y_qry = [], [], [], []

            loss_pre = np.array(loss_pre)
            loss_after = np.array(loss_after)
            logger.train_loss.append(np.mean(loss_pre))
            logger.valid_loss.append(np.mean(loss_after))
            logger.train_conf.append(1.96 * np.std(loss_pre, ddof=0) / np.sqrt(len(loss_pre)))
            logger.valid_conf.append(1.96 * np.std(loss_after, ddof=0) / np.sqrt(len(loss_after)))
            logger.test_loss.append(0)
            logger.test_conf.append(0)

            # Save the logger object
            utils.save_obj(logger, path)
            # Print current results
            logger.print_info(epoch, iter_counter, start_time)
            # Initialize start time again
            start_time = time.time()
            # Increment iteration counter
            iter_counter += 1
        if epoch % 2 == 0:
            print('saving model at iter', epoch)
            logger.valid_model.append(copy.deepcopy(model))

    return logger, model


def evaluate(iter_counter, args, model, logger, dataloader, save_path):
    """
    Evaluate the model
    :param iter_counter: iteration counter
    :param args: set of arguments
    :param model: given model
    :param logger: given logger
    :param dataloader: given DataLoader
    :param save_path: path directory to save results
    """
    logger.prepare_inner_loop(iter_counter, mode='valid')

    for c, batch in enumerate(dataloader):
        x_spt = batch[0]
        y_spt = batch[1]
        x_qry = batch[2]
        y_qry = batch[3]

        for i in range(x_spt.shape[0]):

            # -------------- Inner Update --------------

            logger.log_pre_update(iter_counter, x_spt[i], y_spt[i], x_qry[i], y_qry[i], model, mode='valid')
            fast_parameters = model.parameters()
            for weight in model.parameters():
                weight.fast = None
            for k in range(args.num_grad_steps_eval):
                logits = model(x_spt[i])
                loss = F.cross_entropy(logits, y_spt[i])
                grad = torch.autograd.grad(loss, fast_parameters, create_graph=True)
                fast_parameters = []
                for k, weight in enumerate(model.parameters()):
                    # for usage of weight.fast, please see Linear_fw, Conv_fw in backbone.py
                    if weight.fast is None:
                        weight.fast = weight - args.lr_inner * grad[k]  # create weight.fast
                    else:
                        weight.fast = weight.fast - args.lr_inner * grad[k]
                    fast_parameters.append(weight.fast)

            logger.log_post_update(iter_counter, x_spt[i], y_spt[i],
                                   x_qry[i], y_qry[i], model, mode='valid')

    # This will take the mean over the batches
    logger.summarise_inner_loop(mode='valid')

    # Keep track of best models
    logger.update_best_model(model, save_path)


def evaluate_test(args, model, dataloader):
    """
    Evaluate the model
    :param args: set of arguments
    :param model: given model
    :param dataloader: given DataLoader
    """
    # Evaluate the model
    model.eval()
    loss_all = []
    for c, batch in tqdm(enumerate(dataloader)):
        x_spt = batch[0]
        y_spt = batch[1]
        x_qry = batch[2]
        y_qry = batch[3]
        for i in range(x_spt.shape[0]):
            # -------------- inner update --------------
            fast_parameters = model.final_part.parameters()
            for weight in model.final_part.parameters():
                weight.fast = None
            for k in range(args.num_grad_steps_inner):
                logits = model(x_spt[i])
                loss = F.mse_loss(logits, y_spt[i])
                grad = torch.autograd.grad(loss, fast_parameters, create_graph=True)
                fast_parameters = []
                for k, weight in enumerate(model.final_part.parameters()):
                    if weight.fast is None:
                        weight.fast = weight - args.lr_inner * grad[k]  # create weight.fast
                    else:
                        weight.fast = weight.fast - args.lr_inner * grad[k]
                    fast_parameters.append(weight.fast)
            loss_all.append(F.l1_loss(y_qry[i], model(x_qry[i])).item())
    loss_all = np.array(loss_all)
    print('{}+/-{}'.format(np.mean(loss_all), 1.96 * np.std(loss_all, 0) / np.sqrt(len(loss_all))))


if __name__ == '__main__':
    args = parse_args()
    if not args.test:
        run(args, num_workers=1, log_interval=100, verbose=True, save_path=None)
    else:
        utils.set_seed(args.seed)
        code_root = os.path.dirname(os.path.realpath(__file__))
        # mode_path = utils.get_path_from_args(args)
        mode_path = 'c9603433086fde2b21477898d1fbe5e7'
        path = '{}/{}_result_files/'.format(code_root, args.task) + mode_path
        logger = utils.load_obj(path)
        model = logger.valid_model[-1]
        # old, new_user, new_item, new_item_user
        dataloader_test = DataLoader(DataPrep(args, partition='test', test_way='new_item_user'),
                                     batch_size=1, num_workers=args.num_workers)
        evaluate_test(args, model, dataloader_test)

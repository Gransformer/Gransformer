import create_graphs
import os
import random
from data import MyGraph_sequence_sampler_pytorch, my_decode_adj
from config import Args
# from model import GCADEModel, train
import numpy as np
import torch
from transformer.Models import Transformer
import torch.optim as optim
from transformer.Optim import MyScheduledOptim #, ScheduledOptim
import time
import sys
import utils
import torch.nn.functional as F
from utils import save_graph_list
import pickle
import argparse
from utils import prepare_for_MADE

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

# if __name__ == '__main__':

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# random.seed(123)
# np.random.seed(123)
# torch.manual_seed(123)

args = Args()

graphs = create_graphs.create(args)   ## do not comment this line when use_pre_savede_graphs is True. This line sets args.max_prev_node too.

if args.use_pre_saved_graphs:

    with open(args.graph_save_path + args.fname_test + '0.dat', 'rb') as fin:
        graphs = pickle.load(fin)

    # if use pre-saved graphs
    # dir_input = "/dfs/scratch0/jiaxuany0/graphs/"
    # fname_test = dir_input + args.note + '_' + args.graph_type + '_' + str(args.num_layers) + '_' + str(
    #     args.hidden_size_rnn) + '_test_' + str(0) + '.dat'
    # graphs = load_graph_list(fname_test, is_real=True)
    # graphs_test = graphs[int(0.8 * graphs_len):]
    # graphs_train = graphs[0:int(0.8 * graphs_len)]
    # graphs_validate = graphs[int(0.2 * graphs_len):int(0.4 * graphs_len)]

else:
    random.shuffle(graphs)

graphs_len = len(graphs)
graphs_test = graphs[int((1 - args.test_portion) * graphs_len):]
graphs_train = graphs[0:int(args.training_portion * graphs_len)]
graphs_validate = graphs[int((1 - args.test_portion - args.validation_portion) * graphs_len):
                         int((1 - args.test_portion) * graphs_len)]

if not args.use_pre_saved_graphs:
    # save ground truth graphs
    ## To get train and test set, after loading you need to manually slice
    save_graph_list(graphs, args.graph_save_path + args.fname_train + '0.dat')
    save_graph_list(graphs, args.graph_save_path + args.fname_test + '0.dat')
    print('train and test graphs saved at: ', args.graph_save_path + args.fname_test + '0.dat')

graph_validate_len = 0
for graph in graphs_validate:
    graph_validate_len += graph.number_of_nodes()
graph_validate_len /= len(graphs_validate)
print('graph_validate_len', graph_validate_len)

graph_test_len = 0
for graph in graphs_test:
    graph_test_len += graph.number_of_nodes()
graph_test_len /= len(graphs_test)
print('graph_test_len', graph_test_len)

args.max_num_node = max([graphs[i].number_of_nodes() for i in range(len(graphs))])
min_num_node = min([graphs[i].number_of_nodes() for i in range(len(graphs))])
max_num_edge = max([graphs[i].number_of_edges() for i in range(len(graphs))])
min_num_edge = min([graphs[i].number_of_edges() for i in range(len(graphs))])

# show graphs statistics
print('total graph num: {}, training set: {}'.format(len(graphs), len(graphs_train)))
print('max number node: {}'.format(args.max_num_node))
print('max/min number edge: {}; {}'.format(max_num_edge, min_num_edge))
print('max previous node: {}'.format(args.max_prev_node))



### comment when normal training, for graph completion only
# p = 0.5
# for graph in graphs_train:
#     for node in list(graph.nodes()):
#         # print('node',node)
#         if np.random.rand()>p:
#             graph.remove_node(node)
# for edge in list(graph.edges()):
#     # print('edge',edge)
#     if np.random.rand()>p:
#         graph.remove_edge(edge[0],edge[1])

### dataset initialization
dataset = MyGraph_sequence_sampler_pytorch(graphs_train, args, max_prev_node=args.max_prev_node,
                                             max_num_node=args.max_num_node)
sample_strategy = torch.utils.data.sampler.WeightedRandomSampler([1.0 / len(dataset) for i in range(len(dataset))],
                                                                 num_samples=args.batch_size * args.batch_ratio,
                                                                 replacement=True)
dataset_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, #num_workers=args.num_workers,
                                             sampler=sample_strategy)

val_dataset = MyGraph_sequence_sampler_pytorch(graphs_validate, args, max_prev_node=args.max_prev_node,
                                             max_num_node=args.max_num_node)
# val_sample_strategy = torch.utils.data.sampler.WeightedRandomSampler([1.0 / len(val_dataset) for i in range(len(val_dataset))],
#                                                                  num_samples=args.batch_size * args.batch_ratio,
#                                                                  replacement=True)
val_dataset_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size // 2, #num_workers=args.num_workers,
                                             sampler=None) #val_sample_strategy)


test_dataset = MyGraph_sequence_sampler_pytorch(graphs_test, args, max_prev_node=args.max_prev_node,
                                             max_num_node=args.max_num_node)
# test_sample_strategy = torch.utils.data.sampler.WeightedRandomSampler([1.0 / len(test_dataset) for i in range(len(test_dataset))],
#                                                                  num_samples=args.batch_size * args.batch_ratio,
#                                                                  replacement=True)
test_dataset_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size // 2,  #num_workers=args.num_workers,
                                             sampler=None) #test_sample_strategy)


if args.input_type == 'node_based':
    args.max_seq_len = dataset.max_seq_len
    args.vocab_size = args.max_num_node + 3  # 0 for padding, self.n+1 for add_node, self.n+2 for termination
elif args.input_type in ['preceding_neighbors_vector', 'max_prev_node_neighbors_vec']:
    args.max_seq_len = dataset.max_seq_len
    args.vocab_size = None
else:
    raise NotImplementedError

print('Preparing dataset finished.')

# gcade_model = GCADEModel(args)

# print('Model initiated.')

# train(gcade_model, dataset_loader, args)

model = Transformer(
    args.vocab_size,
    args.vocab_size,
    src_pad_idx=args.src_pad_idx,
    trg_pad_idx=args.trg_pad_idx,
    args=args,
    trg_emb_prj_weight_sharing=args.proj_share_weight,
    emb_src_trg_weight_sharing=args.embs_share_weight,
    d_k=args.d_k,
    d_v=args.d_v,
    d_model=args.d_model,
    d_word_vec=args.d_word_vec,
    d_inner=args.d_inner_hid,
    n_layers=args.n_layers,
    n_ensemble=args.n_ensemble,
    n_head=args.n_head,
    dropout=args.dropout,
    scale_emb_or_prj=args.scale_emb_or_prj).to(args.device)

print('model initiated.')

# optimizer = ScheduledOptim(
#     optim.Adam(model.parameters(), betas=(0.9, 0.98), eps=1e-09),
#     args.lr_mul, args.d_model, args.n_warmup_steps)
optimizer = MyScheduledOptim(
    optim.Adam(model.parameters(), betas=(0.9, 0.98), eps=1e-09),
    optim.Adam(list(model.parameters())[model.encoder.num_shared_parameters:], betas=(0.9, 0.98), eps=1e-09),
    args.milestones, args.lr_list, args.sep_optimizer_start_step)

if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)



def cal_performance(pred, dec_output, gold, trg_pad_idx, args, model, termination_bit_weight=None, smoothing=False):
    ''' Apply label smoothing if needed '''

    loss = cal_loss(pred, dec_output, gold, trg_pad_idx, args, model, termination_bit_weight, smoothing)
    if args.input_type == 'node_based':
        pred = pred.max(1)[1]
        gold = gold.contiguous().view(-1)
        non_pad_mask = gold.ne(trg_pad_idx)
        n_correct = pred.eq(gold).masked_select(non_pad_mask).sum().item()
        n_word = non_pad_mask.sum().item()

        return loss, n_correct, n_word
    elif args.input_type in ['preceding_neighbors_vector', 'max_prev_node_neighbors_vec']:
        return loss, None
    else:
        raise NotImplementedError


def cal_loss(pred, dec_output, gold, trg_pad_idx, args, model, termination_bit_weight=None, smoothing=False):
    ''' Calculate cross entropy loss, apply label smoothing if needed. '''
    if smoothing:
        if args.input_type == 'node_based':
            gold = gold.contiguous().view(-1)
            eps = 0.1
            n_class = pred.size(1)

            one_hot = torch.zeros_like(pred).scatter(1, gold.view(-1, 1), 1)
            one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
            log_prb = F.log_softmax(pred, dim=1)

            non_pad_mask = gold.ne(trg_pad_idx)
            loss = -(one_hot * log_prb).sum(dim=1)
            loss = loss.masked_select(non_pad_mask).sum()  # average later
        else:
            raise NotImplementedError
    else:
        if args.input_type == 'node_based':
            gold = gold.contiguous().view(-1)
            loss = F.cross_entropy(pred, gold, ignore_index=trg_pad_idx, reduction='sum')
        elif args.input_type == 'preceding_neighbors_vector':

            if args.allow_all_zeros and (args.use_max_prev_node or args.use_bfs_incremental_parent_idx):
                raise NotImplementedError

            pred = torch.sigmoid(pred).view(-1, args.max_seq_len, pred.size(-1))
            # eps = 1e-9
            # print('#################### ', ((pred > eps) & (pred < 1-eps)).sum().item(), (pred <= eps).sum().item(), (pred >= 1-eps).sum().item())

            cond_1 = gold != args.trg_pad_idx
            if args.use_max_prev_node:
                cond_mpn = torch.ones(gold.size(0), gold.size(1), gold.size(2)).to(device=args.device)
                cond_mpn = torch.tril(cond_mpn, diagonal=0)
                cond_mpn = torch.triu(cond_mpn, diagonal=-args.max_prev_node+1)
                cond_mpn[:, :, 0] = 1
                cond_1 = cond_1 * cond_mpn

            if args.use_bfs_incremental_parent_idx:
                gold_0 = gold[:, :, 1:].clone()
                ind_dontcare = gold_0 == args.dontcare_input
                ind_0 = gold_0 == args.zero_input
                ind_1 = gold_0 == args.one_input
                gold_0[ind_dontcare] = 0
                gold_0[ind_0] = 0
                gold_0[ind_1] = 1
                cond_bfs_par = gold_0.cumsum(dim=2) > 0
                cond_bfs_par = torch.cat(
                    [torch.zeros(cond_bfs_par.size(0), 1, cond_bfs_par.size(2)).bool().to(args.device),
                     cond_bfs_par[:, :-1, :]], dim=1)
                cond_bfs_par[:, 1, 0] = True
                cond_bfs_par = torch.cat(
                    [torch.ones(cond_bfs_par.size(0), cond_bfs_par.size(1), 1).bool().to(args.device), cond_bfs_par],
                    dim=2)
                cond_bfs_par = torch.tril(cond_bfs_par, diagonal=0)
                cond_1 = cond_1 * cond_bfs_par

            if (not args.use_termination_bit) or args.feed_graph_length:
                cond_1[:, :, 0] = False

            pred_1 = torch.tril(pred * cond_1, diagonal=0)
            gold_1 = torch.tril(gold * cond_1, diagonal=0)
            ind_0 = gold_1 == args.zero_input
            ind_1 = gold_1 == args.one_input
            gold_1[ind_0] = 0
            gold_1[ind_1] = 1

            assert not (args.weight_positions and (termination_bit_weight is not None))

            if args.weight_positions:
                loss_1 = F.binary_cross_entropy(pred_1, gold_1, reduction='none').sum(-1)
                loss_1 = loss_1 * model.positions_weights.view(1,-1)
                loss_1 = loss_1.sum()
            elif termination_bit_weight is not None:
                loss_1 = F.binary_cross_entropy(pred_1, gold_1, reduction='none')
                loss_1[:,:,0] = loss_1[:,:,0] * termination_bit_weight
                loss_1 = loss_1.sum()
            else:
                # tmp = F.binary_cross_entropy(pred_1, gold_1, reduction='none')
                # termination_loss = tmp[:,:,0].sum().item() / tmp.size(0)
                # edges_loss = tmp[:, :, 1:].sum().item() / tmp.size(0)
                # loss_1 = tmp.sum()
                loss_1 = F.binary_cross_entropy(pred_1, gold_1, reduction='sum')

            if args.allow_all_zeros or not args.use_termination_bit:
                loss = loss_1
            else:
                if args.use_MADE:
                    gold_all_zeros = gold.clone()
                    gold_all_zeros[gold == args.one_input] = args.zero_input
                    if args.separate_termination_bit:
                        gold_all_zeros = gold_all_zeros[:, :, 1:]
                    tmp = model.trg_word_MADE(torch.cat([dec_output, prepare_for_MADE(gold_all_zeros, args)], dim=2))
                    if model.scale_prj:
                        tmp *= model.d_model ** -0.5
                    pred_all_zeros = torch.sigmoid(tmp)
                    if args.separate_termination_bit:
                        pred_all_zeros = torch.cat([pred[:,:,:1], pred_all_zeros], dim=2)
                else:
                    pred_all_zeros = pred

                if args.feed_graph_length:
                    cond_0 = gold[:,:,0] == args.zero_input
                else:
                    cond_0 = gold[:,:,0] != args.trg_pad_idx
                cond_0[:, 0] = False
                cond_2 = cond_0.unsqueeze(-1).repeat(1, 1, gold.size(-1))
                if args.use_max_prev_node:
                    cond_2 = cond_2 * cond_mpn
                if args.use_bfs_incremental_parent_idx:
                    cond_2 = cond_2 * cond_bfs_par
                if args.feed_graph_length:
                    cond_2[:,:,0] = False
                pred_2 = torch.tril(pred_all_zeros * cond_2, diagonal=0)
                gold_2 = torch.zeros(gold.size(0), gold.size(1), gold.size(2), device=gold.device)

                p_zero = torch.exp(-F.binary_cross_entropy(pred_2, gold_2, reduction='none').sum(-1))
                # loss_2 = torch.log(1-p_zero[cond_0]).sum()
                loss_2 = torch.log(torch.max(1-p_zero[cond_0], torch.tensor([1e-9]).to(args.device)))
                if args.weight_positions:
                    loss_2 = loss_2 * model.positions_weights
                loss_2 = loss_2.sum()
                # print('\n', 'termination:', termination_loss, '      edges:', edges_loss, '      loss2:', loss_2.item() / cond_0.size(0))
                loss = loss_1 + loss_2
        elif args.input_type == 'max_prev_node_neighbors_vec':

            pred = torch.sigmoid(pred).view(-1, args.max_seq_len, pred.size(-1))

            if args.allow_all_zeros:
                raise NotImplementedError

            cond_pad = gold != args.trg_pad_idx
            cond_max_prev = torch.ones(pred.size(0), pred.size(1), pred.size(2)).to(args.device)
            cond_max_prev = torch.tril(cond_max_prev, diagonal=-1)
            cond_max_prev = torch.flip( cond_max_prev, [2])
            cond_max_prev[:, :, 0] = 1

            if args.use_bfs_incremental_parent_idx:
                gold_0 = gold[:,:,1:].clone()
                ind_0 = gold_0 == args.zero_input
                ind_1 = gold_0 == args.one_input
                gold_0[ind_0] = 0
                gold_0[ind_1] = 1
                cond_bfs_par = gold_0.cumsum(dim=2) > 0
                cond_bfs_par = torch.cat(
                    [cond_bfs_par, torch.ones(cond_bfs_par.size(0), cond_bfs_par.size(1), 1).bool().to(args.device)],
                    dim=2)
                cond_bfs_par = torch.cat(
                    [torch.zeros(cond_bfs_par.size(0), 1, cond_bfs_par.size(2)).bool().to(args.device),
                     cond_bfs_par[:, :-1, :]], dim=1)
                cond_bfs_par[:, :, 0] = True
                cond_max_prev = cond_bfs_par

            pred_1 = pred * cond_pad * cond_max_prev
            gold_1 = gold * cond_pad * cond_max_prev
            ind_0 = gold_1 == args.zero_input
            ind_1 = gold_1 == args.one_input
            gold_1[ind_0] = 0
            gold_1[ind_1] = 1

            loss_1 = F.binary_cross_entropy(pred_1, gold_1, reduction='sum')

            cond_zeros_1d = gold[:, :, 0] != args.trg_pad_idx
            cond_zeros_1d[:, 0] = False
            cond_zeros_2d = cond_zeros_1d.unsqueeze(-1).repeat(1, 1, gold.size(-1))

            pred_2 = pred * cond_zeros_2d * cond_max_prev
            gold_2 = torch.zeros(gold.size(0), gold.size(1), gold.size(2), device=gold.device)

            p_zero = torch.exp(-F.binary_cross_entropy(pred_2, gold_2, reduction='none').sum(-1))
            loss_2 = torch.log(torch.max(1-p_zero[cond_zeros_1d], torch.tensor([1e-9]).to(args.device))).sum()
            loss = loss_1 + loss_2
        else:
            raise NotImplementedError
    return loss

def generate_graph_exact(gg_model, args):

    gg_model.eval()

    assert args.input_type == 'preceding_neighbors_vector'

    # src_seq = args.src_pad_idx * torch.ones((args.test_batch_size, args.max_seq_len, args.max_num_node + 1),
    #                                        dtype=torch.float32).to(args.device)
    src_seq = args.dontcare_input * torch.ones((args.test_batch_size, args.max_seq_len, args.max_num_node + 1),
                                               dtype=torch.float32).to(args.device)
    src_seq[:, 0, :] = args.src_pad_idx
    for i in range(1,args.max_seq_len):
        src_seq[:, i, :i] = args.zero_input

    gold_seq = args.dontcare_input * torch.ones((args.test_batch_size, args.max_seq_len, args.max_num_node + 1),
                                                dtype=torch.float32).to(args.device)
    gold_seq[:, -1, :] = args.trg_pad_idx
    for i in range(args.max_seq_len - 1):
        gold_seq[:, i, :i+1] = args.zero_input

    adj = torch.zeros((args.test_batch_size, args.max_seq_len, args.max_seq_len), dtype=torch.float32).to(
        args.device)

    if args.estimate_num_nodes:
        len_gen = np.random.choice(np.arange(1,args.max_num_node + 1), args.test_batch_size, True, gg_model.num_nodes_prob[1:])
        if args.feed_graph_length:
            for i in range(src_seq.size(0)):
                src_seq[i, len_gen[i]+1, 0] = args.one_input
    not_finished_idx = torch.ones([src_seq.size(0)]).bool().to(args.device)

    if args.use_bfs_incremental_parent_idx:
        min_par_idx = torch.zeros(src_seq.size(0), src_seq.size(2), dtype=torch.int32).bool().to(args.device)

    if args.use_MADE:
        gg_model.trg_word_MADE.update_masks()

    for i in range(args.max_seq_len - 1):


        tmp, dec_output = gg_model(src_seq[not_finished_idx], src_seq[not_finished_idx], gold_seq[not_finished_idx],
                                   adj[not_finished_idx])
        pred_probs = torch.sigmoid(tmp).view(-1, args.max_seq_len, args.max_num_node + 1)

        if args.use_bfs_incremental_parent_idx:

            tmp_ind = ~min_par_idx[not_finished_idx]
            tmp_ind[:, 0] = False
            tmp_ind[:, i+1:] = False
            zero_logprob = (torch.log(torch.max(1 - pred_probs[:, i, :],
                                                # torch.tensor([0]).to(args.device)
                                                torch.tensor([1e-9]).to(args.device))) * tmp_ind).sum(dim=1)

            '''
            zero_logprob = torch.zeros(not_finished_idx.sum().item()).to(args.device)
            for j in range(not_finished_idx.sum().item()):
                tmp_ind = ~ min_par_idx[not_finished_idx][j]
                tmp_ind[0] = False
                tmp_ind[i+1:] = False
                zero_logprob[j] = torch.log(torch.max(1 - pred_probs[j, i, :][tmp_ind],
                                                      # torch.tensor([0.]).to(args.device))).sum()
                                                      torch.tensor([1e-9]).to(args.device))).sum()
                # print('##', i, j, len_gen[not_finished_idx][j], tmp_ind, min_par_idx[not_finished_idx][j],
                #       src_seq[not_finished_idx][j, :i+1, :])
            '''
            '''
            if i > 1 and src_seq[not_finished_idx][0, i, 1] != args.one_input:
                print('@ ', i, min_par_idx[not_finished_idx][0])
                print(src_seq[not_finished_idx][0, i ,:])
                print('\n', pred_probs[0, i, :])
                tmp_ind = ~ min_par_idx[not_finished_idx][0]
                tmp_ind[0] = False
                tmp_ind[i + 1:] = False
                print('\n', pred_probs[0, i, :][tmp_ind])
                input()
            '''
        elif args.use_max_prev_node and i >= args.max_prev_node:
            zero_logprob = torch.log(torch.max(1 - pred_probs[:, i, i - args.max_prev_node + 1: i + 1],
                                               # torch.tensor([0.]).to(args.device))).sum(dim=1)
                                               torch.tensor([1e-9]).to(args.device))).sum(dim=1)
            '''
            print('@@ ', i, args.max_prev_node)
            print('\n', pred_probs[0, i, :])
            print('\n', pred_probs[0, i, i - args.max_prev_node + 1: i + 1])
            input()
            '''
        else:
            zero_logprob = torch.log(torch.max(1 - pred_probs[:, i, 1: i + 1],
                                               # torch.tensor([0.]).to(args.device))).sum(dim=1)
                                               torch.tensor([1e-9]).to(args.device))).sum(dim=1)
            '''
            print('@@@ ', i)
            print('\n', pred_probs[0, i, :])
            print('\n', torch.max(1 - pred_probs[0, i, 1: i + 1],
                                  torch.tensor([0.]).to(args.device)))
                                  # torch.tensor([1e-9]).to(args.device)))
            input()
            '''
        if (not args.feed_graph_length) and args.use_termination_bit:
            zero_logprob = zero_logprob + torch.log(torch.max(1 - pred_probs[:, i, 0],
                                                              # torch.tensor([0.]).to(args.device)))
                                                              torch.tensor([1e-9]).to(args.device)))
        zero_prob = torch.exp(zero_logprob)

        # src_seq[not_finished_idx, i + 1, i + 1:] = args.dontcare_input
        gold = args.dontcare_input * torch.ones(not_finished_idx.sum().item(), src_seq.size(2)).to(args.device)
        still_zero_ind = torch.ones(not_finished_idx.sum().item()).bool().to(args.device)
        subset_zero_logprob = torch.zeros(not_finished_idx.sum().item()).to(args.device)
        for j in range(i + 1):
            if args.use_max_prev_node and i > args.max_prev_node and j > 0 and j < i - args.max_prev_node + 1:
                gold[:, j] = args.dontcare_input
            else:
                if j == 0 and args.estimate_num_nodes:
                    gold[:, 0] = args.zero_input
                    if (not args.feed_graph_length) and args.use_termination_bit:
                        subset_zero_logprob = subset_zero_logprob + torch.log(torch.max(1 - pred_probs[:, i, 0],
                                                                                     # torch.tensor([0.]).to(args.device)))
                                                                                     torch.tensor([1e-9]).to(args.device)))
                else:
                    q_pred = pred_probs[:, i, j].clone()
                    q_pred[still_zero_ind] = q_pred[still_zero_ind] * (1 + zero_prob[still_zero_ind] / \
                                             (torch.exp(subset_zero_logprob[still_zero_ind]) - zero_prob[still_zero_ind]))

                    if j == i:
                        # print('$$$$$$$$$$$ ', i, q_pred.min().item(), q_pred.max().item(),
                        #       q_pred[still_zero_ind].min().item(), q_pred[still_zero_ind].max().item())
                        q_pred[still_zero_ind] = 1.

                    tmp = (torch.rand([not_finished_idx.sum().item()], device=args.device) < q_pred).float()
                    ind_0 = tmp == 0
                    ind_1 = tmp == 1
                    tmp[ind_0] = args.zero_input
                    tmp[ind_1] = args.one_input
                    if args.use_bfs_incremental_parent_idx:
                        tmp[min_par_idx[not_finished_idx, j]] = args.zero_input
                    gold[:, j] = tmp
                    still_zero_ind[gold[:, j] == args.one_input] = False
                    if args.use_bfs_incremental_parent_idx:
                        ind_x = ind_0 & (~ min_par_idx[not_finished_idx, j])
                    else:
                        ind_x = ind_0
                    subset_zero_logprob[ind_x] = subset_zero_logprob[ind_x] + torch.log(torch.max(1 - pred_probs[ind_x, i, j],
                                                                                                  # torch.tensor([0.]).to(args.device)))
                                                                                                  torch.tensor([1e-9]).to(args.device)))
            if args.use_MADE and j < i:
                tmp = gg_model.trg_word_MADE(torch.cat([dec_output[:, i, :], prepare_for_MADE(gold, args)], dim=1))
                if gg_model.scale_prj:
                    tmp *= gg_model.d_model ** -0.5
                pred_probs[:, i, :] = torch.sigmoid(tmp)

        src_seq[not_finished_idx, i + 1, :i + 1] = gold[:, :i + 1]


        if args.estimate_num_nodes:
            new_finished_idx = torch.tensor(len_gen == i).to(args.device)
            src_seq[new_finished_idx, i+1, 0] = args.one_input
        else:
            new_finished_idx = not_finished_idx & (src_seq[:, i + 1, 0] == args.one_input)
        src_seq[new_finished_idx, i + 1, 1:] = args.src_pad_idx
        if i > 0 and args.use_bfs_incremental_parent_idx:
            tmp = src_seq[not_finished_idx, i + 1, :] == args.one_input
            min_par_idx[not_finished_idx, :] = tmp.cumsum(dim=1) == 0
            min_par_idx[not_finished_idx, 0] = False
        if args.estimate_num_nodes:
            not_finished_idx = not_finished_idx & torch.tensor(len_gen > i).to(args.device)
        else:
            not_finished_idx = not_finished_idx & (src_seq[:, i + 1, 0] != args.one_input)
        # if num_trials > 1:
        #     print('                          ', i, '      num of trials:', num_trials)
        if not_finished_idx.sum().item() == 0:
            break

        tmp = src_seq[not_finished_idx, i + 1, 1:i + 1]
        ind_0 = tmp == args.zero_input
        ind_1 = tmp == args.one_input
        tmp[ind_0] = 0
        tmp[ind_1] = 1
        adj[not_finished_idx, i + 1, 1:i + 1] = tmp
        adj[not_finished_idx, 1:i + 1, i + 1] = tmp

    ind_0 = src_seq == args.zero_input
    ind_1 = src_seq == args.one_input
    src_seq[ind_0] = 0
    src_seq[ind_1] = 1


    # save graphs as pickle
    G_pred_list = []
    for i in range(args.test_batch_size):
        adj_pred = my_decode_adj(src_seq[i,1:].cpu().detach().numpy(), args)
        G_pred = utils.get_graph(adj_pred) # get a graph from zero-padded adj
        G_pred_list.append(G_pred)

    return G_pred_list

def generate_graph_rejection(gg_model, args):

    if args.feed_graph_length:
        assert args.estimate_num_nodes

    # return None
    global min_par_idx
    gg_model.eval()

    if args.input_type == 'node_based':
        src_seq = torch.zeros((args.test_batch_size, args.max_seq_len), dtype=torch.long).to(args.device)
        for i in range(args.max_seq_len - 1):
            #pred = gg_model, *_ (src_seq, src_seq).max(1)[1].view([args.test_batch_size, args.max_seq_len])
            pred_logprobs, *_ = gg_model(src_seq, src_seq) #.max(1)[1].view([args.test_batch_size, args.max_seq_len])
            pred_probs = pred_logprobs.exp() / pred_logprobs.exp().sum(axis=-1, keepdim=True).repeat(1,pred_logprobs.size(-1))
            pred = torch.tensor([np.random.choice(np.arange(probs.size(0)),size=1,p=probs.detach().cpu().numpy())[0]
                                 for probs in pred_probs]).view([args.test_batch_size, args.max_seq_len]).to(args.device)
            src_seq[:, i + 1] = pred[:, i]
    elif args.input_type == 'preceding_neighbors_vector':

        if args.use_min_num_nodes:
            assert args.use_termination_bit

        src_seq = args.src_pad_idx * torch.ones((args.test_batch_size, args.max_seq_len, args.max_num_node + 1),
                                  dtype=torch.float32).to(args.device)


        adj = torch.zeros((args.test_batch_size, args.max_seq_len, args.max_seq_len), dtype=torch.float32).to(
            args.device)

        if args.estimate_num_nodes:
            len_gen = np.random.choice(np.arange(1,args.max_num_node + 1), args.test_batch_size, True, gg_model.num_nodes_prob[1:])
            if args.feed_graph_length:
                for i in range(src_seq.size(0)):
                    src_seq[i, len_gen[i]+1, 0] = args.one_input

        not_finished_idx = torch.ones([src_seq.size(0)]).bool().to(args.device)
        damaged_idx = torch.zeros([src_seq.size(0)]).bool().to(args.device)
        if args.use_bfs_incremental_parent_idx:
            min_par_idx = torch.zeros(src_seq.size(0), src_seq.size(2), dtype=torch.int32).bool().to(args.device)
        for i in range(args.max_seq_len - 1):

            tmp, dec_output = gg_model(src_seq, src_seq, src_seq, adj)
            pred_probs = torch.sigmoid(tmp).view(-1, args.max_seq_len, args.max_num_node + 1)
            # if args.use_max_prev_node and i > args.max_prev_node:
            #     pred_probs[:, i, 1:i - args.max_prev_node + 1] = 0
            # if args.use_bfs_incremental_parent_idx:
            #     for j in range(pred_probs.size(0)):
            #         pred_probs[j, i, 1:min_par_idx[j]] = 0
            num_trials = 0
            remainder_idx = not_finished_idx.clone()
            src_seq[remainder_idx, i+1, i+1:] = args.dontcare_input
            while remainder_idx.sum().item() > 0:
                num_trials += 1

                if args.use_MADE:
                    # if args.separate_termination_bit:
                    #     gold = args.trg_pad_idx * torch.ones(remainder_idx.sum().item(), src_seq.size(2) - 1).to(args.device)
                    #     term_bits = torch.rand([remainder_idx.sum().item()], device=args.device) < pred_probs[
                    #         remainder_idx, i, 0]
                    #     pred_probs = pred_probs[:, :, 1:]
                    # else:
                    gold = args.trg_pad_idx * torch.ones(remainder_idx.sum().item(), src_seq.size(2)).to(args.device)
                    for j in range(i + 1):
                        if args.use_max_prev_node and i > args.max_prev_node and j > 0 and j < i - args.max_prev_node + 1:
                            gold[remainder_idx, j] = args.dontcare_input
                        else:
                            if j == 0 and args.estimate_num_nodes:
                                gold[:, 0] = args.zero_input
                            elif j == 0 and args.use_min_num_nodes and i < min_num_node:
                                gold[:, 0] = args.zero_input
                            else:
                                tmp = (torch.rand([remainder_idx.sum().item()], device=args.device) < pred_probs[
                                    remainder_idx,
                                    i, j]).float()
                                ind_0 = tmp == 0
                                ind_1 = tmp == 1
                                tmp[ind_0] = args.zero_input
                                tmp[ind_1] = args.one_input
                                if args.use_bfs_incremental_parent_idx:
                                    tmp[min_par_idx[remainder_idx, j]] = args.zero_input
                                gold[:, j] = tmp
                        if j < i:
                            if args.separate_termination_bit:
                                tmp = gg_model.trg_word_MADE(torch.cat([dec_output[remainder_idx, i, :], prepare_for_MADE(gold[:,1:], args)], dim=1))
                                if gg_model.scale_prj:
                                    tmp *= gg_model.d_model ** -0.5
                                pred_probs[remainder_idx, i, 1:] = torch.sigmoid(tmp)
                            else:
                                tmp = gg_model.trg_word_MADE(torch.cat([dec_output[remainder_idx, i, :], prepare_for_MADE(gold, args)], dim=1))
                                if gg_model.scale_prj:
                                    tmp *= gg_model.d_model ** -0.5
                                pred_probs[remainder_idx, i, :] = torch.sigmoid(tmp)

                    src_seq[remainder_idx, i + 1, :i + 1] = gold[:, :i + 1]
                else:
                    tmp = (torch.rand([remainder_idx.sum().item(), i + 1], device=args.device) < pred_probs[remainder_idx,
                                                                                         i, :i + 1]).float()
                    if args.estimate_num_nodes:
                        tmp[:, 0] = 0
                    ind_0 = tmp == 0
                    ind_1 = tmp == 1
                    tmp[ind_0] = args.zero_input
                    tmp[ind_1] = args.one_input
                    src_seq[remainder_idx, i + 1, :i + 1] = tmp
                    if args.use_bfs_incremental_parent_idx:
                        src_seq[remainder_idx, i+1,:][min_par_idx[remainder_idx, :]] = args.zero_input
                    if args.use_max_prev_node and i > args.max_prev_node:
                        src_seq[remainder_idx, i+1, 1:i - args.max_prev_node + 1] = args.dontcare_input
                if i == 0:
                    src_seq[remainder_idx, i+1, 0] = args.zero_input
                    break

                if args.use_min_num_nodes and i < min_num_node:
                    src_seq[remainder_idx, i + 1, 0] = args.zero_input

                if args.estimate_num_nodes:
                    tmp_new_finished_idx = remainder_idx & torch.tensor(len_gen == i).to(args.device)
                    src_seq[remainder_idx, i+1, 0] = args.zero_input
                    src_seq[tmp_new_finished_idx, i+1, 0] = args.one_input

                if not args.use_termination_bit:
                    if args.estimate_num_nodes:
                        raise NotImplementedError
                    remainder_idx[:] = False
                    tmp_new_finished_idx = remainder_idx & ((src_seq[:, i+1, 1:i+1] == args.one_input).sum(-1) == 0)
                    src_seq[remainder_idx, i + 1, 0] = args.zero_input
                    src_seq[tmp_new_finished_idx, i + 1, 0] = args.one_input
                if args.allow_all_zeros:
                    remainder_idx[:] = False
                else:
                    remainder_idx = remainder_idx & ((src_seq[:, i + 1, : i + 1] == args.one_input).sum(-1) == 0)
                    if num_trials >= args.max_num_generate_trials:
                        # print('   reached max_num_gen_trials   dim:', i, '   num remainder:', remainder_idx.detach().cpu().sum().item())
                        if i+2 < args.max_seq_len:
                            src_seq[remainder_idx, i+2:, 0] = args.src_pad_idx
                        src_seq[remainder_idx, i+1, 0] = args.one_input
                        damaged_idx[remainder_idx] = True
                        remainder_idx[:] = False

            new_finished_idx = not_finished_idx & (src_seq[:, i + 1, 0] == args.one_input)
            src_seq[new_finished_idx, i + 1, 1:] = args.src_pad_idx
            if i > 0 and args.use_bfs_incremental_parent_idx:
                tmp = src_seq[not_finished_idx, i + 1, :] == args.one_input
                min_par_idx[not_finished_idx, :] = tmp.cumsum(dim=1) == 0
                min_par_idx[not_finished_idx, 0] = False
            not_finished_idx = not_finished_idx & (src_seq[:, i + 1, 0] != args.one_input)
            # if num_trials > 1:
            #     print('                          ', i, '      num of trials:', num_trials)
            if not_finished_idx.sum().item() == 0:
                break

            tmp = src_seq[not_finished_idx, i + 1, 1:i + 1]
            ind_0 = tmp == args.zero_input
            ind_1 = tmp == args.one_input
            tmp[ind_0] = 0
            tmp[ind_1] = 1
            adj[not_finished_idx, i + 1, 1:i + 1] = tmp
            adj[not_finished_idx, 1:i + 1, i + 1] = tmp

        ind_0 = src_seq == args.zero_input
        ind_1 = src_seq == args.one_input
        src_seq[ind_0] = 0
        src_seq[ind_1] = 1

        src_seq = src_seq[~damaged_idx]
    else:
        raise NotImplementedError


    # save graphs as pickle
    G_pred_list = []
    for i in range(src_seq.size(0)):
        adj_pred = my_decode_adj(src_seq[i,1:].cpu().numpy(), args)
        G_pred = utils.get_graph(adj_pred) # get a graph from zero-padded adj
        G_pred_list.append(G_pred)

    return G_pred_list

def generate_graph(gg_model, args):
    if args.exact_generation:
        return generate_graph_exact(gg_model, args)
    else:
        return generate_graph_rejection(gg_model, args)

def load_pretrained_model_weights(gg_model, iter, args):
    fname = '-'.join([s for s in args.fname.split('-') if s not in ('estnumnodes', 'useminnumnodes', 'exactgen')])
    fname = args.model_save_path + fname + '_' + args.graph_type + '_' + str(iter) + '.dat'
    gg_model.load_state_dict(torch.load(fname))

def just_generate(gg_model, dataset_train, args, gen_iter):
    if args.estimate_num_nodes:
        print('estimation of num_nodes_prob started')
        gg_model.num_nodes_prob = np.zeros(args.max_num_node + 1)
        for epoch in range(10):
            print(epoch, ' ', end='')
            sys.stdout.flush()
            for data in dataset_train:
                adj = data['adj'].to(args.device)
                for a in adj:
                    idx = a.sum(dim=0).bool().sum().item()
                    gg_model.num_nodes_prob[idx] += 1
        gg_model.num_nodes_prob = gg_model.num_nodes_prob / gg_model.num_nodes_prob.sum()
        print('estimation of num_nodes_prob finished')

    load_pretrained_model_weights(gg_model, gen_iter, args)


    for sample_time in range(1,2): #4):
        print('     sample_time:', sample_time)
        G_pred = []
        while len(G_pred)<args.test_total_size:
            print('        len(G_pred):', len(G_pred))
            G_pred_step = generate_graph(gg_model, args)
            G_pred.extend(G_pred_step)
        # save graphs
        fname = args.graph_save_path + args.fname_pred + str(gen_iter) + '_' + str(sample_time) + '.dat'
        utils.save_graph_list(G_pred, fname)
    print('test done, graphs saved')

    # np.save(args.timing_save_path+args.fname,time_all)

def just_test(gg_model, dataset):

    for epoch in range(0, args.epochs, args.epochs_save):
        fname = args.model_save_path + args.fname + '_' + args.graph_type + '_' + str(epoch) + '.dat'
        gg_model.load_state_dict(torch.load(fname))

        test_running_loss = 0.0
        tsz = 0
        gg_model.eval()
        for data in dataset:
            if args.use_MADE:
                gg_model.trg_word_MADE.update_masks()
            src_seq = data['src_seq'].to(args.device)
            trg_seq = data['src_seq'].to(args.device) 
            gold = data['trg_seq'].contiguous().to(args.device)
            adj = data['adj'].to(args.device)

            pred, dec_output = gg_model(src_seq, trg_seq, gold, adj)
            loss, *_ = cal_performance( pred, dec_output, gold, trg_pad_idx=0, args=args, model=gg_model, smoothing=False)

            test_running_loss += loss.item()
            tsz += src_seq.size(0)

        print('[epoch %d]     loss: %.3f' % (epoch + 1, test_running_loss / tsz))



def train(gg_model, dataset_train, dataset_validation, dataset_test, optimizer, args):

    ## initialize optimizer
    ## optimizer = torch.optim.Adam(list(gcade_model.parameters()), lr=args.lr)
    ## scheduler = MultiStepLR(optimizer, milestones=args.milestones, gamma=args.lr_rate)

    if args.estimate_num_nodes or args.weight_positions:
        print('estimation of num_nodes_prob started')
        num_nodes_prob = np.zeros(args.max_num_node + 1)
        for epoch in range(10):
            print(epoch, ' ', end='')
            sys.stdout.flush()
            for data in dataset_train:
                adj = data['adj'].to(args.device)
                for a in adj:
                    idx = a.sum(dim=0).bool().sum().item()
                    num_nodes_prob[idx] += 1
        num_nodes_prob = num_nodes_prob / num_nodes_prob.sum()
        print('estimation of num_nodes_prob finished')
        if args.estimate_num_nodes:
            gg_model.num_nodes_prob = num_nodes_prob
        if args.weight_positions:
            tmp = np.cumsum(num_nodes_prob, axis=0)
            tmp = 1 - tmp[:-1]
            tmp = np.concatenate([np.array([1.]), tmp])
            tmp[tmp <= 0] = np.min(tmp[tmp > 0])
            position_weights = 1 / tmp
            gg_model.positions_weights = torch.tensor(position_weights).to(args.device).view(1, -1)



    # start main loop
    time_all = np.zeros(args.epochs)
    loss_buffer = []
    if args.epoch_train_start > 0:
        load_pretrained_model_weights(gg_model, args.epoch_train_start - 1, args)
        optimizer.set_n_steps(args.epoch_train_start * args.batch_ratio)
    for epoch in range(args.epoch_train_start, args.epochs):
        time_start = time.time()
        running_loss = 0.0
        trsz = 0
        gg_model.train()
        for i, data in enumerate(dataset_train, 0):
            if args.use_MADE:
                gg_model.trg_word_MADE.update_masks()
            # print(' #', i)
            print('.', end='')
            sys.stdout.flush()
            src_seq = data['src_seq'].to(args.device)
            trg_seq = data['src_seq'].to(args.device)

            '''
            for j in range(src_seq.size(1)):
                ind = src_seq[:,j,0] == args.zero_input
                tmp = args.dontcare_input * torch.ones(ind.sum().item(), src_seq.size(-1)).to(args.device)

                # tmp[:, :] = args.zero_input
                tmp[:, :j] = args.zero_input
                tmp[:, j] = args.one_input

                # tmp[:, :] = args.one_input
                # tmp[:, 0] = args.zero_input

                src_seq[ind, j,  :] = tmp.clone()
                trg_seq[ind, j,  :] = tmp.clone()
            '''

            gold = data['trg_seq'].contiguous().to(args.device)
            adj = data['adj'].to(args.device)

            optimizer.zero_grad()
            pred, dec_output = gg_model(src_seq, trg_seq, gold, adj)
            if (not args.weight_termination_bit) or (epoch > args.termination_bit_weight_last_epoch):
                loss, *_ = cal_performance( pred, dec_output, gold, trg_pad_idx=0, args=args, model=gg_model, smoothing=False)
            else:
                tmp = (args.termination_bit_weight_last_epoch - epoch) / args.termination_bit_weight_last_epoch
                termination_bit_weight = (tmp ** 2) * (args.termination_bit_weight - 1) + 1

                print('                   tbw: ', termination_bit_weight)
                loss, *_ = cal_performance( pred, dec_output, gold, trg_pad_idx=0, args=args, model=gg_model,
                                            termination_bit_weight=termination_bit_weight, smoothing=False)

            # print('  ', loss.item() / input_nodes.size(0))
            loss.backward()
            optimizer.step_and_update_lr()

            running_loss += loss.item()
            trsz += src_seq.size(0)

        val_running_loss = 0.0
        vlsz = 0
        gg_model.eval()
        for i, data in enumerate(dataset_validation):
            if args.use_MADE:
                gg_model.trg_word_MADE.update_masks()
            src_seq = data['src_seq'].to(args.device)
            trg_seq = data['src_seq'].to(args.device) 
            gold = data['trg_seq'].contiguous().to(args.device)
            adj = data['adj'].to(args.device)

            pred, dec_output = gg_model(src_seq, trg_seq, gold, adj)
            loss, *_ = cal_performance( pred, dec_output, gold, trg_pad_idx=0, args=args, model=gg_model, smoothing=False)

            val_running_loss += loss.item()
            vlsz += src_seq.size(0)

        test_running_loss = 0.0
        testsz = 0
        gg_model.eval()
        for i, data in enumerate(dataset_test):
            if args.use_MADE:
                gg_model.trg_word_MADE.update_masks()
            src_seq = data['src_seq'].to(args.device)
            trg_seq = data['src_seq'].to(args.device)
            gold = data['trg_seq'].contiguous().to(args.device)
            adj = data['adj'].to(args.device)

            pred, dec_output = gg_model(src_seq, trg_seq, gold, adj)
            loss, *_ = cal_performance(pred, dec_output, gold, trg_pad_idx=0, args=args, model=gg_model, smoothing=False)

            test_running_loss += loss.item()
            testsz += src_seq.size(0)

        if epoch % args.epochs_save == 0:
            fname = args.model_save_path + args.fname + '_' + args.graph_type + '_'  + str(epoch) + '.dat'
            torch.save(gg_model.state_dict(), fname)

        loss_buffer.append(running_loss / trsz)
        if len(loss_buffer) > 5:
            loss_buffer = loss_buffer[1:]
        print('[epoch %d]     loss: %.3f     val: %.3f     test: %.3f              lr: %f     avg_tr_loss: %f' %
              (epoch + 1, running_loss / trsz, val_running_loss / vlsz, test_running_loss / testsz, optimizer._optimizer.param_groups[0]['lr'], np.mean(loss_buffer))) #get_lr(optimizer)))
        # print(list(gg_model.encoder.layer_stack[0].slf_attn.gr_att_linear_list[0].parameters()))
        sys.stdout.flush()
        time_end = time.time()
        time_all[epoch - 1] = time_end - time_start
        # test
        if epoch % args.epochs_test == 0 and epoch >= args.epochs_test_start:
            for sample_time in range(1,2): #4):
                print('     sample_time:', sample_time)
                G_pred = []
                while len(G_pred)<args.test_total_size:
                    print('        len(G_pred):', len(G_pred))
                    G_pred_step = generate_graph(gg_model, args)
                    G_pred.extend(G_pred_step)
                # save graphs
                fname = args.graph_save_path + args.fname_pred + str(epoch) + '_' + str(sample_time) + '.dat'
                utils.save_graph_list(G_pred, fname)
            print('test done, graphs saved')

    # np.save(args.timing_save_path+args.fname,time_all)


'''
while True:
    for i, data in enumerate(test_dataset_loader):
        adj = data['adj'].to(args.device)
        print('           ##', i, '  ', adj[0, 0, :].sum().item(), '  ', adj.size(0))
        print('           ##', i, '  ', adj[0, 1, :].sum().item(), '  ', adj.size(0))
        print('           ##', i, '  ', adj[0, 2, :].sum().item(), '  ', adj.size(0))
        print('           ##', i, '  ', adj[0, 3, :].sum().item(), '  ', adj.size(0))
        print('           ##', i, '  ', adj[0, 4, :].sum().item(), '  ', adj.size(0))
    print()
    input()
'''

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-g', '--generate', help='number of generation iteration (just generate)', action='store', type=int)
    parser.add_argument('-v', '--validate', help='just validate', action='store_true')
    parser.add_argument('-t', '--test', help='just test', action='store_true')
    console_args = parser.parse_args()

    if console_args.generate is not None:
        just_generate(model, dataset_loader, args, console_args.generate)
    elif console_args.validate:
        just_test(model, val_dataset_loader)
    elif console_args.test:
        just_test(model, test_dataset_loader)
    else:
        train(model, dataset_loader, val_dataset_loader, test_dataset_loader, optimizer, args)

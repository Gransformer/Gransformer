import torch.nn as nn
import torch

class BaseArgs():
    def __init__(self, graph_type, note, batch_ratio):

        self.graph_type = graph_type

        self.use_pre_saved_graphs = True

        # if none, then auto calculate
        self.max_num_node = None  # max number of nodes in a graph
        self.max_prev_node = None  # max previous node that looks back
        self.max_seq_len = None

        ### output config
        # self.dir_input = "/dfs/scratch0/jiaxuany0/"
        self.dir_input = "./"
        self.model_save_path = self.dir_input + 'model_save/'  # only for nll evaluation
        self.graph_save_path = self.dir_input + 'graphs/'
        self.figure_save_path = self.dir_input + 'figures/'
        self.timing_save_path = self.dir_input + 'timing/'
        self.figure_prediction_save_path = self.dir_input + 'figures_prediction/'
        self.nll_save_path = self.dir_input + 'nll/'


        ### training config
        # self.num_workers = 4  # num workers to load data, default 4


        self.sample_time = 2  # sample time in each time step, when validating

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print('self.device:', self.device)


        self.node_ordering = 'bfs' # 'blanket' #
        self.use_max_prev_node = False # True #   ### ignored when using an input_type other than preceding_neighbors_vector'
        if self.use_max_prev_node:
            assert self.node_ordering in ['bfs']

        self.max_num_generate_trials = 1000

        self.input_type = 'preceding_neighbors_vector' # 'max_prev_node_neighbors_vec' # 'node_based' #
        self.only_encoder = True # False

        self.input_bfs_depth = False # True #
        if self.input_bfs_depth:
            assert self.node_ordering in ['bfs']

        if self.input_type == 'node_based':
            self.trg_pad_idx = 0
            self.src_pad_idx = 0
        elif self.input_type in ['preceding_neighbors_vector', 'max_prev_node_neighbors_vec']:
            self.trg_pad_idx = -2
            self.src_pad_idx = -2  # must be equal to self.trg_pad_idx
            # self.avg = False # True #
            # if self.avg == True:
            #     self.zero_input = 0
            #     self.one_input = 1
            #     self.dontcare_input = 0
            # else:
            if True:
                self.zero_input = -1
                self.one_input = 1
                self.dontcare_input = 0
        else:
            raise NotImplementedError


        self.dropout = 0.1
        self.proj_share_weight = False # True
        self.embs_share_weight = True
        self.scale_emb_or_prj = 'prj'

        ### output
        self.use_tb = False  # use tensorboard
        self.output_dir = './output'



        self.ensemble_input_type = 'repeat'  # 'multihop-single' # 'negative' # 'multihop' #

        ################## Default values --- These config parameters can be changed by note argument. ###############33
        self.n_head = 1  # 8
        if self.ensemble_input_type == 'negative':
            self.n_ensemble = 2
        elif self.ensemble_input_type == 'multihop':
            self.ensemble_multihop = [2]
            self.n_ensemble = len(self.ensemble_multihop) + 1
        elif self.ensemble_input_type == 'multihop-single':
            self.ensemble_multihop = [1, 2, 3, 4]
            self.n_ensemble = 1
        elif self.ensemble_input_type == 'repeat':
            self.n_ensemble = 1  # 8
        else:
            raise NotImplementedError('ensemble_input_type', self.ensemble_input_type, 'not recognized.')
        self.use_bfs_incremental_parent_idx = False  # True #    ### so far just implemented for max_pre_node_neighbors_vec input_type
        self.output_positional_embedding = None  # one_hot # tril
        self.k_graph_attention = 0  # 4
        self.normalize_graph_attention = False  # True #
        self.batchnormalize_graph_attention = False  # True #
        self.log_graph_attention = False  # True #
        self.k_graph_positional_encoding = 0  # 4
        self.normalize_graph_positional_encoding = False  # True #
        self.batchnormalize_graph_positional_encoding = False  # True #
        self.log_graph_positional_encoding = False  # True #
        self.use_MADE = False  # True #
        self.MADE_num_masks = 3  # 1
        self.MADE_natural_ordering = False  # True #
        self.MADE_num_hidden_layers = 1  # 3
        self.MADE_dim_reduction_factor = 1
        self.n_layers = 2 # 6
        self.n_grlayers = 0
        self.no_model_layer_norm = False # True #
        assert self.n_grlayers <= self.n_layers
        assert self.n_grlayers == 0 or self.k_graph_attention == 0
        self.estimate_num_nodes = False # True #
        self.typed_edges = False # True
        self.allow_all_zeros = False # True
        self.use_termination_bit = True # False
        self.weight_positions = False # True
        self.separate_termination_bit = False # True
        self.use_min_num_nodes = False # True
        self.sep_optimizer_start_step = 1000000000
        self.weight_termination_bit = False
        self.feed_graph_length = False
        self.exact_generation = False
        #################################################################################

        self.note = note

        note_params = self.note.split('-')
        for param in note_params[1:]:

            if param.endswith('grlayers'):
                self.n_grlayers = int(param[:-8])
            elif param.endswith('grlayer'):
                self.n_grlayers = int(param[:-7])
            elif param.endswith('layers'):
                self.n_layers = int(param[:-6])
            elif param.endswith('layer'):
                self.n_layers = int(param[:-5])
            elif param.startswith('posoutput'):
                if param.endswith('oneHot'):
                    self.output_positional_embedding = 'one_hot'
                elif param.endswith('tril'):
                    self.output_positional_embedding = 'tril'
                else:
                    raise Exception("unknown parameter", str(param))
            elif param == 'bfsincpar':
                self.use_bfs_incremental_parent_idx = True
            elif param.startswith('gattk'):
                if param.endswith('batchnorm'):
                    self.k_graph_attention = int(param[5:-9])
                    self.batchnormalize_graph_attention = True
                elif param.endswith('norm'):
                    self.k_graph_attention = int(param[5:-4])
                    self.normalize_graph_attention = True
                elif param.endswith('log'):
                    self.k_graph_attention = int(param[5:-3])
                    self.log_graph_attention = True
                else:
                    self.k_graph_attention = int(param[5:])
            elif param.startswith('grposenck'):
                if param.endswith('batchnorm'):
                    self.k_graph_positional_encoding = int(param[9:-9])
                    self.batchnormalize_graph_positional_encoding = True
                elif param.endswith('norm'):
                    self.k_graph_positional_encoding = int(param[9:-4])
                    self.normalize_graph_positional_encoding = True
                elif param.endswith('log'):
                    self.k_graph_positional_encoding = int(param[9:-3])
                    self.log_graph_positional_encoding = True
                else:
                    self.k_graph_positional_encoding = int(param[9:])
            elif param.startswith('nhead'):
                self.n_head = int(param[5:])
            elif param.startswith('nensemble'):
                self.n_ensemble = int(param[9:])
            elif param.startswith('MADE'):
                self.use_MADE = True
                assert param[4:6] == 'hl'
                self.MADE_num_hidden_layers = int(param[6])
                assert param[7:10] == 'msk'
                if param[11:18] == 'natuord':
                    self.MADE_num_masks = int(param[10])
                    self.MADE_natural_ordering = bool(param[18])
                    assert param[19:25] == 'dimred'
                    self.MADE_dim_reduction_factor = int(param[25:])
                elif param[12:19] == 'natuord':
                    self.MADE_num_masks = int(param[10:12])
                    self.MADE_natural_ordering = bool(param[19])
                    assert param[20:26] == 'dimred'
                    self.MADE_dim_reduction_factor = int(param[26:])
                else:
                    raise Exception('Unknown note')
            elif param == 'estnumnodes':
                self.estimate_num_nodes = True
            elif param == 'nomodellayernorm':
                self.no_model_layer_norm = True
            elif param.startswith('trainpr'):
                tmp = param.split(',')
                self.training_portion = float(tmp[0][7:])
                self.validation_portion = float(tmp[1][5:])
                self.test_portion = float(tmp[2][6:])
            elif param == 'typededges':
                self.typed_edges = True
            elif param == 'allowAllZeros':
                self.allow_all_zeros = True
            elif param == 'noTerminationBit':
                self.use_termination_bit = False
            elif param == 'weightpositions':
                self.weight_positions = True
            elif param.startswith('separateTerminationBit'):
                self.separate_termination_bit = True
                if len(param) > 22:
                    self.sepTermBitNumLayers = int(param[22:])
                else:
                    self.sepTermBitNumLayers = 1
            elif param == 'useminnumnodes':
                self.use_min_num_nodes = True
            elif param.startswith('sepoptepoch'):
                self.sep_optimizer_start_step = batch_ratio * int(param[11:])
            elif param.startswith('weightterminationbit'):
                tmp = param.split(',')
                self.weight_termination_bit = True
                self.termination_bit_weight = float(tmp[1])
                self.termination_bit_weight_last_epoch = int(tmp[2])
            elif param == 'feedgraphlength':
                self.feed_graph_length = True
            elif param == 'exactgen':
                self.exact_generation = True
            else:
                raise Exception('Unknown note')

        ### filenames to save intemediate and final outputs
        # self.fname = self.note + '_' + self.graph_type + '_' + str(self.num_layers) + '_' + str(
        self.fname = self.note
        #     self.hidden_size_rnn) + '_'
        # self.fname_pred = self.note + '_' + self.graph_type + '_' + str(self.num_layers) + '_' + str(
        #     self.hidden_size_rnn) + '_pred_'
        self.fname_pred = self.note + '_' + self.graph_type + '_' + self.input_type +  '_pred_'
        # self.fname_train = self.note + '_' + self.graph_type + '_' + str(self.num_layers) + '_' + str(
        #     self.hidden_size_rnn) + '_train_'
        self.fname_train = self.note.split('-')[0] + '_' + self.graph_type + '_' + self.input_type + '_train_'
        # self.fname_test = self.note + '_' + self.graph_type + '_' + str(self.num_layers) + '_' + str(
        #     self.hidden_size_rnn) + '_test_'
        self.fname_test = self.note.split('-')[0] + '_' + self.graph_type + '_' + self.input_type + '_test_'
        # self.fname_baseline = self.graph_save_path + self.graph_type + self.generator_baseline + '_' + self.metric_baseline


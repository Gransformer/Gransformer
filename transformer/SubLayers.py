''' Define the sublayers in encoder/decoder layer '''
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
# from transformer.Modules import ScaledDotProductAttention
import torch


# class MultiHeadAttention(nn.Module):
#     ''' Multi-Head Attention module '''
#
#     def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
#         super().__init__()
#
#         self.n_head = n_head
#         self.d_k = d_k
#         self.d_v = d_v
#
#         self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
#         self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
#         self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
#         self.fc = nn.Linear(n_head * d_v, d_model, bias=False)
#
#         self.attention = ScaledDotProductAttention(temperature=d_k ** 0.5)
#
#         self.dropout = nn.Dropout(dropout)
#         self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
#
#
#     def forward(self, q, k, v, mask=None):
#
#         d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
#         sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)
#
#         residual = q
#
#         # Pass through the pre-attention projection: b x lq x (n*dv)
#         # Separate different heads: b x lq x n x dv
#         q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
#         k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
#         v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)
#
#         # Transpose for attention dot product: b x n x lq x dv
#         q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
#
#         if mask is not None:
#             mask = mask.unsqueeze(1)   # For head axis broadcasting.
#
#         q, attn = self.attention(q, k, v, mask=mask)
#
#         # Transpose to move the head dimension back: b x lq x n x dv
#         # Combine the last two dimensions to concatenate all the heads together: b x lq x (n*dv)
#         q = q.transpose(1, 2).contiguous().view(sz_b, len_q, -1)
#         q = self.dropout(self.fc(q))
#         q += residual
#
#         q = self.layer_norm(q)
#
#         return q, attn


class EnsembleMultiHeadAttention(nn.Module):
    '''Ensemble  Multi-Head Attention module '''

    def __init__(self, n_ensemble_q, n_ensemble_k, n_head, d_model, d_k, d_v, no_layer_norm,
                 typed_edges, k_gr_att=0, gr_att_batchnorm=False, dropout=0.1, attn_dropout=0.1):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v
        self.d_model = d_model
        self.no_layer_norm = no_layer_norm
        self.typed_edges = typed_edges

        self.n_ensemble_q = n_ensemble_q
        self.n_ensemble_k = n_ensemble_k

        self.gr_att_batchnorm = gr_att_batchnorm

        if k_gr_att > 0:
            if self.gr_att_batchnorm:
                self.gr_att_layer_norm = nn.LayerNorm(k_gr_att, eps=1e-3)
            self.gr_att_linear_list_1 = nn.ModuleList([
                nn.Linear(k_gr_att, k_gr_att, bias=True)
                for _ in range(n_ensemble_q * n_ensemble_k * n_head)
            ])
            self.gr_att_linear_list_2 = nn.ModuleList([
                nn.Linear(k_gr_att, 1, bias=True)
                for _ in range(n_ensemble_q * n_ensemble_k * n_head)
            ])

        self.w_qs_list = nn.ModuleList([
            nn.Linear(d_model, n_head * d_k, bias=False)
            for _ in range(n_ensemble_q)
        ])
        self.w_ks_list = nn.ModuleList([
            nn.Linear(d_model, n_head * d_k, bias=False)
            for _ in range(n_ensemble_k)
        ])
        self.w_vs_list = nn.ModuleList([
            nn.Linear(d_model, n_head * d_v, bias=False)
            for _ in range(n_ensemble_k)
        ])

        if self.typed_edges:
            self.w_qs_list_2 = nn.ModuleList([
                nn.Linear(d_model, n_head * d_k, bias=False)
                for _ in range(n_ensemble_q)
            ])
            self.w_ks_list_2 = nn.ModuleList([
                nn.Linear(d_model, n_head * d_k, bias=False)
                for _ in range(n_ensemble_k)
            ])
            self.w_vs_list_2 = nn.ModuleList([
                nn.Linear(d_model, n_head * d_v, bias=False)
                for _ in range(n_ensemble_k)
            ])

        self.fc_list = nn.ModuleList([
            nn.Linear(n_head * d_v, d_model, bias=False)
            for _ in range(n_ensemble_q)
        ])


        self.attn_dropout = nn.Dropout(attn_dropout)
        self.temperature=d_k ** 0.5
        # self.attention = EnsembleScaledDotProductAttention(temperature=d_k ** 0.5)

        self.dropout = nn.Dropout(dropout)
        if not no_layer_norm:
            self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)


    def forward(self, q, k, v, mask=None, gr_mask=None, adj=None):

        # q is b x lq x n_ens x d

        d_k, d_v, d_model = self.d_k, self.d_v, self.d_model
        n_head, n_ensemble_q, n_ensemble_k = self.n_head, self.n_ensemble_q, self.n_ensemble_k
        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)

        residual = q

        if mask is not None:
            mask = mask.unsqueeze(1)   # For head axis broadcasting.

        q_prj = torch.zeros(sz_b, len_q, n_ensemble_q, n_head * d_k, device=q.device)
        for i, w_qs in enumerate(self.w_qs_list):
            q_prj[ :, :, i, :] = w_qs( q[ :, :, i, :])

        k_prj = torch.zeros(sz_b, len_k, n_ensemble_k, n_head * d_k, device=k.device)
        for i, w_ks in enumerate(self.w_ks_list):
            k_prj[ :, :, i, :] = w_ks( k[ :, :, i, :])

        v_prj = torch.zeros(sz_b, len_k, n_ensemble_k, n_head * d_v, device=v.device)
        for i, w_vs in enumerate(self.w_vs_list):
            v_prj[ :, :, i, :] = w_vs( v[ :, :, i, :])

        v_prj_tmp = v_prj.view(sz_b, len_k, n_ensemble_k, n_head, d_v).transpose(2,3).transpose(1,2).view(
            sz_b, n_head, len_k * n_ensemble_k, d_v
        )

        if self.typed_edges:
            q_prj_2 = torch.zeros(sz_b, len_q, n_ensemble_q, n_head * d_k, device=q.device)
            for i, w_qs in enumerate(self.w_qs_list_2):
                q_prj_2[:, :, i, :] = w_qs(q[:, :, i, :])

            k_prj_2 = torch.zeros(sz_b, len_k, n_ensemble_k, n_head * d_k, device=k.device)
            for i, w_ks in enumerate(self.w_ks_list_2):
                k_prj_2[:, :, i, :] = w_ks(k[:, :, i, :])

            v_prj_2 = torch.zeros(sz_b, len_k, n_ensemble_k, n_head * d_v, device=v.device)
            for i, w_vs in enumerate(self.w_vs_list_2):
                v_prj_2[:, :, i, :] = w_vs(v[:, :, i, :])

            # v_prj_tmp_2 = v_prj_2.view(sz_b, len_k, n_ensemble_k, n_head, d_v).transpose(2, 3).transpose(1, 2).view(
            #     sz_b, n_head, len_k * n_ensemble_k, d_v
            # )


        if self.typed_edges:
            adj_tril = torch.tril(adj, diagonal=0)
            adj_tril = adj_tril.unsqueeze(1).repeat(1, n_head, 1, 1)

        pre_output = torch.zeros(sz_b, n_head, len_q, n_ensemble_q, d_v, device=v.device)

        for i in range(n_ensemble_q):
            q_ens = q_prj[:, :, i, :].view(sz_b, len_q, n_head, d_k)

            # Transpose for attention dot product: b x n_head x lq x dv
            q_ens = q_ens.transpose(1, 2)

            attn = torch.zeros(sz_b, n_head, len_q, len_k, n_ensemble_k, device=q_prj.device)

            if self.typed_edges:
                q_ens_2 = q_prj_2[:, :, i, :].view(sz_b, len_q, n_head, d_k)
                q_ens_2 = q_ens_2.transpose(1, 2)

            for j in range(n_ensemble_k):

                # Pass through the pre-attention projection: b x lq x (n_head*dv)
                # Separate different heads: b x lq x n x dv
                k_ens = k_prj[:,:,j,:].view(sz_b, len_k, n_head, d_k)
                v_ens = v_prj[:, :, j, :].view(sz_b, len_v, n_head, d_v)

                # Transpose for attention dot product: b x n x lq x dv
                k_ens, v_ens = k_ens.transpose(1, 2), v_ens.transpose(1, 2)

                ################### ScaledDotProductAttention ######################
                attn_ = torch.matmul(q_ens / self.temperature, k_ens.transpose(2, 3))
                '''
                attn_ = torch.zeros(sz_b, n_head, len_q, len_k).to(v.device)
                '''

                if self.typed_edges:
                    k_ens_2 = k_prj_2[:, :, j, :].view(sz_b, len_k, n_head, d_k)
                    v_ens_2 = v_prj_2[:, :, j, :].view(sz_b, len_v, n_head, d_v)
                    k_ens_2, v_ens_2 = k_ens_2.transpose(1, 2), v_ens_2.transpose(1, 2)
                    attn_2_ = torch.matmul(q_ens_2 / self.temperature, k_ens_2.transpose(2, 3))
                    attn_ = attn_ * adj_tril + attn_2_ * (1 - adj_tril)

                if mask is not None:
                    if gr_mask is not None:
                        for h in range(self.n_head):
                            gr_att_linear_1 = self.gr_att_linear_list_1[i * n_ensemble_k * n_head + j * n_head + h]
                            gr_att_linear_2 = self.gr_att_linear_list_2[i * n_ensemble_k * n_head + j * n_head + h]

                            gr_mask_agg = torch.zeros(gr_mask.size(0), gr_mask.size(2), gr_mask.size(3)).to(gr_mask.device)
                            ind = mask[:, 0, :, :] == 1
                            ind_tmp = ind.unsqueeze(-1).repeat(1,1,1,gr_mask.size(1))
                            tmp = gr_mask.transpose(1,2).transpose(2,3)[ind_tmp].reshape(-1, gr_mask.size(1))
                            if self.gr_att_batchnorm:
                                tmp = self.gr_att_layer_norm(tmp)
                            tmp_2 = gr_att_linear_2(F.relu(gr_att_linear_1(tmp))).reshape(-1)
                            gr_mask_agg[ind] = tmp_2

                            # attn_[:,h,:,:] = attn_[:,h,:,:] + gr_mask_agg
                            attn_[:,h,:,:] = attn_[:,h,:,:] + torch.log(torch.sigmoid(gr_mask_agg))
                            # attn_[:,h,:,:] = attn_[:,h,:,:] * gr_mask_agg
                            # attn_[:,h,:,:] = gr_mask_agg
                            # attn_[:,h,:,:] = attn_[:,h,:,:].masked_fill(gr_mask_agg == 0, -1e9)
                            # attn_[:,h,:,:] = 1
                    attn_ = attn_.masked_fill(mask == 0, -1e9)
                attn[:,:,:,:,j] = attn_

            '''
            attn = F.sigmoid(attn)
            attn = attn.view(sz_b, n_head, len_q, len_k * n_ensemble_k)
            attn = self.attn_dropout(attn)
            '''

            attn = attn.view(sz_b, n_head, len_q, len_k * n_ensemble_k)
            attn = self.attn_dropout(F.softmax(attn, dim=-1))

            pre_output[:, :, :, i, :] = torch.matmul(attn, v_prj_tmp)
            ########################################################################

        # Transpose to move the head dimension back: b x lq x n_head x n_ens x dv
        # Combine the last two dimensions to concatenate all the heads together: b x lq x (n_head*dv)
        pre_output = pre_output.transpose(1,2).transpose(2,3).reshape(sz_b, len_q, n_ensemble_q, -1)

        output = torch.zeros(sz_b, len_q, n_ensemble_q, d_model, device=pre_output.device)
        for i, fc in enumerate(self.fc_list):
            output[ :, :, i, :] = self.dropout(fc(pre_output[ :, :, i, :]))

        # if residual.size(2) == 1:
        #     residual = residual.repeat(1,1,n_ensemble_q,1)
        output += residual
        if not self.no_layer_norm:
            output = self.layer_norm(output)

        return output, None # returns None for attn


class PositionwiseFeedForward(nn.Module):
    ''' A two-feed-forward-layer module '''

    def __init__(self, d_in, d_hid, no_layer_norm, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_in, d_hid) # position-wise
        self.w_2 = nn.Linear(d_hid, d_in) # position-wise
        self.no_layer_norm = no_layer_norm
        if not no_layer_norm:
            self.layer_norm = nn.LayerNorm(d_in, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):

        residual = x

        x = self.w_2(F.relu(self.w_1(x)))
        x = self.dropout(x)
        x += residual

        if not self.no_layer_norm:
            x = self.layer_norm(x)

        return x


# sz_b = 5
# n_ensemble_q = n_ensemble_k = 2
# n_head = 1
# d_model = 3
# d_k = d_v = 4
#
# ensembleMultiHeadAttention = EnsembleMultiHeadAttention(n_ensemble_q, n_ensemble_k, n_head, d_model, d_k, d_v)
#
# len_q = len_k = len_v = 6
# q = torch.rand([sz_b, len_q, n_ensemble_q, d_model])
# k = torch.rand([sz_b, len_k, n_ensemble_k, d_model])
# v = torch.rand([sz_b, len_v, n_ensemble_k, d_model])
#
# mask = torch.tensor([[1,0,0,0,0,0],
#                      [1,0,0,0,0,0],
#                      [0,1,1,0,0,0],
#                      [0,0,1,1,0,0],
#                      [0,0,1,0,0,0],
#                      [0,1,0,0,1,0],]).bool().unsqueeze(0).repeat(sz_b, 1, 1)
#
# output,_ = ensembleMultiHeadAttention(q, k, v, mask)
# print(output.size())
# print(output)

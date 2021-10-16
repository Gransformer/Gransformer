
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

# ------------------------------------------------------------------------------

class MaskedLinear(nn.Linear):
    """ same as Linear except has a configurable mask on the weights """
    
    def __init__(self, in_features, out_features, bias=True):
        super().__init__(in_features, out_features, bias)        
        self.register_buffer('mask', torch.ones(out_features, in_features))
        
    def set_mask(self, mask):
        self.mask.data.copy_(torch.from_numpy(mask.astype(np.uint8).T))
        
    def forward(self, input):
        return F.linear(input, self.mask * self.weight, self.bias)
        # return F.linear(input, self.weight, self.bias)

class MADE(nn.Module):
    def __init__(self, nin_extra, hidden_sizes, nout, num_masks=1, natural_ordering=False):
        """
        nin: integer; number of inputs
        hidden sizes: a list of integers; number of units in hidden layers
        nout: integer; number of outputs, which usually collectively parameterize some kind of 1D distribution
              note: if nout is e.g. 2x larger than nin (perhaps the mean and std), then the first nin
              will be all the means and the second nin will be stds. i.e. output dimensions depend on the
              same input dimensions in "chunks" and should be carefully decoded downstream appropriately.
              the output of running the tests for this file makes this a bit more clear with examples.
        num_masks: can be used to train ensemble over orderings/connections
        natural_ordering: force natural ordering of dimensions, don't use random permutations
        """
        
        super().__init__()
        self.nin_extra = nin_extra
        self.nout = nout
        self.hidden_sizes = hidden_sizes
        # assert self.nout % self.nin == 0, "nout must be integer multiple of nin"
        
        # define a simple MLP neural net
        self.net_1 = []
        hs = [nout + nin_extra] + hidden_sizes # + [nout]
        # hs = [nin_extra] + hidden_sizes + [nout]
        for h0,h1 in zip(hs, hs[1:]):
            self.net_1.extend([
                    MaskedLinear(h0, h1),
                    nn.ReLU(),
                ])
        # self.net_1.pop() # pop the last ReLU for the output layer
        self.net_1 = nn.Sequential(*self.net_1)

        self.net_2 = MaskedLinear(nout + nin_extra + hidden_sizes[-1], nout)


        # seeds for orders/connectivities of the model ensemble
        self.natural_ordering = natural_ordering
        self.num_masks = num_masks
        self.seed = 0 # for cycling through num_masks orderings
        
        self.m = {}
        self.update_masks() # builds the initial self.m connectivity
        # note, we could also precompute the masks and cache them, but this
        # could get memory expensive for large number of masks.
        
    def update_masks(self):
        if self.m and self.num_masks == 1: return # only a single seed, skip for efficiency
        L = len(self.hidden_sizes)
        
        # fetch the next seed and construct a random stream
        rng = np.random.RandomState(self.seed)
        self.seed = (self.seed + 1) % self.num_masks
        
        # sample the order of the inputs and the connectivity of all neurons
        self.m[-2] = np.arange(self.nout) if self.natural_ordering else rng.permutation(self.nout)
        t = np.where(self.m[-2] == 0)[0][0]
        self.m[-2][t] = self.m[-2][0]
        self.m[-2][0] = 0
        self.m[-1] = np.concatenate([-1 * np.ones(self.nin_extra), self.m[-2].copy()])
        # self.m[-1] = -1 * np.ones(self.nin_extra)
        for l in range(L):
            self.m[l] = rng.randint(self.m[l-1].min(), self.nout-1, size=self.hidden_sizes[l])
            # self.m[l] = -1 * np.ones([self.hidden_sizes[l]])
        
        # construct the mask matrices
        masks = [self.m[l-1][:,None] <= self.m[l][None,:] for l in range(L)]
        masks.append( np.concatenate([self.m[-1].copy(), self.m[L-1]])[:,None] < self.m[-2][None,:])

        # set the masks in all MaskedLinear layers
        layers = [l for l in self.net_1.modules() if isinstance(l, MaskedLinear)] + [self.net_2]
        for l,m in zip(layers, masks):
            l.set_mask(m)
    
    def forward(self, x):
        y_1 = self.net_1(x)
        return self.net_2(torch.cat([x, y_1], dim=-1))
        #return self.net(x)
        # return self.net(x[:,:,:self.nin_extra])
        # return x[:,:,-self.nout:] * 100

# ------------------------------------------------------------------------------

# if __name__ == '__main__':
#     from torch.autograd import Variable
#
#     # run a quick and dirty test for the autoregressive property
#     D = 10
#     rng = np.random.RandomState(14)
#     x = (rng.rand(1, D) > 0.5).astype(np.float32)
#
#     configs = [
#         (D, [], D, False),                 # test various hidden sizes
#         (D, [200], D, False),
#         (D, [200, 220], D, False),
#         (D, [200, 220, 230], D, False),
#         (D, [200, 220], D, True),          # natural ordering test
#         (D, [200, 220], 2*D, True),       # test nout > nin
#         (D, [200, 220], 3*D, False),       # test nout > nin
#     ]
#
#     for nin, hiddens, nout, natural_ordering in configs:
#
#         print("checking nin %d, hiddens %s, nout %d, natural %s" %
#              (nin, hiddens, nout, natural_ordering))
#         model = MADE(nin, hiddens, nout, natural_ordering=natural_ordering)
#
#         # run backpropagation for each dimension to compute what other
#         # dimensions it depends on.
#         res = []
#         for k in range(nout):
#             xtr = Variable(torch.from_numpy(x), requires_grad=True)
#             xtrhat = model(xtr)
#             loss = xtrhat[0,k]
#             loss.backward()
#
#             depends = (xtr.grad[0].numpy() != 0).astype(np.uint8)
#             depends_ix = list(np.where(depends)[0])
#             isok = k % nin not in depends_ix
#
#             res.append((len(depends_ix), k, depends_ix, isok))
#
#         # pretty print the dependencies
#         res.sort()
#         for nl, k, ix, isok in res:
#             print("output %2d depends on inputs: %30s : %s" % (k, ix, "OK" if isok else "NOTOK"))
#

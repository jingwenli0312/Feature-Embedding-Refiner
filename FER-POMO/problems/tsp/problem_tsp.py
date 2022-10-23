from torch.utils.data import Dataset
import torch
import os
import pickle
import numpy as np
import math
import torch.nn.functional as F
from problems.tsp.state_tsp import StateTSP
from utils import move_to
import copy


class TSP(object):
    NAME = 'tsp'

    def __init__(self, p_size, init_val_met, with_assert=False):

        self.size = p_size  # the number of nodes in tsp
        self.init_val_met = init_val_met
        self.do_assert = with_assert
        self.state = 'eval'
        print(f'TSP with {self.size} nodes.', ' Do assert:', with_assert)
        self.train()

    def eval(self, perturb=False):
        self.training = False
        self.do_perturb = perturb

    def train(self):
        self.training = True
        self.do_perturb = False

    def get_initial_solutions(self, batch_size, dataset, dist, methods):

        def get_solution(methods):
            p_size = self.size

            if methods == 'seq':
                return (torch.linspace(1, p_size, steps=p_size) % p_size).expand(batch_size, p_size).long()

            if methods == 'random':
                return torch.randperm(self.size).expand(batch_size, p_size).long()

            elif methods == 'nearest':
                for i in range(self.size):
                    dist[torch.arange(batch_size), i, i] = 1000  # avoid selecting self
                solution = []
                selected = torch.zeros(batch_size, device=dataset.device).long()
                solution.append(selected)
                dist.scatter_(-1, selected.view(-1, 1, 1).expand(batch_size, len(dataset[0]), 1), 1000)

                for i in range(len(dataset[0]) - 1):
                    next = torch.min(dist[torch.arange(batch_size), selected], 1)[1]
                    solution.append(next)
                    dist.scatter_(-1, next.view(-1, 1, 1).expand(batch_size, len(dataset[0]), 1), 1000)
                    selected = next
                return torch.stack(solution, 1)

        return get_solution(methods).clone()


    def get_costs(self, dataset, rec, multi_solu=True):
        """
        :param dataset: (batch_size, graph_size, 2) coordinates
        :param rec: (num_solu, batch_size, graph_size) permutations representing tours
        :return: (batch_size) lengths of tours
        """
        if multi_solu:
            assert (
                    torch.arange(rec.size(2), out=rec.data.new()).view(1, -1).expand_as(rec) ==
                    rec.data.sort(2)[0]
            ).all(), "Invalid tour"

            num_solu, bs, gs = rec.size()
            # [num_solu, bs, gs, 2]
            d = dataset.unsqueeze(0).repeat(num_solu, 1).gather(2, rec.long().unsqueeze(-1).expand(num_solu, bs, gs, 2))
            length = (d[:, :, 1:] - d[:, :, -1]).norm(p=2, dim=-1).sum(2) + (d[:, :, 0] - d[:, :, -1]).norm(p=2, dim=-1)

            return length  # [num_solu, bs]

        else:
            assert (
                    torch.arange(rec.size(1), out=rec.data.new()).view(1, -1).expand_as(rec) ==
                    rec.data.sort(1)[0]
            ).all(), "Invalid tour"

            # calculate obj value
            d = dataset.gather(1, rec.long().unsqueeze(-1).expand_as(dataset))  # [bs, gs, 2]
            length = (d[:, 1:] - d[:, :-1]).norm(p=2, dim=2).sum(1) + (d[:, 0] - d[:, -1]).norm(p=2, dim=1)

            return length  # [bs]


    @staticmethod
    def make_dataset(*args, **kwargs):
        return TSPDataset(*args, **kwargs)

    @staticmethod
    def make_state(*args, **kwargs):
        return StateTSP.initialize(*args, **kwargs)


class TSPDataset(Dataset):

    def __init__(self, opts=None, filename=None, size=50, num_samples=1000000, offset=0, distribution=None):
        super(TSPDataset, self).__init__()

        self.data_set = []
        if filename is not None:
            assert os.path.splitext(filename)[1] == '.pkl'

            with open(filename, 'rb') as f:
                data = pickle.load(f)
                # print(data)
                # self.data = [torch.FloatTensor(data) for _ in range(10)]
                self.data = [torch.FloatTensor(row) for row in (data[offset:offset + num_samples])]  # N, tsp_size, 2

        else:
            # Sample points randomly in [0, 1] square
            self.data = [torch.FloatTensor(size, 2).uniform_(0, 1) for i in range(num_samples)]

        self.size = len(self.data)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.data[idx]



if __name__ == '__main__':
    a = -10 ** 20
    print(a)

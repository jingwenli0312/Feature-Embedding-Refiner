from torch.utils.data import Dataset
import torch
import os
import pickle

from problems.vrp.state_cvrp import StateCVRP
# from problems.state_sdvrp import StateSDVRP
from utils.beam_search import beam_search
import numpy as np
import random

DUMMY_RAT = 0.4


class CVRP(object):
    NAME = 'cvrp'  # Capacitated Vehicle Routing Problem

    VEHICLE_CAPACITY = 1.0  # (w.l.o.g. vehicle capacity is 1, demands should be scaled)

    def __init__(self, p_size, init_val_met, with_assert=False):
        self.size = p_size

        self.MAX_LENGTH = {
            4: 8,
            20: 40,
            50: 75,
            100: 170
        }

    def eval(self, perturb=False):
        self.training = False
        self.do_perturb = perturb

    def train(self):
        self.training = True
        self.do_perturb = False

    def get_initial_solutions(self, batch_size, dataset, dist, methods):

        batch_size = dataset['loc'].size(0)

        def get_solution(methods):
            p_size = self.size
            data = torch.cat((dataset['depot'].unsqueeze(1), dataset['loc']), 1)
            demand = torch.cat((torch.zeros(batch_size, 1, device=data.device), dataset['demand']), 1)

            if methods == 'random':
                candidates = torch.zeros(batch_size, self.size + 1, device=data.device).bool()
                solution = []
                selected_node = torch.zeros(batch_size, 1, device=data.device).long()
                solution.append(selected_node)
                current_load = torch.zeros(batch_size, 1, device=data.device)
                visited = torch.zeros(batch_size, self.size + 1, dtype=torch.uint8, device=data.device)
                visited.scatter_(1, selected_node, 1)
                candidates.scatter_(1, selected_node, 1)

                while visited.all() != 1:
                    dists = torch.rand(batch_size, p_size + 1, device=data.device)
                    
                    # if selected_node != 0:  
                    # dists.scatter_(1, selected_node, 1e5)
                    dists[candidates] = 1e5
                    selected_demand = demand.gather(1, selected_node)
                    current_load += selected_demand
                    dists[current_load + demand > 1.] = 1e5
                    next_selected_node = dists.min(-1)[1].view(-1, 1)
                    current_load[next_selected_node == 0] = 0
                    # print('dist', dists)
                    # print('visit', visited)
                    # print('candidate', candidates)
                    solution.append(next_selected_node)
                    visited.scatter_(1, next_selected_node, 1)
                    candidates.scatter_(1, next_selected_node, 1)
                    candidates[torch.arange(batch_size), 0] = (next_selected_node == 0).squeeze(-1) & (
                            (visited[:, 1:] == 0).int().sum(-1) > 0)

                   #  print('select', next_selected_node)
                    selected_node = next_selected_node

                pi = torch.stack(solution, -1).squeeze(1)
                zero_tensor = torch.zeros((pi.size(0),
                                           self.MAX_LENGTH[self.size] - pi.size(-1)), dtype=torch.int64,
                                          device=demand.device)
                pi = torch.cat((pi, zero_tensor), -1)

                return pi

            if methods == 'nearest':

                candidates = torch.zeros(batch_size, self.size + 1, device=data.device).bool()
                solution = []
                selected_node = torch.zeros(batch_size, 1, device=data.device).long()
                solution.append(selected_node)
                current_load = torch.zeros(batch_size, 1, device=data.device)
                visited = torch.zeros(batch_size, self.size + 1, dtype=torch.uint8, device=data.device)
                visited.scatter_(1, selected_node, 1)

                d2 = data.clone()

                while visited.all() != 1:
                    d1 = data.gather(1, selected_node.unsqueeze(-1).expand(batch_size, self.size + 1, 2))
                    dists = (d1 - d2).norm(p=2, dim=2)  # bs, gs+1

                    dists.scatter_(1, selected_node, 1e5)
                    dists[candidates] = 1e5
                    selected_demand = demand.gather(1, selected_node)
                    current_load += selected_demand
                    dists[current_load + demand > 1.] = 1e5
                    next_selected_node = dists.min(-1)[1].view(-1, 1)
                    current_load[next_selected_node == 0] = 0

                    solution.append(next_selected_node)
                    visited.scatter_(1, next_selected_node, 1)
                    candidates.scatter_(1, next_selected_node, 1)
                    candidates[torch.arange(batch_size), 0] = (next_selected_node == 0).squeeze(-1) & (
                                (visited[:, 1:] == 0).int().sum(-1) > 0)
                    selected_node = next_selected_node

                pi = torch.stack(solution, -1).squeeze(1)
                zero_tensor = torch.zeros((pi.size(0),
                                           self.MAX_LENGTH[self.size] - pi.size(-1)), dtype=torch.int64,
                                          device=demand.device)
                pi = torch.cat((pi, zero_tensor), -1)
                return pi

        return get_solution(methods).clone()

    @staticmethod
    def get_costs(dataset, pi, multi_solu=False):
        if multi_solu:
            pass
        else:
            batch_size, graph_size = dataset['demand'].size()
            # Check that tours are valid, i.e. contain 0 to n -1
            sorted_pi = pi.data.sort(1)[0]
    

            # Sorting it should give all zeros at front and then 1...n
            assert (
                           torch.arange(1, graph_size + 1, out=pi.data.new()).view(1, -1).expand(batch_size,
                                                                                                 graph_size) ==
                           sorted_pi[:, -graph_size:]
                   ).all() and (sorted_pi[:, :-graph_size] == 0).all(), "Invalid tour"

            # Visiting depot resets capacity so we add demand = -capacity (we make sure it does not become negative)
            demand_with_depot = torch.cat(
                (
                    torch.full_like(dataset['demand'][:, :1], -CVRP.VEHICLE_CAPACITY),
                    dataset['demand']
                ),
                1
            )
            d = demand_with_depot.gather(1, pi)

            used_cap = torch.zeros_like(dataset['demand'][:, 0])
            for i in range(pi.size(1)):
                used_cap += d[:, i]  # This will reset/make capacity negative if i == 0, e.g. depot visited
                # Cannot use less than 0
                used_cap[used_cap < 0] = 0
                assert (used_cap <= CVRP.VEHICLE_CAPACITY + 1e-5).all(), "Used more than capacity"

            # Gather dataset in order of tour
            loc_with_depot = torch.cat((dataset['depot'][:, None, :], dataset['loc']), 1)  # (bs, gs, 2)
            d = loc_with_depot.gather(1, pi[..., None].expand(*pi.size(), loc_with_depot.size(-1)))

            # Length is distance (L2-norm of difference) of each next location to its prev and of first and last to depot
            return (
                    (d[:, 1:] - d[:, :-1]).norm(p=2, dim=2).sum(1)
                    + (d[:, 0] - dataset['depot']).norm(p=2, dim=1)  # Depot to first
                    + (d[:, -1] - dataset['depot']).norm(p=2, dim=1)  # Last to depot, will be 0 if depot is last
            )

    @staticmethod
    def make_dataset(*args, **kwargs):
        return VRPDataset(*args, **kwargs)

    @staticmethod
    def make_state(*args, **kwargs):
        return StateCVRP.initialize(*args, **kwargs)

    @staticmethod
    def beam_search(input, beam_size, expand_size=None,
                    compress_mask=False, model=None, max_calc_batch_size=4096):

        assert model is not None, "Provide model"

        fixed = model.precompute_fixed(input)

        def propose_expansions(beam):
            return model.propose_expansions(
                beam, fixed, expand_size, normalize=True, max_calc_batch_size=max_calc_batch_size
            )

        state = CVRP.make_state(
            input, visited_dtype=torch.int64 if compress_mask else torch.uint8
        )

        return beam_search(state, beam_size, propose_expansions)


class SDVRP(object):
    NAME = 'sdvrp'  # Split Delivery Vehicle Routing Problem

    VEHICLE_CAPACITY = 1.0  # (w.l.o.g. vehicle capacity is 1, demands should be scaled)

    @staticmethod
    def get_costs(dataset, pi):
        pi = torch.tensor([[0, 8, 17, 4, 3, 15, 0, 16, 9, 11, 20, 6, 1, 13, 18, 0, 12, 19, 5, 14, 7, 10, 0,2,0]])
        batch_size, graph_size = dataset['demand'].size()

        # Each node can be visited multiple times, but we always deliver as much demand as possible
        # We check that at the end all demand has been satisfied
        demands = torch.cat(
            (
                torch.full_like(dataset['demand'][:, :1], -SDVRP.VEHICLE_CAPACITY),
                dataset['demand']
            ),
            1
        )
        rng = torch.arange(batch_size, out=demands.data.new().long())
        used_cap = torch.zeros_like(dataset['demand'][:, 0])
        a_prev = None
        for a in pi.transpose(0, 1):
            assert a_prev is None or (demands[((a_prev == 0) & (a == 0)), :] == 0).all(), \
                "Cannot visit depot twice if any nonzero demand"
            d = torch.min(demands[rng, a], SDVRP.VEHICLE_CAPACITY - used_cap)
            demands[rng, a] -= d
            used_cap += d
            used_cap[a == 0] = 0
            a_prev = a
        assert (demands == 0).all(), "All demand must be satisfied"

        # Gather dataset in order of tour
        loc_with_depot = torch.cat((dataset['depot'][:, None, :], dataset['loc']), 1)
        d = loc_with_depot.gather(1, pi[..., None].expand(*pi.size(), loc_with_depot.size(-1)))

        # Length is distance (L2-norm of difference) of each next location to its prev and of first and last to depot
        print('cost is', (
                       (d[:, 1:] - d[:, :-1]).norm(p=2, dim=2).sum(1)
                       + (d[:, 0] - dataset['depot']).norm(p=2, dim=1)  # Depot to first
                       + (d[:, -1] - dataset['depot']).norm(p=2, dim=1)  # Last to depot, will be 0 if depot is last
               ))
        return (
                       (d[:, 1:] - d[:, :-1]).norm(p=2, dim=2).sum(1)
                       + (d[:, 0] - dataset['depot']).norm(p=2, dim=1)  # Depot to first
                       + (d[:, -1] - dataset['depot']).norm(p=2, dim=1)  # Last to depot, will be 0 if depot is last
               ), None

    @staticmethod
    def make_dataset(*args, **kwargs):
        return VRPDataset(*args, **kwargs)

    @staticmethod
    def make_state(*args, **kwargs):
        return StateSDVRP.initialize(*args, **kwargs)

    @staticmethod
    def beam_search(input, beam_size, expand_size=None,
                    compress_mask=False, model=None, max_calc_batch_size=4096):
        assert model is not None, "Provide model"
        assert not compress_mask, "SDVRP does not support compression of the mask"

        fixed = model.precompute_fixed(input)

        def propose_expansions(beam):
            return model.propose_expansions(
                beam, fixed, expand_size, normalize=True, max_calc_batch_size=max_calc_batch_size
            )

        state = SDVRP.make_state(input)

        return beam_search(state, beam_size, propose_expansions)


def make_instance(args):
    depot, loc, demand, capacity, *args = args
    grid_size = 1
    if len(args) > 0:
        depot_types, customer_types, grid_size = args
    return {
        'loc': torch.tensor(loc, dtype=torch.float) / grid_size,
        'demand': torch.tensor(demand, dtype=torch.float) / capacity,
        'depot': torch.tensor(depot, dtype=torch.float) / grid_size
    }


class VRPDataset(Dataset):

    def __init__(self, filename=None, size=50, num_samples=1000000, offset=0, distribution=None):
        super(VRPDataset, self).__init__()

        self.data_set = []
        if filename is not None:
            assert os.path.splitext(filename)[1] == '.pkl'

            with open(filename, 'rb') as f:
                data = pickle.load(f)

            self.data = [make_instance(args) for args in data[offset:offset + num_samples]]

        else:

            # From VRP with RL paper https://arxiv.org/abs/1802.04240
            CAPACITIES = {
                4: 10,
                10: 20.,
                20: 30.,
                50: 40.,
                100: 50.
            }

            self.data = [
                {
                    'loc': torch.FloatTensor(size, 2).uniform_(0, 1),
                    # Uniform 1 - 9, scaled by capacities
                    'demand': (torch.FloatTensor(size).uniform_(0, 9).int() + 1).float() / CAPACITIES[size],
                    'depot': torch.FloatTensor(2).uniform_(0, 1)
                }
                for i in range(num_samples)
            ]

        self.size = len(self.data)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.data[idx]

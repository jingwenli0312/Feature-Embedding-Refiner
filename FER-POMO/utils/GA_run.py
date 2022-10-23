import os
import io
import json
import random
import argparse
import numpy as np
import pickle
import time
import statistics

random.seed(0)

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str, default="./data/Input_Data.json", required=False,
                        help="Enter the input Json file name")
    parser.add_argument('--pop_size', type=int, default=50, required=False,
                        help="Enter the population size")
    parser.add_argument('--mute_prob', type=float, default=0.5, required=False,
                        help="Mutation Probabilty")
    parser.add_argument('--iterations', type=int, default=5000, required=False,
                        help="Number of iterations to run")

    return parser.parse_args()


def load_instance(json_file):
    """
    Inputs: path to json file
    Outputs: json file object if it exists, or else returns NoneType
    """
    if os.path.exists(path=json_file):
        with io.open(json_file, 'rt', newline='') as file_object:
            return json.load(file_object)
    return None


def initialize_population(n_customers, n_population):
    population = []
    while len(population) < n_population:
        chromosome = random.sample([i for i in range(1, n_customers+1)], n_customers)
        if chromosome not in population:
            population.append(chromosome)
    return population


def evaluate(chromosome, distance_matrix, demand, cap_vehicle, return_subroute=False):
    total_distance = 0
    cur_load = 0
    route = []
    sub_route = []
    for customer in chromosome:
        cur_load += demand[customer]
        if cur_load > cap_vehicle:
            if return_subroute:
                sub_route.append([0])
                sub_route.append(route[:])
            total_distance += calculate_distance(route, distance_matrix)
            cur_load = demand[customer]
            route = [customer]
        else:
            route.append(customer)

    total_distance += calculate_distance(route, distance_matrix)
    if return_subroute:
        sub_route.append(route[:])
        return sub_route
    return total_distance


def calculate_distance(route, distance_matrix):
    distance = 0
    distance += distance_matrix[0][route[0]]
    distance += distance_matrix[route[-1]][0]
    for i in range(0, len(route)-1):
        distance += distance_matrix[route[i]][route[i+1]]
    return distance


def get_chromosome(population, func, *params, reverse=False, k=1):
    scores = []
    for chromosome in population:
        scores.append([func(chromosome, *params), chromosome])
    scores.sort(reverse=reverse)
    if k == 1:
        return scores[0]
    elif k > 1:
        return scores[:k]
    else:
        raise Exception("invalid k")


def ordered_crossover(chromo1, chromo2):
    # Modifying this to suit our needs
    #  If the sequence does not contain 0, this throws error
    #  So we will modify inputs here itself and then
    #       modify the outputs too

    ind1 = [x-1 for x in chromo1]
    ind2 = [x-1 for x in chromo2]
    size = min(len(ind1), len(ind2))
    a, b = random.sample(range(size), 2)
    if a > b:
        a, b = b, a

    holes1, holes2 = [True] * size, [True] * size
    for i in range(size):
        if i < a or i > b:
            holes1[ind2[i]] = False
            holes2[ind1[i]] = False

    # We must keep the original values somewhere before scrambling everything
    temp1, temp2 = ind1, ind2
    k1, k2 = b + 1, b + 1
    for i in range(size):
        if not holes1[temp1[(i + b + 1) % size]]:
            ind1[k1 % size] = temp1[(i + b + 1) % size]
            k1 += 1

        if not holes2[temp2[(i + b + 1) % size]]:
            ind2[k2 % size] = temp2[(i + b + 1) % size]
            k2 += 1

    # Swap the content between a and b (included)
    for i in range(a, b + 1):
        ind1[i], ind2[i] = ind2[i], ind1[i]

    # Finally adding 1 again to reclaim original input
    ind1 = [x+1 for x in ind1]
    ind2 = [x+1 for x in ind2]
    return ind1, ind2


def mutate(chromosome, probability):
    if random.random() < probability:
        index1, index2 = random.sample(range(len(chromosome)), 2)
        chromosome[index1], chromosome[index2] = chromosome[index2], chromosome[index1]
        index1, index2 = sorted(random.sample(range(len(chromosome)), 2))
        mutated = chromosome[:index1] + list(reversed(chromosome[index1:index2+1]))
        if index2 < len(chromosome) - 1:
            mutated += chromosome[index2+1:]
        return mutated
    return chromosome


def replace(population, chromo_in, chromo_out):
    population[population.index(chromo_out)] = chromo_in


def check_validity(chromosome, length):
    for i in range(1, length+1):
        if i not in chromosome:
            raise Exception("invalid chromosome")

def make_instance(args):
    depot, loc, *args = args
    grid_size = 1

    if len(args) > 0:
        demand, capacity = args
    return {
        'loc': np.array(loc, dtype=np.float) / grid_size,
        'depot': np.array(depot, dtype=np.float) / grid_size,
        'demand': np.array(demand, dtype=np.float) / grid_size
    }


def cal_dist_matrix(data):
    N_data = data.shape[0]
    dists = np.zeros((N_data, N_data), dtype=np.float)
    d1 = -2 * np.dot(data, data.T)
    d2 = np.sum(np.square(data), axis=1)
    d3 = np.sum(np.square(data), axis=1).reshape(1, -1).T
    dists = d1 + d2 + d3
    dists[dists < 0] = 0
    return np.sqrt(dists)



if __name__ == '__main__':
    n_customers = 20
    filename = './datasets/vrp20_test_seed1234.pkl'
    num_samples = 2

    CAPACITIES = {
        20: 30.,
        50: 40.,
        100: 50.
    }

    with open(filename, 'rb') as f:
        data = pickle.load(f)
    data1 = [make_instance(args) for args in data[: num_samples]]

    result = []
    start_time = time.time()

    for j in range(num_samples):
        args = get_parser()
        demand = {}
        for i in range(1, n_customers+1):
            demand[i] = data1[j]['demand'][i - 1]

        data1[j]['depot'] = data1[j]['depot'] * 100
        data1[j]['loc'] = data1[j]['loc'] * 100
        coordinates = np.concatenate(([data1[j]['depot']], data1[j]['loc']), axis=0)
        distance_matrix = cal_dist_matrix(coordinates)
        cap_vehicle = CAPACITIES[n_customers]

        n_population = args.pop_size
        iteration = args.iterations
        cur_iter = 1
        mutate_prob = args.mute_prob

        population = initialize_population(n_customers, n_population)
        prev_score, chromosome = get_chromosome(population, evaluate, distance_matrix, demand, cap_vehicle)

        score_history = [prev_score]

        while cur_iter <= iteration:
            chromosomes = get_chromosome(population, evaluate, distance_matrix, demand, cap_vehicle, k=2)
            chromosome1 = chromosomes[0][1]
            chromosome2 = chromosomes[1][1]
            offspring1, offspring2 = ordered_crossover(chromosome1, chromosome2)
            offspring1 = mutate(offspring1, mutate_prob)
            offspring2 = mutate(offspring2, mutate_prob)
            score1 = evaluate(offspring1, distance_matrix, demand, cap_vehicle)
            score2 = evaluate(offspring2, distance_matrix, demand, cap_vehicle)
            score, chromosome = get_chromosome(population, evaluate, distance_matrix, demand, cap_vehicle, reverse=True)

            if score1 < score:
                replace(population, chromo_in=offspring1, chromo_out=chromosome)

            score, chromosome = get_chromosome(population, evaluate, distance_matrix, demand, cap_vehicle, reverse=True)

            if score2 < score:
                replace(population, chromo_in=offspring2, chromo_out=chromosome)

            score, chromosome = get_chromosome(population, evaluate, distance_matrix, demand, cap_vehicle)
            score_history.append(score)
            prev_score = score
            cur_iter += 1

        print('current cost: ', score / 100)
        result.append(score / 100)
        print('average cost until instance: ', j, statistics.mean(result))

print('final average cost: ', statistics.mean(result))
print('total computation time: ', time.time() - start_time)

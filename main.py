import tsplib95 # type: ignore
import networkx as nx # type: ignore
import matplotlib.pyplot as plt # type: ignore
import time
import random

def plot_route(instance, route):
    G = instance.get_graph()

    route_edges = [(route[i], route[i+1]) for i in range(len(route)-1)]

    position = 'coord' if instance.display_data_type == 'COORD DISPLAY' else 'display'

    pos = {city: (G.nodes[city][position][0], G.nodes[city][position][1]) for city in route}

    nx.draw_networkx_nodes(G, pos, node_size=100, node_color='lightgray')
    nx.draw_networkx_edges(G, pos, edgelist=route_edges, edge_color='black')
    nx.draw_networkx_labels(G, pos, font_size=8)

    plt.title(f'{instance.name} - Route')
    plt.show()


def update_best_route(instance, old_route, new_route):
    old_route_cost = calculate_route_cost(instance, old_route)
    new_route_cost = calculate_route_cost(instance, new_route)

    return (new_route, new_route_cost) if new_route_cost < old_route_cost else (old_route, old_route_cost)


def calculate_route_cost(instance, route):
    return sum(
        instance.get_weight(route[i], route[i + 1])
        for i in range(len(route) - 1)
    )


def three_opt(instance, route):
    for i in range(1, len(route) - 4):
        for j in range(i + 1, len(route) - 3):
            for k in range(j + 1, len(route) - 2):
                if instance.get_weight(route[i], route[i + 1]) + instance.get_weight(route[j], route[j + 1]) + instance.get_weight(route[k], route[k + 1]) > instance.get_weight(route[i], route[j]) + instance.get_weight(route[i + 1], route[j + 1]) + instance.get_weight(route[k], route[k + 1]):
                    route[i + 1], route[j] = route[j], route[i + 1]
                    route[i + 2:j + 1] = reversed(route[i + 2:j + 1])
                    route[j + 1], route[k] = route[k], route[j + 1]
                    route[j + 2:k + 1] = reversed(route[j + 2:k + 1])

    return route


def or_opt(instance, route):
    for i in range(1, len(route) - 3):
        for j in range(i + 1, len(route) - 2):
            for k in range(j + 1, len(route) - 1):
                if instance.get_weight(route[i], route[i + 1]) + instance.get_weight(route[j], route[j + 1]) + instance.get_weight(route[k], route[k + 1]) > instance.get_weight(route[i], route[j + 1]) + instance.get_weight(route[k], route[i + 1]) + instance.get_weight(route[j], route[k + 1]):
                    route[i + 1:j + 1] = reversed(route[i + 1:j + 1])

    return route


def two_opt(instance, route):
    for i in range(1, len(route) - 2):
        for j in range(i + 1, len(route) - 1):
            if instance.get_weight(route[i], route[i + 1]) + instance.get_weight(route[j], route[j + 1]) > instance.get_weight(route[i], route[j]) + instance.get_weight(route[i + 1], route[j + 1]):
                route[i + 1], route[j] = route[j], route[i + 1]
                route[i + 2:j + 1] = reversed(route[i + 2:j + 1])

    return route


def basic_variable_search_descendent(instance, route, k):
    if k == 1:
        return two_opt(instance, route)
    elif k == 2:
        return or_opt(instance, route)
    elif k == 3:
        return three_opt(instance, route)
    else:
        route


def shake(route, k):
    # Random Node Swap
    for _ in range(k):
        idx1 = random.randint(1, len(route) - 2)
        idx2 = random.randint(1, len(route) - 2)

        route[idx1], route[idx2] = route[idx2], route[idx1]

    return route


def general_variable_neighborhood_search(instance, route, k_max, l_max):
    k = 1
    while k <= k_max:
        perturbed_route = shake(route[:], k)
        l = 1
        while l <= l_max:
            optimized_route = basic_variable_search_descendent(instance, perturbed_route, k)
            # Neighborhood Change Sequential - VND
            if calculate_route_cost(instance, optimized_route) < calculate_route_cost(instance, route):
                route = optimized_route
                l = 1
            else:
                l += 1
        # Neighborhood Change Sequential - VNS
        if calculate_route_cost(instance, route) > calculate_route_cost(instance, optimized_route):
            route = optimized_route
            k = 1
        else:
            k += 1

    return route


def greedy_randomized_adaptive_search_procedure(instance, alpha):
    # Create Candidate List
    candidate_list = list(instance.get_nodes())
    initial_city = candidate_list[0]
    candidate_list.remove(initial_city)
    candidate_list.sort(key=lambda city: instance.get_weight(initial_city, city)) # Sort by distance to initial city

    # Create Restricted Candidate List - c(e) ∈ [cmin, cmin + α(cmax − cmin)]
    route = [initial_city]
    while candidate_list:
        smallest_city = candidate_list[0]
        biggest_city = candidate_list[-1]
        c_min = instance.get_weight(initial_city, smallest_city)
        c_max = instance.get_weight(initial_city, biggest_city)
        threshold_min = c_min
        threshold_max = c_min + alpha * (c_max - c_min)
        restricted_candidate_list = [city for city in candidate_list if threshold_min <= instance.get_weight(initial_city, city) <= threshold_max]
        random_city = random.choice(restricted_candidate_list)
        route.append(random_city)
        candidate_list.remove(random_city)
    route.append(initial_city)

    return route


def grasp_gvns(instance):
    # GRASP Parameters
    alpha = 0.2
    max_iterations = 100
    # GVNS Parameters
    k_max = 3
    l_max = 3

    for _ in range(1, max_iterations + 1):
        grasp_route = greedy_randomized_adaptive_search_procedure(instance, alpha)
        gvns_route = general_variable_neighborhood_search(instance, grasp_route, k_max, l_max)
        if _ == 1:
            best_route = gvns_route
        best_route, route_cost = update_best_route(instance, best_route, gvns_route)

    return best_route, route_cost


def read_instance(instance_path):
    return tsplib95.load(instance_path)


if __name__ == "__main__":
    instance_path = 'instances/att48.tsp'
    instance = read_instance(instance_path)
    print(f'Instance: {instance.name} - {instance.dimension} cities')
    print(f'{instance.comment}')
    time_start = time.time()
    best_route, route_cost = grasp_gvns(instance)
    time_end = time.time()
    print(f'Best route: {best_route}')
    print(f'Execution time: {(time_end - time_start):.2f}s')
    print(f'Route cost: {route_cost}')
    plot_route(instance, best_route) if instance.display_data_type else None

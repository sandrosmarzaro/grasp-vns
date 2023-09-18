import argparse
import itertools
import tsplib95 # type: ignore
import networkx as nx # type: ignore
import matplotlib.pyplot as plt # type: ignore
import googlemaps # type: ignore
import gmplot # type: ignore
import webbrowser
import logging
import time
import random
import os
from datetime import datetime


def plot_route_in_googlemaps_directions(route_coordinates):
    gmap = gmplot.GoogleMapPlotter(route_coordinates[0]['lat'], route_coordinates[0]['lng'], 15, apikey=get_api_key())
    gmap.directions(
        (float(route_coordinates[0]['lat']), float(route_coordinates[0]['lng']),),
        (float(route_coordinates[-1]['lat']), float(route_coordinates[-1]['lng']),),
        waypoints=[(float(point['lat']), float(point['lng'])) for point in route_coordinates[1:-1]]
    )
    gmap.draw("route_map_direction.html")
    webbrowser.open("route_map_direction.html")


def plot_route_in_googlemaps(route_coordinates):
    latitudes = [coord['lat'] for coord in route_coordinates]
    longitudes = [coord['lng'] for coord in route_coordinates]
    gmap = gmplot.GoogleMapPlotter(latitudes[0], longitudes[0], 15, apikey=get_api_key())
    for lat, lng in zip(latitudes, longitudes):
        gmap.marker(lat, lng)
    gmap.plot(latitudes, longitudes, 'blue', edge_width=2)
    gmap.draw("route_map.html")
    webbrowser.open("route_map.html")


def get_latlng_from_address(address):
    gmaps = googlemaps.Client(key=get_api_key())
    geocode_result = gmaps.geocode(address)

    if not geocode_result or 'geometry' not in geocode_result[0]:
        return None
    location = geocode_result[0]['geometry']['location']
    return {'lat': location['lat'], 'lng': location['lng']}


def converter_tsplib_route_to_googlemaps_latlng(distance_matrix, route):
    route_coordinates = []
    destination_addresses = distance_matrix['destination_addresses']
    for city_index in route:
        address = destination_addresses[city_index - 1]
        latlng = get_latlng_from_address(address)
        route_coordinates.append(latlng)

    return route_coordinates


def main_maps(destinations):
    configure_logging()
    logger = logging.getLogger()
    distance_matrix = get_distance_matrix(destinations)
    instance_name = datetime.now().strftime("%Y%m%d%H%M%S")
    instance = converter_distance_matrix_to_tsplib_instance(distance_matrix, instance_name)
    with open(f'instances/maps/{instance_name}.tsp', 'w+') as file:
        instance.write(file)
    time_start = time.time()
    best_route, route_cost = grasp_gvns(instance)
    time_end = time.time()
    logger.info(f'Best route: {best_route}')
    logger.info(f'Execution time: {(time_end - time_start):.2f}s')
    logger.info(f'Route cost: {route_cost}')
    logger.handlers[0].close()
    route_coordinates = converter_tsplib_route_to_googlemaps_latlng(distance_matrix, best_route)
    plot_route_in_googlemaps(route_coordinates)
    plot_route_in_googlemaps_directions(route_coordinates)


def converter_distance_matrix_to_tsplib_instance(distance_matrix, name):
    dimension = len(distance_matrix['destination_addresses'])
    edge_weights_matrix = [[0 for _ in range(dimension)] for _ in range(dimension)]
    for i, j in itertools.product(range(dimension), range(dimension)):
        edge_weights_matrix[i][j] = distance_matrix['rows'][i]['elements'][j]['distance']['value']

    return tsplib95.models.StandardProblem(
        name=name,
        comment='Instance generated from Google Maps API',
        type='TSP',
        dimension=dimension,
        nodes=list(range(1, dimension + 1)),
        edge_weight_type='EXPLICIT',
        edge_weight_format='FULL_MATRIX',
        edge_weights=edge_weights_matrix
    )


def get_distance_matrix(destinations):
    gmaps = googlemaps.Client(key=get_api_key())
    return gmaps.distance_matrix(destinations, destinations, mode='driving', units='metric')


def get_api_key():
    return os.getenv('GOOGLE_MAPS_API_KEY')


def plot_route_in_networkx(instance, route):
    G = instance.get_graph()

    route_edges = [(route[i], route[i+1]) for i in range(len(route)-1)]

    position = 'coord' if instance.display_data_type == 'COORD_DISPLAY' else 'display'

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


def configure_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[
        logging.FileHandler('log_file.txt', mode='w'),
        logging.StreamHandler()
        ]
    )


def main_tsplib(instance_name):
    configure_logging()
    logger = logging.getLogger()
    instance = read_instance(f'instances/tsp/{instance_name}.tsp')
    logger.info(f'Instance: {instance.name} - {instance.dimension} cities')
    logger.info(f'{instance.comment}')
    time_start = time.time()
    best_route, route_cost = grasp_gvns(instance)
    time_end = time.time()
    logger.info(f'Best route: {best_route}')
    logger.info(f'Execution time: {(time_end - time_start):.2f}s')
    logger.info(f'Route cost: {route_cost}')
    logger.handlers[0].close()
    plot_route_in_networkx(instance, best_route) if instance.display_data_type else None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='GRASP-GVNS for TSP')
    parser.add_argument('--tsplib', type=str, help='Name of TSP instance file')
    parser.add_argument('--maps', type=str, nargs='+', help='Destination addresses')
    args = parser.parse_args()

    if args.tsplib:
        main_tsplib(args.tsplib)
    elif args.maps:
        destinations = args.maps
        main_maps(destinations)
    else:
        print("Please provide either '--tsplib' or '--maps' argument.")

import itertools
import tsplib95
import networkx as nx
import matplotlib.pyplot as plt
import googlemaps
import gmplot
import logging
import time
import random
import os
from datetime import datetime
import re
import sys
import json
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QHBoxLayout, QVBoxLayout, QWidget, QLabel, QLineEdit, QListWidget, QFileDialog, QMessageBox, QSizePolicy, QToolBar, QTextEdit, QSplitter, QSpinBox, QComboBox
from PyQt5.QtGui import QFont, QIntValidator
from PyQt5.QtCore import Qt, QUrl
from PyQt5.QtWidgets import QAction
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from PyQt5.QtWebEngineWidgets import QWebEngineView


def add_waypoint_markers(file_path, best_route):
    with open(file_path, 'r') as file:
        content = file.read()

    origin_match = re.search(r"origin: new google.maps.LatLng\(([\-\d.]+), ([\-\d.]+)\)", content)
    origin_lat, origin_lng = origin_match.groups()
    waypoints_match = re.search(r"waypoints: \[(.*?)\]", content, re.DOTALL)
    waypoints_str = waypoints_match[1]
    waypoints = re.findall(r"new google.maps.LatLng\(([\-\d.]+), ([\-\d.]+)\)", waypoints_str)
    waypoints.insert(0, (origin_lat, origin_lng))
    marker_code = "".join(
        f"""
        new google.maps.Marker({{
            position: new google.maps.LatLng({lat}, {lng}),
            label: "{idx}",
            map: map
        }});
        """
        for idx, (lat, lng) in zip(best_route, waypoints)
    )
    pattern = r"(new google.maps.DirectionsRenderer\({.*?}\).setDirections\(response\);)"
    replacement = r"\1" + marker_code
    new_content = re.sub(pattern, replacement, content, flags=re.DOTALL)

    with open(file_path, 'w') as file:
        file.write(new_content)


def suppress_markers_in_html(file_path):
    with open(file_path, 'r') as file:
        content = file.read()

    pattern = r"(new google.maps.DirectionsRenderer\({)"
    replacement = r"\1\n                suppressMarkers: true,"
    new_content = re.sub(pattern, replacement, content)

    with open(file_path, 'w') as file:
        file.write(new_content)


def plot_route_in_googlemaps_directions(route_coordinates, best_route):
    gmap = gmplot.GoogleMapPlotter(route_coordinates[0]['lat'], route_coordinates[0]['lng'], 15, apikey=get_api_key())
    gmap.directions(
        (float(route_coordinates[0]['lat']), float(route_coordinates[0]['lng']),),
        (float(route_coordinates[-1]['lat']), float(route_coordinates[-1]['lng']),),
        waypoints=[(float(point['lat']), float(point['lng'])) for point in route_coordinates[1:-1]]
    )
    gmap.draw("route_map_direction.html")
    suppress_markers_in_html("route_map_direction.html")
    add_waypoint_markers("route_map_direction.html", best_route)


def plot_route_in_googlemaps(route_coordinates):
    latitudes = [coord['lat'] for coord in route_coordinates]
    longitudes = [coord['lng'] for coord in route_coordinates]
    gmap = gmplot.GoogleMapPlotter(latitudes[0], longitudes[0], 15, apikey=get_api_key())
    for lat, lng in zip(latitudes, longitudes):
        gmap.marker(lat, lng)
    gmap.plot(latitudes, longitudes, 'blue', edge_width=2)
    gmap.draw("route_map.html")


def get_coordinates(destination):
    gmaps = googlemaps.Client(key=get_api_key())

    if re.match(r"^-?\d+\.\d+,-?\d+\.\d+$", destination):
        lat, lng = destination.split(',')
        return {'lat': float(lat), 'lng': float(lng), 'type': 'coordinates'}

    if result := gmaps.geocode(destination):
        location = result[0]['geometry']['location']
        return {'lat': location['lat'], 'lng': location['lng'], 'type': 'address'}

    if re.match(r"^ChIJ[a-zA-Z0-9]{27}$", destination):
        if result := gmaps.place(destination):
            location = result['result']['geometry']['location']
            return {'lat': location['lat'], 'lng': location['lng'], 'type': 'place_id'}

    return None


def main_maps(destinations, time_limit, iterations_limit, alpha, neighborhoods):
    configure_logging()
    logger = logging.getLogger('app')
    distance_matrix = get_distance_matrix(destinations)
    instance_name = datetime.now().strftime("%Y%m%d%H%M%S")
    logger.info(f'Instance: {instance_name}')
    addresses = []
    for idx, dest in enumerate(destinations):
        coord_info = get_coordinates(dest)
        if coord_info['type'] in ['coordinates', 'place_id']:
            addresses.append(distance_matrix['destination_addresses'][idx])
        else:
            addresses.append(dest)
    comment = "\n".join(f"{idx}: {address}" for idx, address in enumerate(addresses))
    logger.info(f'Comment: {comment}')
    instance = converter_distance_matrix_to_tsplib_instance(addresses, distance_matrix, instance_name)
    with open(f'instances/maps/{instance_name}.tsp', 'w+') as file:
        instance.write(file)
    time_start = time.time()
    if alpha:
        best_route, route_cost = reactive_grasp_gvns(instance, time_limit, iterations_limit, neighborhoods)
    else:
        best_route, route_cost = grasp_gvns(instance, time_limit, iterations_limit, alpha, neighborhoods)
    time_end = time.time()
    logger.info(f'Best Route: {best_route}')
    logger.info(f'Execution Time: {(time_end - time_start):.2f}s')
    logger.info(f'Route Cost: {route_cost}m')
    logger.handlers[0].close()
    route_coordinates = [get_coordinates(address) for address in destinations]
    ordered_route_coordinates = [route_coordinates[i] for i in best_route]
    plot_route_in_googlemaps(ordered_route_coordinates)
    plot_route_in_googlemaps_directions(ordered_route_coordinates, best_route)


def converter_distance_matrix_to_tsplib_instance(addresses, distance_matrix, name):
    dimension = len(distance_matrix['destination_addresses'])
    edge_weights_matrix = [[0 for _ in range(dimension)] for _ in range(dimension)]
    for i, j in itertools.product(range(dimension), range(dimension)):
        edge_weights_matrix[i][j] = distance_matrix['rows'][i]['elements'][j]['distance']['value']

    return tsplib95.models.StandardProblem(
        name=name,
        comment=f'Instance generated from Google Maps API of addresses:\n{addresses}',
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
    return os.getenv('GOOGLE_MAPS_API_KEY', None)


def plot_route_in_networkx(instance, route):
    G = instance.get_graph()
    fig = plt.Figure(figsize=(6, 6))
    ax = fig.add_subplot(111)

    route_edges = [(route[i], route[i+1]) for i in range(len(route)-1)]
    position = 'coord' if instance.display_data_type == 'COORD_DISPLAY' else 'display'
    pos = {city: (G.nodes[city][position][0], G.nodes[city][position][1]) for city in route}

    nx.draw_networkx_nodes(G, pos, node_size=100, node_color='lightgray', ax=ax)
    nx.draw_networkx_edges(G, pos, edgelist=route_edges, edge_color='black', ax=ax)
    nx.draw_networkx_labels(G, pos, font_size=8, ax=ax)

    ax.set_title(f'{instance.name} - Route')
    return fig


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
    n = len(route)
    best_delta = float('inf')

    for i in range(n - 4):
        for j in range(i + 2, n - 2):
            for k in range(j + 2, n - (2 if (i != 0 or j != 1) else 1)):

                delta1 = instance.get_weight(route[i], route[j]) + instance.get_weight(route[i+1], route[k]) + \
                        instance.get_weight(route[j+1], route[k+1]) - instance.get_weight(route[i], route[i+1]) - \
                        instance.get_weight(route[j], route[j+1]) - instance.get_weight(route[k], route[k+1])
                delta2 = instance.get_weight(route[i], route[j]) + instance.get_weight(route[i+1], route[k+1]) + \
                        instance.get_weight(route[j+1], route[k]) - instance.get_weight(route[i], route[i+1]) - \
                        instance.get_weight(route[j], route[j+1]) - instance.get_weight(route[k], route[k+1])
                delta3 = instance.get_weight(route[i], route[k]) + instance.get_weight(route[i+1], route[j]) + \
                        instance.get_weight(route[j+1], route[k+1]) - instance.get_weight(route[i], route[i+1]) - \
                        instance.get_weight(route[j], route[j+1]) - instance.get_weight(route[k], route[k+1])
                delta4 = instance.get_weight(route[i], route[j+1]) + instance.get_weight(route[k], route[i+1]) + \
                        instance.get_weight(route[j], route[k+1]) - instance.get_weight(route[i], route[i+1]) - \
                        instance.get_weight(route[j], route[j+1]) - instance.get_weight(route[k], route[k+1])
                delta5 = instance.get_weight(route[i], route[k]) + instance.get_weight(route[j+1], route[i+1]) + \
                        instance.get_weight(route[j], route[k+1]) - instance.get_weight(route[i], route[i+1]) - \
                        instance.get_weight(route[j], route[j+1]) - instance.get_weight(route[k], route[k+1])
                delta6 = instance.get_weight(route[i], route[j+1]) + instance.get_weight(route[k], route[j]) + \
                        instance.get_weight(route[i+1], route[k+1]) - instance.get_weight(route[i], route[i+1]) - \
                        instance.get_weight(route[j], route[j+1]) - instance.get_weight(route[k], route[k+1])
                delta7 = instance.get_weight(route[i], route[k+1]) + instance.get_weight(route[j+1], route[j]) + \
                        instance.get_weight(route[k], route[i+1]) - instance.get_weight(route[i], route[i+1]) - \
                        instance.get_weight(route[j], route[j+1]) - instance.get_weight(route[k], route[k+1])

                deltas = [(delta1, 1), (delta2, 2), (delta3, 3), (delta4, 4), (delta5, 5), (delta6, 6), (delta7, 7)]
                best_variant = min(deltas, key=lambda x: x[0])

                if best_variant[0] < best_delta:
                    best_delta = best_variant[0]
                    a, b, c, d = i+1, j, j+1, k+1

                    if best_variant[1] == 1:
                        route[a:c] = reversed(route[a:c])
                    elif best_variant[1] == 2:
                        route[a:b], route[b:c] = reversed(route[a:b]), reversed(route[b:c])
                    elif best_variant[1] == 3:
                        route[a:b], route[c:d] = reversed(route[a:b]), reversed(route[c:d])
                    elif best_variant[1] == 4:
                        route = route[:a] + route[b:c+1] + route[a:b] + route[c+1:d] + route[d:]
                    elif best_variant[1] == 5:
                        route = route[:a] + route[b:c+1][::-1] + route[a:b] + route[c+1:d] + route[d:]
                    elif best_variant[1] == 6:
                        route = route[:a] + route[c+1:d+1] + route[b:c+1][::-1] + route[a:b] + route[d+1:]
                    elif best_variant[1] == 7:
                        route = route[:a] + route[c+1:d+1] + route[b:c+1][::-1] + route[a:b][::-1] + route[d+1:]

                    return route

    return route


def or_opt(instance, route):
    n = len(route)

    for i in range(n - 3):
        for j in range(i + 2, n):
            a, b, c = route[i], route[i+1], route[j]

            original_cost = instance.get_weight(a, b) + instance.get_weight(b, c)
            new_cost = instance.get_weight(a, c)

            if new_cost < original_cost:
                route[i+1:j] = route[i+1:j][::-1]

    return route


def two_opt(instance, route):
    for i in range(1, len(route) - 2):
        for j in range(i + 1, len(route) - 1):
            if instance.get_weight(route[i], route[i + 1]) + instance.get_weight(route[j], route[j + 1]) > \
                instance.get_weight(route[i], route[j]) + instance.get_weight(route[i + 1], route[j + 1]):
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
        restricted_candidate_list = [city for city in candidate_list if threshold_min <= \
                                        instance.get_weight(initial_city, city) <= threshold_max]
        random_city = random.choice(restricted_candidate_list)
        route.append(random_city)
        candidate_list.remove(random_city)
    route.append(initial_city)

    return route


def reactive_grasp_gvns(instance, time_limit, iterations_limit, neighborhoods):
    alphas = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    m = len(alphas)
    probabilities = [1/m for _ in alphas]
    average_solutions = [0 for _ in alphas]
    count_alphas = [0 for _ in alphas]
    best_route = None
    best_cost = float('inf')

    start_time = time.time()

    for _ in range(iterations_limit):
        selected_alpha = random.choices(alphas, probabilities)[0]
        alpha_index = alphas.index(selected_alpha)

        grasp_route = greedy_randomized_adaptive_search_procedure(instance, selected_alpha)
        gvns_route = general_variable_neighborhood_search(instance, grasp_route, neighborhoods, neighborhoods)
        route_cost = calculate_route_cost(instance, gvns_route)

        if route_cost < best_cost:
            best_cost = route_cost
            best_route = gvns_route

        average_solutions[alpha_index] = (average_solutions[alpha_index] * count_alphas[alpha_index] + route_cost) \
                                                                                / (count_alphas[alpha_index] + 1)
        count_alphas[alpha_index] += 1

        z_star = best_cost
        q_values = [z_star / (avg if avg != 0 else 1e-10) for avg in average_solutions]
        total_q = sum(q_values)
        probabilities = [q / total_q for q in q_values]

        if time_limit:
            elapsed_time = time.time() - start_time
            if elapsed_time >= time_limit:
                break

    return best_route, best_cost


def grasp_gvns(instance, time_limit, iterations_limit, alpha, neighborhoods):

    start_time = time.time()

    for _ in range(1, iterations_limit + 1):
        grasp_route = greedy_randomized_adaptive_search_procedure(instance, alpha)
        gvns_route = general_variable_neighborhood_search(instance, grasp_route, neighborhoods, neighborhoods)
        if _ == 1:
            best_route = gvns_route
        best_route, route_cost = update_best_route(instance, best_route, gvns_route)

        if time_limit:
            elapsed_time = time.time() - start_time
            if elapsed_time >= time_limit:
                break

    return best_route, route_cost


def read_instance(instance_path):
    return tsplib95.load(instance_path)


def configure_logging():
    logger = logging.getLogger('app')
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
        file_handler = logging.FileHandler('log_file.txt', mode='w')
        file_handler.setFormatter(formatter)
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        logger.addHandler(stream_handler)


def load_best_known_solutions():
    with open("best_known_solutions_symmetric_tsp.json", "r") as file:
        return json.load(file)


def main_tsplib(filepath, time_limit, iterations_limit, alpha, neighborhoods):
    configure_logging()
    logger = logging.getLogger('app')
    instance = read_instance(filepath)
    name_without_extension = os.path.splitext(instance.name)[0]
    logger.info(f'Instance: {name_without_extension}')
    logger.info(f'Comment: {instance.comment}')
    time_start = time.time()
    if alpha:
        best_route, route_cost = reactive_grasp_gvns(instance, time_limit, iterations_limit, neighborhoods)
    else:
        best_route, route_cost = grasp_gvns(instance, time_limit, iterations_limit, alpha, neighborhoods)
    time_end = time.time()
    logger.info(f'Best Route: {best_route}')
    logger.info(f'Execution Time: {(time_end - time_start):.2f}s')
    logger.info(f'Route Cost: {route_cost}')
    best_known_solutions = load_best_known_solutions()
    if name_without_extension in best_known_solutions:
        best_known = best_known_solutions[name_without_extension]
        logger.info(f'Best Known Solution: {best_known}')
    logger.handlers[0].close()

    return plot_route_in_networkx(instance, best_route) if instance.display_data_type else None


class App(QMainWindow):
    def __init__(self):
        super().__init__()
        self.title = 'GRASP-VNS for TSP'
        self.left = 10
        self.top = 10
        self.width = 1280
        self.height = 720
        self.initUI()

        self.time_limit = 0
        self.iterations_limit = 100
        self.alpha = 0
        self.neighborhoods = 3
        self.setup_connections()

    def initUI(self):
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)
        self.create_main_menu()

    def setup_connections(self):
        self.time_limit_input.textChanged.connect(self.update_time_limit)
        self.iterations_limit_input.textChanged.connect(self.update_iterations_limit)
        self.alpha_input.currentIndexChanged.connect(self.update_alpha)
        self.neighborhood_spinbox.valueChanged.connect(self.update_neighborhoods)

    def update_time_limit(self):
        try:
            self.time_limit = float(self.time_limit_input.text())
        except ValueError:
            self.time_limit = 0

    def update_iterations_limit(self):
        try:
            self.iterations_limit = int(self.iterations_limit_input.text())
        except ValueError:
            self.iterations_limit = 100

    def update_alpha(self):
        try:
            self.alpha = float(self.alpha_input.currentText())
        except ValueError:
            self.alpha = 0.5

    def update_neighborhoods(self):
        self.neighborhoods = self.neighborhood_spinbox.value()

    def create_main_menu(self):
        layout = QVBoxLayout()
        sub_layout = QHBoxLayout()

        font = QFont()
        font.setPointSize(24)

        self.tsplib_button = QPushButton('TSPLIB', self)
        self.tsplib_button.setFont(font)
        self.tsplib_button.clicked.connect(self.tsplib_button_clicked)
        sizePolicy = QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.tsplib_button.setSizePolicy(sizePolicy)
        sub_layout.addWidget(self.tsplib_button)
        sub_layout.addSpacing(10)
        self.set_font_size(self.tsplib_button, 30)

        self.google_maps_button = QPushButton('Google Maps', self)
        self.google_maps_button.setFont(font)
        self.google_maps_button.clicked.connect(self.google_maps_button_clicked)
        self.google_maps_button.setSizePolicy(sizePolicy)
        sub_layout.addWidget(self.google_maps_button)
        self.set_font_size(self.google_maps_button, 30)

        layout.addLayout(sub_layout)
        layout.addSpacing(20)

        time_limit_layout = QHBoxLayout()
        time_limit_label = QLabel("Time Limit")
        self.time_limit_input = QLineEdit("0")
        self.time_limit_validator = QIntValidator(0, 999999999, self)
        self.time_limit_input.setValidator(self.time_limit_validator)
        time_limit_layout.addWidget(time_limit_label)
        time_limit_layout.addWidget(self.time_limit_input)
        layout.addLayout(time_limit_layout)
        self.set_font_size(time_limit_label, 18)
        self.set_font_size(self.time_limit_input, 18)

        iterations_limit_layout = QHBoxLayout()
        iterations_limit_label = QLabel("Iterations Limit")
        self.iterations_limit_input = QLineEdit("100")
        self.iterations_limit_validator = QIntValidator(0, 999999999, self)
        self.iterations_limit_input.setValidator(self.iterations_limit_validator)
        iterations_limit_layout.addWidget(iterations_limit_label)
        iterations_limit_layout.addWidget(self.iterations_limit_input)
        layout.addLayout(iterations_limit_layout)
        self.set_font_size(iterations_limit_label, 18)
        self.set_font_size(self.iterations_limit_input, 18)

        alpha_layout = QHBoxLayout()
        alpha_label = QLabel("Alpha")
        self.alpha_input = QComboBox(self)
        alpha_values = ["0", "0.1", "0.2", "0.3", "0.4", "0.5", "0.6", "0.7", "0.8", "0.9", "1"]
        self.alpha_input.addItems(alpha_values)
        alpha_layout.addWidget(alpha_label)
        alpha_layout.addWidget(self.alpha_input)
        layout.addLayout(alpha_layout)
        self.set_font_size(alpha_label, 18)
        self.set_font_size(self.alpha_input, 18)

        neighborhood_layout = QHBoxLayout()
        neighborhood_label = QLabel("Number of Neighborhoods")
        self.neighborhood_spinbox = QSpinBox()
        self.neighborhood_spinbox.setRange(1, 3)
        self.neighborhood_spinbox.setValue(3)
        neighborhood_layout.addWidget(neighborhood_label)
        neighborhood_layout.addWidget(self.neighborhood_spinbox)
        layout.addLayout(neighborhood_layout)
        self.set_font_size(neighborhood_label, 18)
        self.set_font_size(self.neighborhood_spinbox, 18)

        central_widget = QWidget()
        central_layout = QHBoxLayout()
        v_box = QVBoxLayout()
        v_box.addStretch()
        v_box.addLayout(layout)
        v_box.addStretch()
        central_layout.addStretch()
        central_layout.addLayout(v_box)
        central_layout.addStretch()
        central_widget.setLayout(central_layout)

        self.setCentralWidget(central_widget)

    def set_font_size(self, widget, size):
        font = widget.font()
        font.setPointSize(size)
        widget.setFont(font)

    def tsplib_button_clicked(self):
        time_limit = float(self.time_limit_input.text())
        iterations_limit = int(self.iterations_limit_input.text())
        alpha = float(self.alpha_input.currentText())
        neighborhoods = self.neighborhood_spinbox.value()
        self.on_tsplib(time_limit, iterations_limit, alpha, neighborhoods)

    def google_maps_button_clicked(self):
        time_limit = float(self.time_limit_input.text())
        iterations_limit = int(self.iterations_limit_input.text())
        alpha = float(self.alpha_input.currentText())
        neighborhoods = self.neighborhood_spinbox.value()
        self.on_google_maps(time_limit, iterations_limit, alpha, neighborhoods)

    def on_tsplib(self, time_limit, iterations_limit, alpha, neighborhoods):
        filepath, _ = QFileDialog.getOpenFileName(self, "Select TSPLIB instance", "", "TSP files (*.tsp)")
        if filepath:
            fig = main_tsplib(filepath, time_limit, iterations_limit, alpha, neighborhoods)
            self.display_plot(fig)

    def display_plot(self, fig):
        self.central_widget = QWidget(self)
        self.setCentralWidget(self.central_widget)
        layout = QVBoxLayout(self.central_widget)

        splitter = QSplitter(Qt.Horizontal)

        self.log_display = QTextEdit(self)
        self.log_display.setReadOnly(True)
        self.log_display.setAcceptRichText(True)

        self.set_font_size(self.log_display, 14)

        with open('log_file.txt', 'r') as log_file:
            self.log_display.setHtml(self.log_and_display(log_file.read()))
        splitter.addWidget(self.log_display)

        canvas = FigureCanvas(fig)

        self._add_widget_to_splitter_and_layout(splitter, canvas, layout)
        toolbar_layout = QHBoxLayout()

        back_button = QToolBar(self)
        self.create_back_button(back_button)
        toolbar_layout.addWidget(back_button)

        load_tsplib_action = QAction("Load TSPLIB Instance", self)
        load_tsplib_action.triggered.connect(lambda: self.on_tsplib(self.time_limit, self.iterations_limit, self.alpha, self.neighborhoods))
        load_tsplib_button = QToolBar(self)
        load_tsplib_button.addAction(load_tsplib_action)
        self.set_font_size(load_tsplib_button, 18)
        toolbar_layout.addWidget(load_tsplib_button)

        toolbar_layout.addStretch(1)

        toolbar = NavigationToolbar(canvas, self)
        toolbar_layout.addWidget(toolbar)

        layout.addLayout(toolbar_layout)

    def log_and_display(self, message):
        message = re.sub(
            r'^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2} - \w+ - ',
            '',
            message,
            flags=re.MULTILINE,
        )
        message = message.replace("Instance: ", "<b>Instance</b><br>")
        message = message.replace("Comment: ", "<br><br><b>Comment</b><br>")
        message = message.replace("Best Route: ", "<br><br><b>Best Route</b><br>")
        message = message.replace("Execution Time: ", "<br><br><b>Execution Time</b><br>")
        message = message.replace("Route Cost: ", "<br><br><b>Route Cost</b><br>")
        message = message.replace("Best Known Solution: ", "<br><br><b>Best Known Solution</b><br>")
        message = message.replace("\n", "<br>")

        return message

    def create_back_button(self, toolbar):
        back_action = QAction("Back to Menu", self)
        back_action.triggered.connect(self.create_main_menu)
        toolbar.addAction(back_action)
        self.set_font_size(toolbar, 18)

    def on_google_maps(self, time_limit, iterations_limit, alpha, neighborhoods):
        self.time_limit = time_limit
        self.iterations_limit = iterations_limit
        self.alpha = alpha
        self.neighborhoods = neighborhoods

        self.central_widget = QWidget(self)
        self.setCentralWidget(self.central_widget)
        main_layout = QVBoxLayout()

        self.address_label = QLabel("Enter Address", self)
        self.address_label.setAlignment(Qt.AlignCenter)
        self.set_font_size(self.address_label, 18)
        main_layout.addWidget(self.address_label)

        h_layout = QHBoxLayout()

        h_layout.addStretch()

        self.address_input = QLineEdit(self)
        self.set_font_size(self.address_input, 18)
        h_layout.addWidget(self.address_input)

        self.add_address_button = QPushButton("Add", self)
        self.add_address_button.clicked.connect(self.add_address_to_list)
        self.set_font_size(self.add_address_button, 18)
        h_layout.addWidget(self.add_address_button)

        h_layout.addStretch()

        main_layout.addLayout(h_layout)

        self.address_list = QListWidget(self)
        self.address_list.setMaximumHeight(400)
        self.address_list.itemDoubleClicked.connect(self.remove_address_from_list)
        self.set_font_size(self.address_list, 18)
        main_layout.addWidget(self.address_list)

        buttons_layout = QHBoxLayout()
        buttons_layout.addStretch(1)

        self.back_to_menu_button = QPushButton("Back to Menu", self)
        self.back_to_menu_button.clicked.connect(self.create_main_menu)
        self.set_font_size(self.back_to_menu_button, 18)
        buttons_layout.addWidget(self.back_to_menu_button)

        buttons_layout.addSpacing(130)

        self.calculate_route_button = QPushButton("Calculate Route", self)
        self.calculate_route_button.clicked.connect(lambda: self.execute_main_maps(time_limit, iterations_limit, alpha, neighborhoods))
        self.set_font_size(self.calculate_route_button, 18)
        buttons_layout.addWidget(self.calculate_route_button)

        buttons_layout.addStretch(1)
        main_layout.addLayout(buttons_layout)
        main_layout.addStretch()
        self.central_widget.setLayout(main_layout)

    def add_address_to_list(self):
        if self.address_list.count() >= 10:
            QMessageBox.warning(self, "Address Limit", "You can only add up to 10 addresses.")
            return

        if address := self.address_input.text().strip():
            self.address_list.addItem(address)
            self.address_input.clear()

    def remove_address_from_list(self, item):
        row = self.address_list.row(item)
        self.address_list.takeItem(row)

    def execute_main_maps(self, time_limit, iterations_limit, alpha, neighborhoods):
        api_key = get_api_key()
        if not api_key:
            QMessageBox.warning(self, "API Key Missing", "Please provide a valid Google Maps API key.")
            return
        if len(self.address_list) < 2:
            QMessageBox.warning(self, "Insufficient Addresses", "Please provide at least 2 addresses.")
            return
        self.addresses = [self.address_list.item(i).text() for i in range(self.address_list.count())]
        main_maps(self.addresses, time_limit, iterations_limit, alpha, neighborhoods)
        self.display_html("route_map_direction.html")

    def display_html(self, html_path):
        self.central_widget = QWidget(self)
        self.setCentralWidget(self.central_widget)
        layout = QVBoxLayout(self.central_widget)

        splitter = QSplitter(Qt.Horizontal)

        self.log_display = QTextEdit(self)
        self.log_display.setReadOnly(True)
        self.log_display.setAcceptRichText(True)
        self.set_font_size(self.log_display, 14)
        with open('log_file.txt', 'r') as log_file:
            self.log_display.setHtml(self.log_and_display(log_file.read()))
        splitter.addWidget(self.log_display)

        web_view = QWebEngineView(self)
        web_view.load(QUrl.fromLocalFile(os.path.abspath(html_path)))
        self._add_widget_to_splitter_and_layout(splitter, web_view, layout)
        toolbar = QToolBar(self)
        self.create_back_button(toolbar)
        back_to_address_list_action = QAction("Back to Address List", self)
        back_to_address_list_action.triggered.connect(self.back_to_address_list)
        back_to_address_list_button = QToolBar(self)
        back_to_address_list_button.addAction(back_to_address_list_action)
        self.set_font_size(back_to_address_list_button, 18)
        toolbar.addWidget(back_to_address_list_button)
        layout.addWidget(toolbar)

        self.central_widget.setLayout(layout)

    def _add_widget_to_splitter_and_layout(self, splitter, arg1, layout):
        splitter.addWidget(arg1)
        splitter.setSizes([int(self.width / 3), int(2 * self.width / 3)])
        layout.addWidget(splitter)

    def back_to_address_list(self):
        self.on_google_maps(self.time_limit, self.iterations_limit, self.alpha, self.neighborhoods)
        self.address_list.clear()
        for address in self.addresses:
            self.address_list.addItem(address)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    ex = App()
    ex.show()
    sys.exit(app.exec_())

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
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QHBoxLayout, QVBoxLayout, QWidget, QLabel, QLineEdit, QListWidget, QFileDialog, QMessageBox, QSizePolicy, QToolBar, QTextEdit, QSplitter
from PyQt5.QtGui import QFont
from PyQt5.QtCore import Qt, QUrl
from PyQt5.QtWidgets import QAction
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from PyQt5.QtWebEngineWidgets import QWebEngineView


def add_waypoint_markers(file_path):
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
            map: map
        }});
        """
        for lat, lng in waypoints
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


def plot_route_in_googlemaps_directions(route_coordinates):
    gmap = gmplot.GoogleMapPlotter(route_coordinates[0]['lat'], route_coordinates[0]['lng'], 15, apikey=get_api_key())
    gmap.directions(
        (float(route_coordinates[0]['lat']), float(route_coordinates[0]['lng']),),
        (float(route_coordinates[-1]['lat']), float(route_coordinates[-1]['lng']),),
        waypoints=[(float(point['lat']), float(point['lng'])) for point in route_coordinates[1:-1]]
    )
    gmap.draw("route_map_direction.html")
    suppress_markers_in_html("route_map_direction.html")
    add_waypoint_markers("route_map_direction.html")


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


def main_maps(destinations):
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
    best_route, route_cost = grasp_gvns(instance)
    time_end = time.time()
    logger.info(f'Best Route: {best_route}')
    logger.info(f'Execution Time: {(time_end - time_start):.2f}s')
    logger.info(f'Route Cost: {route_cost}')
    logger.handlers[0].close()
    route_coordinates = [get_coordinates(address) for address in destinations]
    ordered_route_coordinates = [route_coordinates[i] for i in best_route]
    plot_route_in_googlemaps(ordered_route_coordinates)
    plot_route_in_googlemaps_directions(ordered_route_coordinates)


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
    return os.getenv('GOOGLE_MAPS_API_KEY')


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
    for i in range(1, len(route) - 4):
        for j in range(i + 1, len(route) - 3):
            for k in range(j + 1, len(route) - 2):
                if instance.get_weight(route[i], route[i + 1]) + instance.get_weight(route[j], route[j + 1]) + \
                    instance.get_weight(route[k], route[k + 1]) > instance.get_weight(route[i], route[j]) + \
                        instance.get_weight(route[i + 1], route[j + 1]) + instance.get_weight(route[k], route[k + 1]):
                    route[i + 1], route[j] = route[j], route[i + 1]
                    route[i + 2:j + 1] = reversed(route[i + 2:j + 1])
                    route[j + 1], route[k] = route[k], route[j + 1]
                    route[j + 2:k + 1] = reversed(route[j + 2:k + 1])

    return route


def or_opt(instance, route):
    for i in range(1, len(route) - 3):
        for j in range(i + 1, len(route) - 2):
            for k in range(j + 1, len(route) - 1):
                if instance.get_weight(route[i], route[i + 1]) + instance.get_weight(route[j], route[j + 1]) + \
                    instance.get_weight(route[k], route[k + 1]) > instance.get_weight(route[i], route[j + 1]) + \
                        instance.get_weight(route[k], route[i + 1]) + instance.get_weight(route[j], route[k + 1]):
                    route[i + 1:j + 1] = reversed(route[i + 1:j + 1])

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


def main_tsplib(filepath):
    configure_logging()
    logger = logging.getLogger('app')
    instance = read_instance(filepath)
    name_without_extension = os.path.splitext(instance.name)[0]
    logger.info(f'Instance: {name_without_extension}')
    logger.info(f'Comment: {instance.comment}')
    time_start = time.time()
    best_route, route_cost = grasp_gvns(instance)
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

    def initUI(self):
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)
        self.create_main_menu()

    def create_main_menu(self):
        layout = QHBoxLayout()

        font = QFont()
        font.setPointSize(24)

        self.tsplib_button = QPushButton('TSPLIB', self)
        self.tsplib_button.setFont(font)
        self.tsplib_button.clicked.connect(self.on_tsplib)
        sizePolicy = QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.tsplib_button.setSizePolicy(sizePolicy)
        layout.addWidget(self.tsplib_button)

        self.google_maps_button = QPushButton('Google Maps', self)
        self.google_maps_button.setFont(font)
        self.google_maps_button.clicked.connect(self.on_google_maps)
        self.google_maps_button.setSizePolicy(sizePolicy)
        layout.addWidget(self.google_maps_button)

        central_widget = QWidget()
        central_widget.setLayout(layout)
        self.setCentralWidget(central_widget)

    def on_tsplib(self):
        filepath, _ = QFileDialog.getOpenFileName(self, "Select TSPLIB instance", "", "TSP files (*.tsp)")
        if filepath:
            fig = main_tsplib(filepath)
            self.display_plot(fig)

    def display_plot(self, fig):
        self.central_widget = QWidget(self)
        self.setCentralWidget(self.central_widget)
        layout = QVBoxLayout(self.central_widget)

        splitter = QSplitter(Qt.Horizontal)

        self.log_display = QTextEdit(self)
        self.log_display.setReadOnly(True)
        self.log_display.setAcceptRichText(True)

        font = self.log_display.font()
        font.setPointSize(14)
        self.log_display.setFont(font)

        with open('log_file.txt', 'r') as log_file:
            self.log_display.setHtml(self.log_and_display(log_file.read()))
        splitter.addWidget(self.log_display)

        canvas = FigureCanvas(fig)

        self._add_widget_to_splitter_and_layout(splitter, canvas, layout)
        toolbar_layout = QHBoxLayout()

        back_action = QAction("Back to Menu", self)
        back_action.triggered.connect(self.create_main_menu)
        back_button = QToolBar(self)
        back_button.addAction(back_action)
        toolbar_layout.addWidget(back_button)

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

    def on_google_maps(self):
        self.central_widget = QWidget(self)
        self.setCentralWidget(self.central_widget)
        main_layout = QVBoxLayout()

        self.address_label = QLabel("Enter Address", self)
        self.address_label.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(self.address_label)

        h_layout = QHBoxLayout()

        h_layout.addStretch()

        self.address_input = QLineEdit(self)
        h_layout.addWidget(self.address_input)

        self.add_address_button = QPushButton("Add", self)
        self.add_address_button.clicked.connect(self.add_address_to_list)
        h_layout.addWidget(self.add_address_button)

        h_layout.addStretch()

        main_layout.addLayout(h_layout)

        self.address_list = QListWidget(self)
        self.address_list.setMaximumHeight(200)
        self.address_list.itemDoubleClicked.connect(self.remove_address_from_list)
        main_layout.addWidget(self.address_list)

        buttons_layout = QHBoxLayout()

        self.back_to_menu_button = QPushButton("Back to Menu", self)
        self.back_to_menu_button.clicked.connect(self.create_main_menu)
        buttons_layout.addWidget(self.back_to_menu_button)

        self.calculate_route_button = QPushButton("Calculate Route", self)
        self.calculate_route_button.clicked.connect(self.execute_main_maps)
        buttons_layout.addWidget(self.calculate_route_button)

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

    def execute_main_maps(self):
        addresses = [self.address_list.item(i).text() for i in range(self.address_list.count())]
        main_maps(addresses)
        self.display_html("route_map_direction.html")

    def display_html(self, html_path):
        self.central_widget = QWidget(self)
        self.setCentralWidget(self.central_widget)
        layout = QVBoxLayout(self.central_widget)

        splitter = QSplitter(Qt.Horizontal)

        self.log_display = QTextEdit(self)
        self.log_display.setReadOnly(True)
        self.log_display.setAcceptRichText(True)
        font = self.log_display.font()
        font.setPointSize(14)
        self.log_display.setFont(font)
        with open('log_file.txt', 'r') as log_file:
            self.log_display.setHtml(self.log_and_display(log_file.read()))
        splitter.addWidget(self.log_display)

        web_view = QWebEngineView(self)
        web_view.load(QUrl.fromLocalFile(os.path.abspath(html_path)))
        self._add_widget_to_splitter_and_layout(splitter, web_view, layout)
        toolbar = QToolBar(self)
        self.create_back_button(toolbar)
        layout.addWidget(toolbar)

        self.central_widget.setLayout(layout)

    def _add_widget_to_splitter_and_layout(self, splitter, arg1, layout):
        splitter.addWidget(arg1)
        splitter.setSizes([int(self.width / 3), int(2 * self.width / 3)])
        layout.addWidget(splitter)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    ex = App()
    ex.show()
    sys.exit(app.exec_())

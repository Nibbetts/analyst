import numpy as np
import scipy.spatial as sp
import matplotlib.pyplot as plt
from tqdm import tqdm

import cluster
import node

class Clusterizer:

    def __init__(self, category="Cluster", generic_stats=True):
        self.category = category
        #   Name of this cluster type, typically in plural form.
        #   This is allowed to be overridden per instance, since instances may
        #   be given different starting parameters.
        self.clusters = []
        #   Should contain clusters after compute_clusters is called.
        self.data_dict = {}
        #   Additional information to be included in the printed report.
        self.starred = []
        #   Names of attributes which should be marked with a star in report.
        self.generic_stats = generic_stats
        #   Whether or not the analyst should compute and include basic stats
        #   over this type of cluster.
        
    def compute_clusters(
            # Available kwargs (always given in case needed):
            #   strings, metric_str, printer_fn, metric_fn,
            #   encoder_fn, decoder_fn, nearest_neighbors_ix,
            #   second_nearest_neighbors_ix, furthest_neighbors_ix
            self, vectors, distance_matrix, show_progress=True, **kwargs):
        pass

    def _add_star(self, attribute):
        self.starred.append(attribute)


class ExtremityClusterizer(Clusterizer, object):

    def __init__(self, category="Extremities", generic_stats=False):
        super(ExtremityClusterizer, self).__init__(category, generic_stats)
        # NOTE: We don't need the generic stats for simple pairs of objects.

    def compute_clusters(
            self, vectors, distance_matrix, show_progress=True, **kwargs):
        # The use of **kwargs allows you to pull many pre-computed types
        #   of data from the analyst to save on computation/programming time:
        strings = kwargs["strings"]
        encode = kwargs["encoder_fn"]
        metric = kwargs["metric_fn"]
        furthest = kwargs["furthest_neighbors_ix"]

        # Compute Extremities:
        self.clusters = [
            clusters.Node(strings[i], strings[furthest[i]], encode, metric)
            for i in tqdm(
                range(len(vectors)),
                desc="Measuring the Reaches (Computing Extremities)",
                disable=(not show_progress))
            if (i == furthest[furthest[i]]
                and i < furthest[i])]

        # self.data_dict = {}


class NodeClusterizer(Clusterizer, object):

    def __init__(self, category="Nodes", generic_stats=False):
        super(NodeClusterizer, self).__init__(category, generic_stats)
        # NOTE: We don't need the generic stats for simple pairs of objects.
        self.computed = False
        #   Built to prevent recalculation (if you must, then make a new one).

    def compute_clusters(
            self, vectors, distance_matrix, show_progress=True, **kwargs):
        if self.computed: return

        strings = kwargs["strings"]
        encode = kwargs["encoder_fn"]
        metric = kwargs["metric_fn"]
        nearest = kwargs["nearest_neighbors_ix"]
        printer = kwargs["printer_fn"]

        # Compute Nodes:
        self.clusters = [
            clusters.Node(strings[i], strings[nearest[i]], encode, metric)
            for i in tqdm(
                range(len(strings)),
                desc="Watching the Galaxies Coelesce (Computing Nodes)",
                disable=(not show_progress))
            if (i == nearest[nearest[i]]
                and i < nearest[i])]

        # Useful data to store:
        self.string_node_dict = {}
        for node in self.clusters:
            self.string_node_dict[node[0]] = node
            self.string_node_dict[node[1]] = node

        # Now, we can add extra data to the analyst report,
        #   specific to our cluster type:
        if len(self.clusters) > 0:
            # Nodal Factor
            printer("Comparing the Cosmos (Calculating Nodal Factor)")
            self.data_dict["Nodal Factor"] = (
                len(self.clusters)*2.0/float(len(vectors)))
            self._add_star("Nodal Factor")

            # Alignment Factor
            printer("Musing over Magnetic Moments (Calculating Alignment Factor)")
            avg_align = np.mean([n.alignment for n in self.clusters], axis=0)
            avg_align /= np.linalg.norm(avg_align)
            self.data_dict["Alignment Factor"] = (
                np.mean([
                    np.abs(sp.distance.cosine(avg_align, n.alignment))
                    for n in self.clusters]))
            self._add_star("Alignment Factor")

        # Don't repeat calculations
        self.computed = True

class HubClusterizer(Clusterizer, object):

    def __init__(self, node_clusterizer, threshold=4,
            category="Hubs", generic_stats=True):
        #   Notice we can add custom parameters.

        # node_clusterizer: requires node info to work. Don't make a new
        #   node clusterizer either, feed it the same instance, or everything
        #   will be recomputed.
        # threshold: how big a hub has to be to be counted.
        #   Ex: threshold=4 means the object plus 3 more for whom it is nearest.

        super(HubClusterizer, self).__init__(category, generic_stats)
        self.node_clusterizer = node_clusterizer
        self.threshold = threshold
        self.s_to_node = None

    def compute_clusters(
            self, vectors, distance_matrix, show_progress=True, **kwargs):
        strings = kwargs["strings"]
        neighbors = kwargs["nearest_neighbors_ix"]
        nearest = kwargs["nearest_fn"]
        metric = kwargs["metric_fn"]
        encoder = kwargs["encoder_fn"]

        # Make sure Nodes are computed before Hubs:
        #   Note: the node clusterizer is built not to repeat calculation.
        self.node_clusterizer.compute_clusters(vectors, distance_matrix,
            show_progress, **kwargs)
        self.s_to_node = self.node_clusterizer.string_node_dict

        # Calculate potential hubs:
        temp_hubs = []
        for i in tqdm(range(len(vectors)),
                desc="Finding Galactic Hubs (Finding Potential Hubs)",
                disable=(not show_progress)):
            temp_hubs.append(clusters.Cluster(
                encoder, metric, nearest=nearest,
                objects=[strings[i]], nodes=[], auto=False,
                name=strings[i]))
                # Its name is the original object's decoded string.
            for index, neighbor in enumerate(neighbors):
                if neighbor == i:
                    temp_hubs[i].add_objects([strings[index]])
                # The 0th index in the hub's list of objects is also
                #   it's original object (is included in hub).

        # Find the real hubs:
        j = 0
        for h in tqdm(temp_hubs,
                desc="Erecting Centers of Commerce (Finding Hubs)",
                disable=(not show_progress)):
            if len(h) >= self.threshold:
                self.clusters.append(h)
                h.ID = j
                h.nodes = ([s_to_node[h.name]]
                    if h.name in s_to_node.keys() else [])
                h.calculate()
                j += 1


# def compute_hubs(metric_fn, encoder_fn, nearest_fn, nearest_neighbors_ix,
#                  strings, string_node_map, show_progress=True):
#     hubs = []
#     temp_hubs = []
#     for i in tqdm(range(len(strings)),
#             desc="Finding Galactic Hubs",
#             disable=(not show_progress)):
#         temp_hubs.append(clusters.Cluster(
#             encoder_fn, metric_fn, nearest=nearest_fn,
#             objects=[strings[i]], nodes=[], auto=False,
#             name=strings[i]))
#             # Its name is the original object's decoded string.
#         for index, neighbor in enumerate(nearest_neighbors_ix):
#             if neighbor == i:
#                 temp_hubs[i].add_objects([strings[index]])
#             # The 0th index in the hub's list of objects
#             #   is also it's original object (is included in hub).
#     j = 0
#     for h in tqdm(temp_hubs, desc="Erecting Centers of Commerce",
#             disable=(not show_progress)):
#         if len(h) >= 4: # obj plus 3 or more for whom it is nearest.
#             hubs.append(h)
#             h.ID = j
#             h.nodes = ([string_node_map[h.name]]
#                 if h.name in string_node_map.keys() else [])
#             h.calculate()
#             j += 1
#     return hubs

class SupernodeClusterizer(Clusterizer, object):

    def __init__(self, node_clusterizer,
            category="Supernodes", generic_stats=False):
        #   Notice we can add parameters.
        super(SupernodeClusterizer, self).__init__(category, generic_stats)
        self.node_clusterizer = node_clusterizer
        self.nodes = []

    def compute_clusters(
            self, vectors, distance_matrix, show_progress=True, **kwargs):
        printer = kwargs["printer_fn"]
        metric_str = kwargs["metric_str"]
        metric_fn = kwargs["metric_fn"]

        # Make sure Nodes are computed before Supernodes:
        self.node_clusterizer.compute_clusters(vectors, distance_matrix,
            show_progress, **kwargs)
        self.nodes = self.node_clusterizer.clusters

        # Compute distance matrix and nearest neighbors for node centroids:
        centroids = [n.centroid for n in self.nodes]
        printer("Fracturing the Empire (Computing Nodal Distance Matrix)")
        node_dist_matrix = sp.distance.squareform(
            sp.distance.pdist(
                centroids,
                metric_str if metric_str != None else metric_fn))
        printer("Establishing a Hierocracy (Computing Nearest Neighbor Nodes)")
        neighbors = np.argmax(node_dist_matrix, axis=1)
            
        # Compute the Supernodes:
        self.clusters = [
            clusters.Node(node,
                self.nodes[neighbors[i]],
                clusters.Node.get_centroid, metric_fn)
            for i, node in enumerate(tqdm(self.nodes,
                desc="Ascertaining Universe Filaments (Finding Supernodes)",
                disable=(not show_progress)))
            if (i == neighbors[neighbors[i]]
                and i < neighbors[i])]

        # Supernode-unique data:
        if len(self.clusters) > 0:
            # Island Factor
            printer("Minding the Macrocosm (Calculating Island Factor)")
            self.data_dict["Island Factor"] = (
                len(self.clusters)*4.0/float(len(vectors)))
            self._add_star("Island Factor")

            # Hierarchical Factor
            printer("Deliberating over Dominions (Calculating Hierarchical Factor)")
            self.data_dict["Hierarchical Factor"] = (
                len(self.clusters)*2.0/float(len(self.nodes)))
            self._add_star("Hierarchical Factor")


# def compute_supernodes(nodes, printer_fn, metric_str, metric_fn,
#                        show_progress=True):
#     centroids = [n.centroid for n in nodes]
#     printer_fn("Fracturing the Empire")
#     dist_matrix = sp.distance.squareform(
#         sp.distance.pdist(
#             centroids,
#             metric_str if metric_str != None else metric_fn))
#     printer_fn("Establishing a Hierocracy")
#     neighbors = np.argmax(dist_matrix, axis=1)
#     #neighbors_dist = dist_matrix[range(len(dist_matrix)), neighbors]

#     # Compute the Supernodes:
#     return [
#         clusters.Node(node,
#             nodes[neighbors[i]],
#             clusters.Node.get_centroid, metric_fn)
#         for i, node in enumerate(tqdm(nodes,
#             desc="Ascertaining Universe Filaments",
#             disable=(not show_progress)))
#         if (i == neighbors[neighbors[i]]
#             and i < neighbors[i])]

# def compute_nuclei():
#     pass

# def compute_chains():
#     pass

# def compute_NCC():
#     pass

# def compute_LNCC():
#     pass

# def compute_anti_hubs():
#     pass


# def compute_nodes(metric_fn, encoder_fn, nearest_neighbors_ix,
#                   strings, show_progress=True):
#     return [
#         clusters.Node(strings[i],
#             strings[nearest_neighbors_ix[i]],
#             encoder_fn,
#             metric_fn)
#         for i in tqdm(
#             range(len(strings)),
#             desc="Watching the Galaxies Coelesce",
#             disable=(not show_progress))
#         if (i == nearest_neighbors_ix[nearest_neighbors_ix[i]]
#             and i < nearest_neighbors_ix[i])
#     ]
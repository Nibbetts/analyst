# analyst

Toolset for studying high-dimensional embedding spaces.

## Dependencies

numpy: numerical computing  
scipy: for its distance metrics  
matplotlib: plotting and graphing  
tqdm: progress bars  
dill: pickling and loading, allows for unusual data members
tkinter? (sudo apt-get install python3-tk)

Each of these should be accessible through the pip package manager:
```bash
sudo pip install name_of_package
```

## Usage

Generally you would initialize one Analyst instance per one embedding space, and perform analyses and access tools in the toolset through that analyst. The exception is in experimentation with differing metrics; here you would use multiple analysts initialized with the same embeddings.

The Analyst is designed to be abstract; it requires you to tell it what metric to use or to define your own. Likewise it requires a list of cluster types to compute - either callables or recognized tags - and it will iterate through and compute all of them. The built-in, experimental clustering algorithms will save each other compute time where possible by using already-processed parts from others, but custom input clustering functions will be processed entirely individually. Thereafter, the Analyst will run its same analyses on each of those cluster types no matter the source.

The objects encoded in the space must be some sort of strings, or the internal conversion functions will not work. The Analyst either requires an encoder function and a decoder function as parameters, or a list of strings in place of the encoder from which to build an encoder and decoder.

## Definitions

- **node:** a pair of obj in the space whose nearest neighbors are each other.
- **supernode:** a pair of nodes whose nearest neighbor-nodes are each other.
- **extremity:** a pair of objects in the space whose furthest neighbors are each other. (implemented with the same Node class as nodes are.)
- **outlier:** an object which is a member of an extremity.
- **loner:** an object which has been rejected when forming clusters, making it a cluster unto itself, of sorts.
- **hub:** an obj that is the nearest neigbor of three or more other objects.
- **nodal factor:** ratio of words belonging to nodes; a measure of the scale or impact of relationships in the space
- **alignment factor:** normalize mean of vectors from a to b in nodes, then measure average absolute value of cosine similarity of each to that. Think magnetic moments - how lined-up nodes are in the space.
- **hierarchical factor:** ratio of nodes belonging to supernodes; a further measure of relationships in the space.
- **island factor:** ratio of objects belonging to supernodes; another measure of relationships in the space.
- **nucleus:** grouping of relatively nearby objects. Starting with nodes and all obj whose nearest are one of those, then finding average dist to center, then combining all clusters whose central nodes are closer than one of their averages.
- **chain:** different clustering where each cluster has a sole node and recursively finds all whose nearest's nearest's etc. nearest is a member of that node.
- **cluster:** Find nuclei, then add chainable objects whose nearest is closer than the avg dist of those whose nearest are in one of the nodes, (those we started with only) and whose nearest is in the cluster.
- **strong cluster:** Same as cluster, but builds nuclei and clusters requiring objects to have both first and second nearest belonging to the same grouping in order for them to be added.
- **unpartitioned clusters:** strong clusters, except we find all contingent clusters, and add all objects from those that are closer than own dispersion (avg obj dist from centroid). //MAY CHANGE - USE A DROPOFF DISTANCE DEPENDING ON MAX DIST OF OBJECTS IN NUCLEUS?
- **center:** average location of all objects in a cluster.
- **string factor:** average cluster span divided by space dispersion
- **regularity:** find average cluster pop, then find average difference btw. cluster pop and average cluster pop. Regularity is 1/(1+this)
- **repulsion:** as used here, avg distance of nodes to their nearest neighbors. While not a real measure of spatial or volumentric density, this is a metric for relationships between encoded objects, and thus could potentially be used as a density metric if inverted.
- **dispersion:** avg distance of nodes to the center of their distribution
- **focus:** averaged location of nodes in a cluster; concentration center.
- **skew:** distance from focus to centroid.
- **anti-hub:** list of objects whose furthest are all the same outlier.
- **contingent cluster:** other clusters whose centroid is closer than own //ADD STUFF!! dispersion, to which self is closer than other's dispersion.
- **contingency:** distance from a cluster to its nearest neighbor // ADD STUFF! cluster, minus its own dispersion

## Tools

Listed are various statistics the analyst can find and print in a tidy report. They are grouped by topic or type, some of which are included in the default analysis.

Listed are both generic stats done on all types of clusters, and information about the specific algorithms included with the analyst.

*NOTE:* Some of the properties expressed cannot be used as measures of the properties of generic embedding spaces since they depend directly on the number of embeddings, dimensionality, sphericality, or some similar property. These could be seen as specific information.

### General:
```python
# NOTE: These are all static methods.
Analyst.save(obj, path) # Returns True if successful.
    # General use case is for an already-processed analyst object,
    # but should work on most objects in most cases.
    # Will overwrite files with the same name.
    # Detects/automatically adds .pickle extensions.
Analyst.load(path) # returns unpickled object, or None if failed.
Analyst.unsave(path) # deletes a saved file. Rtrns True if success.
```

### Spatial:

*NOTE:* When "stats" is listed for a property, these are included: avg, min, max, range, distribution graph of.

- count
- centroid
- dist. to centroid stats
- medoid
- dispersion
- repulsion -- avg dist to nearest
- dist. to nearest stats
- broadness -- max dist to furthest
- dist. to furthest stats

### Clustering:

*NOTE:* When "stats" is listed for a property of a cluster type, these are included for that property: avg, min, max, range, distribution graph of.

#### Extremities: (Mutual Furthest-Neighbor Pairs)

- num extremities -- probably depends strongly on dimensionality, but is related to the sphericality of the distribution in the space.
- extremity length stats

#### Nodes: (Mutual Nearest-Neighbor Pairs)

- num nodes
- nodal factor
- node length stats
- alignment factor

#### Hubs: (Common Nearest-Neighbor Groups)

- num hubs
- hub num stats

#### Supernodes: (Hierarchical Node Pairs)

- num supernodes
- hierarchical factor, burst factor
- island factor
- supernode length stats

#### Nuclei: (Multi-Nodal Proximity Groups)

- num nuclei
- nucleus factor -- num nuclei / num objects in space
- ratio in nuclei versus not
- nucleus population stats
- nuclei string factor, nucleus span stats
- nucleus regularity
- nucleus dispersion factor -- avg. nucleus disp. / space disp, nucleus dispersion stats
- node count stats
- nucleus repulsion factor -- avg. nucleus repulsion divided by overall space repulsion, nucleus repulsion stats
- nucleus skew factor, nucleus skew stats

#### Chains: (Nearest-Neighbor-Path Partitions)

- chain population stats
- chain string factor -- avg. chain span / space dispersion, chain span stats
- chain dispersion factor -- avg. chain disp. / space disp, chain dispersion stats
- chain repulsion factor -- avg. chain repulsion / overall space repulsion, chain repulsion stats
- chain skew factor, chain skew stats

*NOTE:* num chains is equal to num nodes  
*NOTE:* all objects in the space belong to a chain  

#### Clusters: (NODAL Conglomerate CLUSTERS)

- num clusters
- cluster factor -- num clusters divided by num objects
- string factor -- avg. cluster span / space dispersion
- regularity -- related to span, cluster span stats
- ratio clustered versus loners
- avg cluster population, cluster population stats
- cluster dispersion factor -- avg. cluster disp. / space disp, cluster dispersion stats
- avg num nodes per cluster, node count stats
- cluster repulsion factor -- avg cluster repulsion / overall space repulsion, cluster repulsion stats
- cluster skew factor -- avg. cluster skew / space dispersion, cluster skew stats

#### Strong Clusters: (Dispersion and Dual LIMITED NODAL Conglomerate CLUSTERS)  

Same info as for clusters.

#### Anti-Hubs: (Common Futhest-Neighbor Groups)  

More or less the same information as for clusters, but it will NOT mean the same things. Also note that these clusters do NOT include the word that is their farthest neighbor.

### Analogical:
```python
run_analogies() # !!!CANT--UNKNOWN OUTPUT??!!!
member_of(object) # displays cluster this object is a member of.
cluster([list of objects])
    # make new cluster composed solely of the given objects.
seeded_cluster([list of objects])
    # make new cluster composed of listed objects,
    # plus all nearby objects likely to be clustered with these,
    # if these were treated as being together.
inspect_clustering([list of objects])
    # analysis on given objects,
    # prints:
    # - number of unique clusters these words are found across
    # - average ward dissimilarity of involved clusters
    # returns:
    # - list of tuples containing: (object, cluster_index)
circular_walk_graph(obj1, obj2)
    # takes a walk around the space twice, in the direction from
    # obj1 to obj2, first finding those we pass closest to,
    # then graphing them as we go around the second time.
    # most useful in a normalized space, like word2vec.
```

### Comparative:
```python
compare_difference(analyst2, simple_diff=False)
    # prints a full report with three numbers for each property:
    # val_for_A, val_for_B, A_B_compared.
    # The third number is a representation of how different A, B are,
    #     either a simple difference or weighted by their scales,
    #     depending on the value of simple_diff.
Analyst.compare([list_of_analysts])
    # @staticmethod which lists
    # side by side the values for each analyst in the list.
```

### Specifics / Inspection:
```python
rank_outliers()
    # Rank by number of objects for which this one is furthest
    # neighbor. Resulting list contains exactly all objects which are
    # members of an extremity.
rank_clusters() # Rank by size; lists the indeces of the clusters.
rank_hubs() # Rank by number for which this one is nearest neighbor.
graph(graph_key, bins) # produce graph given key printed in report.
centroid # accessible vector; can be used externally.
clusters # accessible variable; a list of the clusters.
    # Further info is available in the internal vars of each cluster.
strong clusters # ''
nodes # ''
supernodes # ''
nuclei # ''
chains # ''
extremities # ''
anti-clusters # dict keying outlier objects to anti-clusters.
as_string(obj) # generic type converter for individual objects
as_index(obj) # ''
as_vector(obj) # ''
```

### Simulation:
```python
Analyst.simulate_space()
    # @staticmethod which generates an entire
    # fake embedding space with the specified properties,
    # and returns it wrapped in a new analyst.
    # NOTE: Includes cluster generation. No need to add to it.
Analyst.simulate_cluster()
    # @staticmethod which generates generic
    # test clusters to compare with, or to examine properties.
    # Available cluster types listed in function comments.
TestSet2D
    # a class which can be treated like a small 2D embedding
    # space and fed into an analyst for testing. Has encoder and
    # decoder functions to be fed in also.
```

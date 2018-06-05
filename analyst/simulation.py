#--------------------------------------------------------------------------#
# Simulation:                                                              #
#--------------------------------------------------------------------------#

def simulate_space(parameters):
    '''
    parameters:
        A list of lists, each of which follows the format:
            ["space_type", "cluster_type", num_clusters, space_radius,
                space_dims (cluster_min_pop, cluster_max_pop),
                (cluster_min_radius, cluster_max_radius),
                cluster_occupied_dims, cluster_total_dims, randomize_dims,
                noise, normalize]

            Types: (used for both cluster and space)
                "shell" (circle if occupied_dims==2)
                "ball"
                "radial" (like ball only random direction and radius instead
                    of x,y,z,... Ends up concentrated in center.)
                "cube" (random x,y,z,... but in plane or hypercube instead
                    of ball)
                "even" (attempts amorphous semi-uniformity of distances btw.
                    points)
                "grid" (attempts a gridlike uniformity)
                "pairs" (generates points in pairs of close repulsion --
                    forces excessive node generation)
                "line" (generates points in lines)
                "snake" (generate points in curvy lines)
                "oval" (like radial, but randomly varies size of each axis
                    within allowed radius sizes)
                "hierarchy" (attempts to recursively make closer and closer
                    pairs of groupings.)

        NOTE: Multiple lists in the parameters can be used to fill the space
                with varied types of data.
            occupied dimensions must be <= total dimensions.
            min_num <= max_num.
            randomize_dims: boolean.
                If false, will use same set of dims for each cluster.
            noise: a float; how much to randomly vary locations.
            normalize: boolean. If true will afterward scale all vectors
                to unit length of 1, creating a hypersphere.

    returns:
        A new analyst object, and
        A list of the clusters used to create the space before clustering
            was recalculated, for comparison. This will be different if
            clusters overlapped.
    '''
    raise NotImplementedError()
    #note, need to make it create a generic identity function for
    #   encode/decode. or use indeces.


def simulate_cluster(type, population, radius, occupied_dims,
    total_dims, randomize_dims=True, noise=0, normalize=False):
    # Same usage as the cluster parameters in simulate_space().
    # NOTE: when this function is called by simulate_space(), normalize
    #   is never True here. That would be done after the fact,
    #   on the whole simulated space, not on any one cluster.
    raise NotImplementedError()
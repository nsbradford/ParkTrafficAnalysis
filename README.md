#IQP

A method for the Worcester Polytechnic Institute (WPI) Interactive Qualifying Project (IQP), Melbourne-B15: Team CERES.
Used to analyze the locations of CERES park visitors to find hotspots.

Author: Nicholas S. Bradford

##Description

Takes a satellite image from /images to use as a reference base. Then, generate a point cloud by recording all changed pixels between the base and the /observations observation images (park visitor locations marked on the map). Using this point cloud, apply the KMeans clustering algorithm to find centroids, and a gaussian distribution to create Heatmap. Finally, blend with base image and save to /output. Original satellite image size and file names are hard-coded in as globals.

##Usage:

    iqp.py [-h] [-o N_OBSERVATIONS] [-c N_CLUSTERS] [-b N_BINS]
    optional arguments:
      -h, --help            show this help message and exit
      -o N_OBSERVATIONS, --n_observations N_OBSERVATIONS
                            number of observation files
      -c N_CLUSTERS, --n_clusters N_CLUSTERS
                            number of clusters to generate with KMeans
      -b N_BINS, --n_bins N_BINS
                            number of bins to use for the Heatmap

##Input
![huzzah](output/1_data_map.png)

##Output
![huzzah](output/5_heatmap.png)

###EOF

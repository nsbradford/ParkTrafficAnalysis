#!/usr/bin/python

"""IQP location analysis tool.

A method for the Worcester Polytechnic Institute (WPI) 
Interactive Qualifying Project (IQP), Melbourne-B15: Team CERES.
Used to analyze the locations of CERES park visitors to find hotspots.
Authored by Nicholas S. Bradford.

Takes a satellite image from /images to use as a reference base. Then, 
generate a point cloud by recording all changed pixels between the base and
the observation images (park visitor locations marked on the map). Using
this point cloud, apply the KMeans clustering algorithm to find centroids,
and a gaussian distribution to create Heatmap. Finally, blend with base image.

Usage:
    usage: iqp.py [-h] [-o N_OBSERVATIONS] [-c N_CLUSTERS] [-b N_BINS]

    optional arguments:
      -h, --help            show this help message and exit
      -o N_OBSERVATIONS, --n_observations N_OBSERVATIONS
                            number of observation files
      -c N_CLUSTERS, --n_clusters N_CLUSTERS
                            number of clusters to generate with KMeans
      -b N_BINS, --n_bins N_BINS
                            number of bins to use for the Heatmap
"""

import argparse
import numpy as np
import cv2
from sklearn.cluster import KMeans
from scipy.stats import kde
import matplotlib.pyplot as plt
from PIL import Image, ImageChops

#images are 1163 x 828
X_MIN = 0
X_MAX = 1163
Y_MIN = 0
Y_MAX = 828
SATELLITE_FILE = "images/satellite.png"
OBSERVATION_FILE = "observations/observation_0"
OBSERVATION_FILE_EXT = ".png"
MAP_VISUAL_FILE = "output/1_data_map.png"
RAW_VISUAL_FILE = "output/2_data_raw.png"
RAW_CLUSTERS_FILE = "output/3_data_clusters_raw.png"
MAP_CLUSTERS_FILE = "output/4_data_clusters_map.png"
RAW_HEATMAP_FILE = "output/heatmap_raw.png"
TRANSFORMED_HEATMAP_FILE = "output/heatmap_transformed.png"
HEATMAP_FILE = "output/5_heatmap.png"


def cluster_map(data, n_clusters, output_img, base_img):
    """Apply KMeans clustering to
    Args:
        data: list of [x, y] coordinates.
        n_clusters: number of clusters to generate with KMeans.
        output_img: image to overlay clusters onto.
        base_img: original map image without any data or observations
    Returns:
        None
    """
    kmeans = KMeans(n_clusters)
    kmeans_data = data
    kmeans.fit(kmeans_data)
    for x, y in data:
        cv2.circle(output_img, (int(y), int(x)), 2, (0, 0, 200), -1)
    cv2.imwrite(RAW_VISUAL_FILE, output_img - base_img)
    cv2.imwrite(MAP_VISUAL_FILE, output_img)
    for x, y in kmeans.cluster_centers_:
        cv2.circle(output_img, (int(y), int(x)), 12, (0, 0, 255), -1)
    cv2.imwrite(RAW_CLUSTERS_FILE, output_img - base_img)
    cv2.imwrite(MAP_CLUSTERS_FILE, output_img)


def heat_map(data, n_bins, output_img):
    """Overlay a Heat Map onto the output image.
    Args:
        data: list of [x, y] coordinates.
        n_bins: number of bins to use for the Heatmap.
        output_img: image to overlay Heatmap onto.
    """
    data = np.array(data)

    # Pyplot: Evaluate a gaussian kde on a regular grid of n_bins x n_bins
    x, y = [i[0] for i in data], [j[1] for j in data]
    fig, axes = plt.subplots(ncols=1, nrows=1, sharex=True, sharey=True)
    k = kde.gaussian_kde(data.T)
    xi, yi = np.mgrid[Y_MIN:Y_MAX:n_bins*1j, X_MIN:X_MAX:n_bins*1j]
    zi = k(np.vstack([xi.flatten(), yi.flatten()]))
    plt.ylim((X_MIN, X_MAX))
    plt.xlim((Y_MIN, Y_MAX))
    plt.axis('off')
    axes.pcolormesh(xi, yi, zi.reshape(xi.shape))
    plt.savefig(RAW_HEATMAP_FILE, bbox_inches='tight', pad_inches=0)

    # PIL: remove bordering whitespace (crop), rotate, and resize
    im = Image.open(RAW_HEATMAP_FILE)
    bg = Image.new(im.mode, im.size, im.getpixel((0, 0)))
    diff = ImageChops.difference(im, bg)
    diff = ImageChops.add(diff, diff, 2.0, -100)
    bbox = diff.getbbox()
    im = im.crop(bbox)
    angle = 270
    w, h = im.size
    im = im.rotate(angle, expand=1)
    im = im.resize((X_MAX, Y_MAX))
    im.save(TRANSFORMED_HEATMAP_FILE)

    # OpenCV: blend images
    heat_img = cv2.imread(TRANSFORMED_HEATMAP_FILE, cv2.IMREAD_COLOR)
    dst = cv2.addWeighted(output_img, 0.8, heat_img, 0.5, 0)
    cv2.imwrite(HEATMAP_FILE, dst)
    cv2.imshow('Heatmap with Clusters', dst)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def main(n_observations, n_clusters, n_bins):
    """Use a set of observations to create Clusters and a Heatmap.
    Args:
        n_observations: number of observation files.
        n_clusters: number of clusters to generate with KMeans.
        n_bins: number of bins to use for the Heatmap.
    Returns:
        None
    """
    base_img = cv2.imread(SATELLITE_FILE, cv2.IMREAD_COLOR)
    assert base_img is not None, "SATELLITE_FILE did not load."
    data = []
    for i in xrange(1, n_observations + 1):
        data_file = OBSERVATION_FILE + str(i) + OBSERVATION_FILE_EXT
        data_img = cv2.imread(data_file, cv2.IMREAD_COLOR)
        assert data_img is not None, "OBSERVATION_FILE did not load: " + data_file
        img = base_img - data_img
        output_img = base_img.copy()
        rows, columns, channels = img.shape
        for i in xrange(rows):
            for j in xrange(columns):
                if img.item(i, j, 2) > 0:
                    data.append([i, j])
    cluster_map(data, n_clusters, output_img, base_img)
    heat_map(data, n_bins, output_img)


if __name__ == "__main__":
    np.random.seed(13)
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--n_observations", default=4, type=int,
        help="number of observation files")
    parser.add_argument("-c", "--n_clusters", default=12, type=int,
        help="number of clusters to generate with KMeans")
    parser.add_argument("-b", "--n_bins", default=1000, type=int, 
        help="number of bins to use for the Heatmap")
    args = parser.parse_args()
    main(args.n_observations, args.n_clusters, args.n_bins)

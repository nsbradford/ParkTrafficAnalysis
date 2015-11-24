#!/usr/bin/python

import argparse
import numpy as np
import cv2
from sklearn.cluster import KMeans

# heat maps
import matplotlib.pyplot as plt
from scipy.stats import kde

# rotating, scaling, and cropping
from PIL import Image, ImageChops


#images are 1163 x 828
x_min = 0
x_max = 1163
y_min = 0
y_max = 828

satellite_file = "images/satellite.png"
observation_file = "observations/observation_0"
observation_file_ext = ".png"

heatmap_file = "output/heat_map.png"
raw_heatmap = "output/foo.png"
transformed_heatmap = "output/new_foo.png"


def main(n_observations, n_clusters, n_bins):
    base_img = cv2.imread(satellite_file, cv2.IMREAD_COLOR)
    assert base_img is not None, "data_img did not load."
    data = []
    for i in xrange(1, n_observations + 1):
        data_file = observation_file + str(i) + observation_file_ext
        data_img = cv2.imread(data_file, cv2.IMREAD_COLOR)
        assert data_img is not None, "data_img did not load: " + data_file
        img = base_img - data_img
        output_img = base_img.copy()
        rows, columns, channels = img.shape
        for i in xrange(rows):
            for j in xrange(columns):
                if img.item(i,j,2) > 0:
                    data.append([i, j])
    cluster_map(data, n_clusters, output_img, base_img)
    heat_map(data, n_bins, output_img)


def cluster_map(data, n_clusters, output_img, base_img):
    """ Apply KMeans clustering to """
    kmeans = KMeans(n_clusters)
    kmeans_data = data
    kmeans.fit(kmeans_data)
    for x, y in data:
        cv2.circle(output_img,(int(y), int(x)), 4, (0,0,255), -1)
    cv2.imwrite('output/input_data.png', output_img)
    cv2.imwrite('output/raw_data.png', output_img - base_img)
    for x, y in kmeans.cluster_centers_:
        cv2.circle(output_img, (int(y), int(x)), 10, (255,150,0), -1)
    cv2.imwrite('output/output_clusters.png', output_img)
    cv2.imwrite('output/raw_clusters.png', output_img - base_img)   


# from http://stackoverflow.com/questions/19390320/scatterplot-contours-in-matplotlib
def heat_map(data, n_bins, output_img):
    data = np.array(data)

    # Pyplot: Evaluate a gaussian kde on a regular grid of n_bins x n_bins over data extents
    x, y = [i[0] for i in data], [j[1] for j in data]
    fig, axes = plt.subplots(ncols=1, nrows=1, sharex=True, sharey=True) 
    k = kde.gaussian_kde(data.T)
    xi, yi = np.mgrid[y_min:y_max:n_bins*1j, x_min:x_max:n_bins*1j] #xi, yi = np.mgrid[x.min():x.max():n_bins*1j, y.min():y.max():n_bins*1j]
    zi = k(np.vstack([xi.flatten(), yi.flatten()]))
    plt.ylim((x_min, x_max))
    plt.xlim((y_min, y_max))
    plt.axis('off')
    axes.pcolormesh(xi, yi, zi.reshape(xi.shape))
    plt.savefig(raw_heatmap, bbox_inches='tight', pad_inches=0)
    
    # PIL: remove bordering whitespace (crop), rotate, and resize   
    im = Image.open(raw_heatmap)
    bg = Image.new(im.mode, im.size, im.getpixel((0,0)))
    diff = ImageChops.difference(im, bg)
    diff = ImageChops.add(diff, diff, 2.0, -100)
    bbox = diff.getbbox()
    im = im.crop(bbox)
    angle = 270
    w, h = im.size
    im = im.rotate( angle, expand=1 )
    im = im.resize((x_max, y_max))
    im.save(transformed_heatmap)

    # OpenCV: 
    heat_img = cv2.imread(transformed_heatmap, cv2.IMREAD_COLOR)
    dst = cv2.addWeighted(output_img, 0.8, heat_img, 0.5, 0)
    cv2.imshow('dst', dst)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imwrite(heatmap_file, dst)
    
    
if __name__ == "__main__":
    np.random.seed(13)
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--n_observations", default=4, type=int)
    parser.add_argument("-c", "--n_clusters", default=12, type=int)
    parser.add_argument("-b", "--n_bins", default=1000, type=int)
    args = parser.parse_args()
    main(args.n_observations, args.n_clusters, args.n_bins)
#!/usr/bin/python

import argparse
import numpy as np
import cv2
from sklearn.cluster import KMeans

import matplotlib.pyplot as plt
from scipy.stats import kde

from PIL import Image


#images are 1163 x 828
x_min = 0
x_max = 1163
y_min = 0
y_max = 828
satellite_file = "images/satellite.png"
observation_file = "observations/observation_0"
observation_file_ext = ".png"
heatmap_file = "output/heat_map.png"

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
                    #print "(", i, ",", j, ")"
                    data.append([i, j])
    kmeans = KMeans(n_clusters)
    kmeans_data = data
    kmeans.fit(kmeans_data)
    #print kmeans.cluster_centers_
    #print data
    for x, y in data:
        cv2.circle(output_img,(int(y), int(x)), 4, (0,0,255), -1)
    cv2.imwrite('output/input_data.png', output_img)
    cv2.imwrite('output/raw_data.png', output_img - base_img)
    for x, y in kmeans.cluster_centers_:
        cv2.circle(output_img, (int(y), int(x)), 10, (255,150,0), -1)
    cv2.imshow('image', output_img)
    cv2.imwrite('output/output_clusters.png', output_img)
    cv2.imwrite('output/raw_clusters.png', output_img - base_img)

    heat_map(data, n_bins, output_img)


# from http://stackoverflow.com/questions/19390320/scatterplot-contours-in-matplotlib
def heat_map(data, n_bins, output_img):
    data = np.array(data)
    x, y = [i[0] for i in data], [j[1] for j in data]
    fig, axes = plt.subplots(ncols=1, nrows=1, sharex=True, sharey=True)

    # Evaluate a gaussian kde on a regular grid of n_bins x n_bins over data extents
    k = kde.gaussian_kde(data.T)
    xi, yi = np.mgrid[y_min:y_max:n_bins*1j, x_min:x_max:n_bins*1j] #xi, yi = np.mgrid[x.min():x.max():n_bins*1j, y.min():y.max():n_bins*1j]
    zi = k(np.vstack([xi.flatten(), yi.flatten()]))

    #axes.set_title('Gaussian KDE')
    axes.pcolormesh(xi, yi, zi.reshape(xi.shape))
    plt.ylim((x_min, x_max))
    plt.xlim((y_min, y_max))
    plt.axis('off')
    fig.tight_layout()
    plt.savefig('output/foo.png', bbox_inches='tight')
    plt.show()

    #rotate clockwise 
    #img = cv2.imread('', cv2.IMREAD_COLOR)
    
    src_im = Image.open("output/foo.png")
    angle = 270
    size = x_max, y_max

    #dst_im = Image.new("RGBA", (x_max, y_max), "blue" )
    im = src_im.convert('RGBA')
    rot = im.rotate( angle, expand=1 ).resize(size)
    #dst_im.paste( rot, (50, 50), rot )
    rot.save("output/new_foo.png")

    heat_img = cv2.imread('output/new_foo.png', cv2.IMREAD_COLOR)
    dst = cv2.addWeighted(output_img, 0.5, heat_img, 0.5, 0)
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
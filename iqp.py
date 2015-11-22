#!/usr/bin/python

import argparse
import numpy as np
import cv2
from sklearn.cluster import KMeans

def main(n_observations, n_clusters):
    base_img = cv2.imread('images/satellite.png', cv2.IMREAD_COLOR)
    assert base_img is not None, "data_img did not load."
    data = []
    for i in xrange(1, n_observations + 1):
        data_file = "observations/observation_0" + str(i) + ".png"
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
        cv2.circle(output_img,(int(y), int(x)), 3, (255,150,0), -1)
    cv2.imwrite('output/input_data.png', output_img)
    for x, y in kmeans.cluster_centers_:
        cv2.circle(output_img, (int(y), int(x)), 7, (0,0,255), -1)
    cv2.imshow('image', output_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imwrite('output/output_clusters.png', output_img)

if __name__ == "__main__":
    np.random.seed(13)
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--n_observations", default=1, type=int)
    parser.add_argument("-c", "--n_clusters", default=1, type=int)
    args = parser.parse_args()
    main(args.n_observations, args.n_clusters)
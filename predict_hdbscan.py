import os
import sys
import argparse
import math
import numpy as np
import pandas as pd
import scipy
from scipy import stats

from numpy import mean
from numpy import std
from numpy.random import randn
from numpy.random import seed
from matplotlib import pyplot

from sklearn.cluster import DBSCAN
import hdbscan
import auxiliary.laserscan

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict labels for point cloud data")

    parser.add_argument(
      '--dataset', '-d',
      type=str,
      required=True,
      help='Dataset dir. No Default',
    )

    parser.add_argument(
      '--predictions', '-p',
      type=str,
      required=True,
      help='Predictions dir. No Default',
    )

    parser.add_argument(
        '--sequence', '-s',
        type=str,
        default="00",
        required=True,
        help='Sequence to parse. Defaults to %(default)s',
    )

    parser.add_argument(
        '--color', '-c',
        type=bool,
        default=False,
        required=False,
        help='Use random colors instead of actual segmentation labels. Defaults to %(default)s',
    )

    args = parser.parse_args()
    print("  ========================== Arguments ==========================  ")
    print("\n".join(["  {}:\t{}".format(k,v) for (k,v) in vars(args).items()]))
    print("  ===============================================================  \n")

    files = sorted(os.listdir(os.path.join(args.dataset, "sequences", \
                                           args.sequence, "velodyne")))
    outdir = args.predictions

    # random class labels for random color mode
    classes = [ 1, 10, 11, 13, 15, 16, 18, 20, 30, 31, 32, 40, 44, 48, 49,
               50, 51, 52, 60, 70, 71, 72, 80, 81, 90, 252, 256, 253, 254, \
               255, 257, 258, 259 ]
    class_i = 0

    # object tracking
    objs = list()
    obj_id = 0
    idlist = list()
    idlist_previous = list()

    # iterate through files, that each describe one scanned frame
    for i,filename in enumerate(files):
        frame = filename[:-4]

        # both csv and bin files are supported for reading
        if str(filename).endswith(".bin"):
            inpath = os.path.join(args.dataset, "sequences", args.sequence, \
                                  "velodyne", frame + ".bin")
            if not os.path.exists(inpath):
                raise RuntimeError("velodyne directory missing: " + inpath)

            # read data from bin
            scan = auxiliary.laserscan.LaserScan()
            scan.open_scan(inpath)
            dataset = scan.points
        elif str(filename).endswith(".csv"):
            inpath = os.path.join(args.dataset, "sequences", args.sequence, \
                                  "velodyne_csv", frame + ".csv")
            if not os.path.exists(inpath):
                raise RuntimeError("velodyne directory missing: " + inpath)

            # read data from csv
            df = pd.read_csv(inpath, sep=';'  , engine='python')
            dataset = df[['X', 'Y', 'Z']].to_numpy()
        else:
            continue

        print("\nNew frame: {}\n=========================\n".format(frame))

        # cluster the points in current frame
        dbscan = hdbscan.HDBSCAN(min_cluster_size=30)
        #dbscan = DBSCAN(eps=0.8, min_samples=5)
        dbscan.fit(dataset)

        # prepare clustered data
        labels = dbscan.labels_
        labels_unique = np.unique(labels)
        n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise_ = list(labels).count(-1)
        print('Estimated number of clusters: %d' % n_clusters_)
        print('Estimated number of noise points: %d' % n_noise_)

        obj = list()

        # iterate through clusters
        for i,label in enumerate(labels_unique):
            pos = (labels == label) # find index of the cluster label
            points = dataset[pos] # [[x1,y1,z1], [x2,y2,z2], ..]
            c = np.mean(points, axis=0) # mean [x,y,z]
            mins = np.argmin(points, axis=0) # indices of minima for each column
            maxs = np.argmax(points, axis=0) # indices of maxima for each column
            px = points[mins[0]] # front point [x,y,z]
            py = points[mins[1]] # left point
            pz = points[mins[2]] # bottom point
            px_ = points[maxs[0]] # far back point
            py_ = points[maxs[1]] # right point
            pz_ = points[maxs[2]] # top point
            w = abs(py_[0] - py[0])
            l = abs(px_[0] - px[0])
            h = abs(pz_[0] - pz[0])

            phi = math.degrees(math.acos(w / l)) if l != 0 else 0

            if l < 0.05 or w < 0.05 or h < 0.05:
                segm = 30 # person
            elif l < 0.85 or w < 0.85 or h < 0.85:
                segm = 51 # fence
            elif l < 5 and w < 5 and h < 5 and w / h > 0.5 and w / h < 2:
                segm = 10 # car
            else:
                segm = 1 # outlier

            if segm != 1:
                obj.append({"segm": segm, "w":w, "l":l, "h":h, "c":c, "phi":phi, "obj_id":obj_id })
                obj_id += 1

            if args.color:
                segm = classes[class_i]
                class_i = class_i + 1
                if class_i >= len(classes):
                    class_i = 0

            # overwrite all labels with the given dbscan label with our new
            # segmentation result.
            labels = np.where(labels == label, segm, labels)

        outpath = os.path.join(outdir, "sequences", args.sequence, "predictions", frame + ".label")
        if not os.path.exists(outdir): raise RuntimeError("out directory missing: " + outpath)

        # frame processing complete
        #labels.astype('uint32').byteswap().tofile('out.label')x
        labels.astype('uint32').tofile(outpath)
        objs.append(obj)

    # simple object tracking
    for i,obj in enumerate(objs):
        for i2,el in enumerate(obj):
            objs[i][i2]["prev_obj_id"] = objs[i][i2]["obj_id"]
            objs[i][i2]["new"] = True
            objs[i][i2]["frame"] = i
            objs[i][i2]["persistance"] = 0

            if i > 0:
                for i3,el2 in enumerate(objs[i-1]):
                    # check if object lies within the bounding box of an object
                    # in the previous frame
                    x1 = el2["c"][0] - el2["w"] / 2
                    x2 = el2["c"][0] + el2["w"] / 2
                    y1 = el2["c"][1] - el2["l"] / 2
                    y2 = el2["c"][1] + el2["l"] / 2
                    z1 = el2["c"][2] - el2["h"] / 2
                    z2 = el2["c"][2] + el2["h"] / 2

                    if el["c"][0] > x1 and el["c"][0] < x2 and \
                            el["c"][1] > y1 and el["c"][1] < y2 and \
                            el["c"][2] > z1 and el["c"][2] < z2:
                        # objects at same locations => we assume it is the same
                        objs[i][i2]["prev_obj_id"] = el2["prev_obj_id"]
                        objs[i][i2]["persistance"] = el2["persistance"]+1
                        objs[i][i2]["new"] = False

            if objs[i][i2]["new"]:
                # object observed for the first time
                print("frame {frame} id {obj_id} class {segm} w {w} l {l} h {h} phi {phi}".format(**objs[i][i2]))
            else:
                # object observed in multiple frames
                print("frame {frame} id {obj_id} persists for {persistance} frames and is the same as object {prev_obj_id}".format(**objs[i][i2]))
            idlist.append(el["obj_id"])

        # find objects that went missing
        diff = [item for item in idlist_previous if item not in idlist]
        for obj_id in diff:
            print("id {} from previous frame vanished".format(obj_id))
        idlist_previous = idlist.copy()
        idlist.clear()

#!/usr/bin/python

import argparse
import re
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
from skimage.measure import grid_points_in_poly, regionprops
from skimage.segmentation import find_boundaries
from BKlib import tiff_to_ndarray, KalmanSmoother2D, write_video



def correct_orientation(im):
    """Correct the frame orientations (to match imagej)"""
    return np.fliplr(im)


def smooth_2D_trajectory(pts, noise=15):

    # estimate initial velocity via regression on first few timepoints
    init_pts = 5
    vx0, _, _, _, _ = linregress(range(init_pts), pts[:init_pts, 0])
    vy0, _, _, _, _ = linregress(range(init_pts), pts[:init_pts, 1])
    initial_state = np.array([pts[0,0], pts[0,1], vx0, vy0])

    # smooth the cell center positions
    smoother = KalmanSmoother2D(noise, noise)
    smoother.set_initial_state(initial_state)
    smoother.set_measurements(pts)
    return smoother.get_smoothed_measurements()



def parse_command_line_args():

    # TODO: additional arguments: padding, kalman smoothing, add velocity vector, add cell boundary

    description_str = 'Save a window around a tracked cell as a movie file, so that the cell motion is \"frozen.\"'
    parser = argparse.ArgumentParser(description=description_str)
    parser.add_argument('tiff_movie', metavar='<tiff movie>', type=str, help='a tiff movie of the desired channel')
    parser.add_argument('boundaries_fn', metavar='<cell boundary file>', type=str,
                        help='A numpy file containing the boundary points from cell tracking (cell[label]_boundary_points.npy)')
    parser.add_argument('-o', '--out', metavar='<output movie file>', type=str, help='output movie file')
    args = parser.parse_args()
    return args.tiff_movie, args.boundaries_fn, args.out




if __name__ == '__main__':

    ### Parse command line arguments
    tiff_fn, boundaries_fn, out_fn = parse_command_line_args()
    if not out_fn:
        cell_label = re.findall(r'\d+', boundaries_fn)[0]
        fn = 'cell%s.avi' % cell_label
        out_fn = os.path.join('.', fn)
        print 'Output movie name unspecified, movie will be saved to %s.' % out_fn


    ### Load data
    print 'Loading %s...' % tiff_fn,
    sys.stdout.flush()
    frames = tiff_to_ndarray(tiff_fn)
    all_boundary_pts = np.load(boundaries_fn)
    print 'done.'
    sys.stdout.flush()


    ### compute masks from boundaries
    frame_sz = frames[0].shape
    masks = [grid_points_in_poly(frame_sz , pts) for pts in all_boundary_pts]


    ### compute window size (max mask size + padding)
    padding = 10
    bboxes = [regionprops(mask)[0].bbox for mask in masks]
    bbox_widths = [(bbox[2]-bbox[0]) for bbox in bboxes]
    bbox_heights = [(bbox[3]-bbox[1]) for bbox in bboxes]
    win_size = (max(bbox_widths)+2*padding, max(bbox_heights)+2*padding)

    # make it square (apparently avconv doesn't work correctly with arbitrary frame size...)
    win_size = (max(win_size), max(win_size))


    ### centroids
    cell_centers = np.array([regionprops(mask)[0].centroid for mask in masks])
    smooth_cell_centers = smooth_2D_trajectory(cell_centers)


    ### extract windows and put into big array

    windows = np.empty((len(frames), win_size[0], win_size[1]))
    for frame_num, (frame, center, mask) in enumerate(zip(frames, smooth_cell_centers, masks)):

        # make boundary bright
        boundary = find_boundaries(mask)
        frame += 0.5*frame.max()*boundary

        # extract window centered on mask centroid
        i, j = int(round(center[0])), int(round(center[1]))
        ista, jsta = i-win_size[0]/2, j-win_size[1]/2
        iend, jend = ista+win_size[0], jsta+win_size[1]
        win = frame[ista:iend, jsta:jend]
        windows[frame_num,:,:] = correct_orientation(win)


    # save to file
    print 'Saving movie to %s.' % out_fn
    write_video(windows, out_fn)



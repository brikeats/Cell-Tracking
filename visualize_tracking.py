#!/usr/bin/python

import re
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
from skimage import measure
from skimage.segmentation import find_boundaries
from scipy.interpolate import splprep, splev
from BKlib import tiff_to_ndarray, write_video


def correct_orientation(im):
    """Correct the frame orientations (to match imagej)"""
    return np.fliplr(im)


def interpolate_boundary_pts(pts, N=200):
    """Interpolate sparse (closed) boundary points via spline."""
    # interpolate boundary
    tck, u = splprep(pts.T, u=None, s=0.0, per=1)
    u_new = np.linspace(u.min(), u.max(), 1000)
    x_new, y_new = splev(u_new, tck, der=0)

    u_dense = np.linspace(0, 1, N+1)
    x_dense, y_dense = splev(u_dense, tck, der=0)
    return x_dense, y_dense


def parse_command_line_args():

    # TODO: additional arguments: frames per second

    description_str = 'View \"track cell\" output as a movie and optionally save the visualization to file.'
    parser = argparse.ArgumentParser(description=description_str)
    parser.add_argument('tiff_movie', metavar='<tiff movie>', type=str, help='an imagej-style tiff movie')
    parser.add_argument('boundaries_fn', metavar='<cell boundary file>', type=str,
                        help='A numpy file containing the boundary points from cell tracking (cell[label]_boundary_points.npy)')
    parser.add_argument('-o', '--out', metavar='<output movie file>', type=str, help='output movie file')
    args = parser.parse_args()
    return args.tiff_movie, args.boundaries_fn, args.out



# TODO: a mini video player would be useful here

if __name__ == '__main__':

    ### Parse command line arguments
    tiff_fn, boundaries_fn, out_fn = parse_command_line_args()


    ### Load data
    print 'Loading %s...' % tiff_fn,
    sys.stdout.flush()
    frames = tiff_to_ndarray(tiff_fn)
    all_boundary_pts = np.load(boundaries_fn)
    print 'done.'
    sys.stdout.flush()


    ### visualize cell motion to check the results ###
    boundary_pts = all_boundary_pts[0]
    boundary_x, boundary_y = interpolate_boundary_pts(boundary_pts)
    mask = measure.grid_points_in_poly(frames[0].shape, boundary_pts)
    center = measure.regionprops(mask)[0].centroid

    plt.ion()
    fig = plt.figure()
    implot = plt.imshow(frames[0], cmap='gray')
    boundary_plot, = plt.plot(boundary_y, boundary_x, 'bo', markeredgecolor='none', markersize=1)
    boundary_pts_plot, = plt.plot(boundary_pts[:,1], boundary_pts[:,0], 'ro', markeredgecolor='none', markersize=4)
    center_point, = plt.plot(center[1], center[0], 'ro', markeredgecolor='r', markersize=7)
    plt.axis('off')
    fig.canvas.draw()

    for frame_num, (frame, boundary_pts) in enumerate(zip(frames, all_boundary_pts)):

        boundary_x, boundary_y = interpolate_boundary_pts(boundary_pts)
        mask = measure.grid_points_in_poly(frame.shape, boundary_pts)
        center = measure.regionprops(mask)[0].centroid

        implot.set_data(frame)
        boundary_plot.set_data(boundary_y, boundary_x)
        boundary_pts_plot.set_data(boundary_pts[:,1], boundary_pts[:,0])
        center_point.set_data(center[1], center[0])
        plt.title('frame %i' % (frame_num+1))
        fig.canvas.draw()

    plt.ioff()
    plt.show()


    ### save to file
    if out_fn:
        # cell_label = re.findall(r'\d+', boundaries_fn)[0]
        movie_frames = np.empty_like(frames)
        for frame_num, (frame, boundary_pts) in enumerate(zip(frames.copy(), all_boundary_pts)):
            mask = measure.grid_points_in_poly(frame.shape, boundary_pts)
            boundary = find_boundaries(mask)
            frame += 0.5*frame.max()*boundary
            movie_frames[frame_num,:,:] = correct_orientation(frame)
        # fn = 'cell%i_visualization.avi' % cell_label
        # print 'Saving visualization to', fn
        write_video(movie_frames, out_fn)
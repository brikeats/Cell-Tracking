#!/usr/bin/python

import re
import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
from skimage.measure import grid_points_in_poly, regionprops
import pims
from BKlib import KalmanSmoother2D


def orient_for_imshow(frame):
    return np.fliplr(frame.T)


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

    description_str = 'Display a few plot to quanitify cell motion.'
    parser = argparse.ArgumentParser(description=description_str)
    parser.add_argument('boundaries_fn', metavar='cell_boundary_file', type=str,
                        help='A numpy file containing the boundary points from cell tracking (cell[label]_boundary_points.npy)')
    parser.add_argument('-bg', metavar='background_movie', type=str,
                        required=False, help='a tiff movie to display under trajectory')
    args = parser.parse_args()
    return args.boundaries_fn, args.bg



if __name__ == '__main__':

    ### Parse command line arguments
    boundaries_fn, tif_fn = parse_command_line_args()


    ### Load data
    all_boundary_pts = np.load(boundaries_fn)
    cell_label = re.findall(r'\d+', boundaries_fn)[0]

    if tif_fn:
        frames = pims.TiffStack(tif_fn)
        frame = frames[0]

        ### smooth and plot cell trajectory ###
        masks = [grid_points_in_poly(frame.shape, pts) for pts in all_boundary_pts]
        centers = np.array([regionprops(mask)[0].centroid for mask in masks])


        # estimate initial velocity via regression on first few timepoints
        init_pts = 5
        vx0, _, _, _, _ = linregress(range(init_pts), centers[:init_pts, 0])
        vy0, _, _, _, _ = linregress(range(init_pts), centers[:init_pts, 1])
        initial_state = np.array([centers[0,0], centers[0,1], vx0, vy0])

        # smooth the cell center positions
        position_noise = 15.0  # higher noise -> heavier smoothing
        smoother = KalmanSmoother2D(position_noise, position_noise)
        smoother.set_initial_state(initial_state)
        smoother.set_measurements(centers)
        smooth_cell_centers = smoother.get_smoothed_measurements()
        cell_velocities = smoother.get_velocities()

        # plot trajectory
        plt.figure()
        plt.imshow(orient_for_imshow(frame), cmap='gray')

        plt.plot(centers[:,1], centers[:,0], 'bx')
        plt.plot(smooth_cell_centers[:,1], smooth_cell_centers[:,0],'r-')
        plt.title('Trajectory for cell %s' % cell_label)
        plt.axis('off')
        # plt.axis('equal')
        # plt.gca().invert_yaxis()
        # plt.xlabel('x (pixels)')
        # plt.ylabel('y (pixels)')

        # show histogram of velocity
        plt.figure()
        plt.hist(cell_velocities[:,1], bins=25)
        plt.hist(cell_velocities[:,0], bins=25)
        plt.legend(['x speed', 'y speed'], numpoints=1)
        plt.title('Speed distribution for cell %s' % cell_label)
        plt.xlabel('speed (pixels/frame)')
        plt.ylabel('frame count')


        plt.show()
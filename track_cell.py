#!/usr/bin/python

import os
import sys
import argparse
import time
import pims
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from scipy.ndimage.filters import uniform_filter
from scipy.interpolate import splprep, splev
from scipy.integrate import simps
from skimage import filters, feature, morphology, measure
from skimage.morphology import disk
from skimage.color import rgb2gray
from BKlib import fit_snake, tiff_to_ndarray

    

def enhance_ridges(frame, mask=None):
    """A ridge detection filter (larger hessian eigenvalue)"""
    blurred = filters.gaussian_filter(frame, 2)
    sigma = 4.5
    Hxx, Hxy, Hyy = feature.hessian_matrix(blurred, sigma=sigma, mode='nearest')
    ridges = feature.hessian_matrix_eigvals(Hxx, Hxy, Hyy)[0]
    return np.abs(ridges)


def create_mask(frame):
    """"Create a big mask that encompasses all the cells"""
    
    # detect ridges
    ridges = enhance_ridges(frame)

    # threshold ridge image
    thresh = filters.threshold_otsu(ridges)
    thresh_factor = 1.1
    prominent_ridges = ridges > thresh_factor*thresh
    prominent_ridges = morphology.remove_small_objects(prominent_ridges, min_size=128)

    # the mask contains the prominent ridges
    mask = morphology.convex_hull_image(prominent_ridges)
    mask = morphology.binary_erosion(mask, disk(10))
    return mask


def frame_to_distance_images(frame):
    """
    Compute the skeleton of the cell boundaries, return the 
    distance transform of the skeleton and its branch points.
    """
    
    # distance from ridge midlines
    frame = rgb2gray(frame)
    ridges = enhance_ridges(frame)
    thresh = filters.threshold_otsu(ridges)
    prominent_ridges = ridges > 0.8*thresh
    skeleton = morphology.skeletonize(prominent_ridges)
    edge_dist = ndimage.distance_transform_edt(-skeleton)
    edge_dist = filters.gaussian_filter(edge_dist, sigma=2)

    # distance from skeleton branch points (ie, ridge intersections)
    blurred_skeleton = uniform_filter(skeleton.astype(float), size=3)
    corner_im = blurred_skeleton > 4./9
    corner_dist = ndimage.distance_transform_edt(-corner_im)
    
    return edge_dist, corner_dist


def mask_to_boundary_pts(mask, pt_spacing=5):
    """
    Convert a binary image containing a single object to a set
    of 2D points that are equally spaced along the object's contour.
    """

    # interpolate boundary
    boundary_pts = measure.find_contours(mask, 0)[0]
    tck, u = splprep(boundary_pts.T, u=None, s=0.0, per=1)
    u_new = np.linspace(u.min(), u.max(), 1000)
    x_new, y_new = splev(u_new, tck, der=0)

    # get equi-spaced points along spline-interpolated boundary
    x_diff, y_diff = np.diff(x_new), np.diff(y_new)
    S = simps(np.sqrt(x_diff**2 + y_diff**2))
    N = int(round(S/pt_spacing))

    u_equidist = np.linspace(0, 1, N+1)
    x_equidist, y_equidist = splev(u_equidist, tck, der=0)
    return np.array(zip(x_equidist, y_equidist))


def segment_cells(frame, mask=None):
    """
    Compute the initial segmentation based on ridge detection + watershed.
    This works reasonably well, but is not robust enough to use by itself.
    """
    
    blurred = filters.gaussian_filter(frame, 2)
    ridges = enhance_ridges(frame)
    
    # threshold ridge image
    thresh = filters.threshold_otsu(ridges)
    thresh_factor = 0.6
    prominent_ridges = ridges > thresh_factor*thresh
    prominent_ridges = morphology.remove_small_objects(prominent_ridges, min_size=256)
    prominent_ridges = morphology.binary_closing(prominent_ridges)
    prominent_ridges = morphology.binary_dilation(prominent_ridges)
    
    # skeletonize
    ridge_skeleton = morphology.medial_axis(prominent_ridges)
    ridge_skeleton = morphology.binary_dilation(ridge_skeleton)
    ridge_skeleton *= mask
    ridge_skeleton -= mask
    
    # label
    cell_label_im = measure.label(ridge_skeleton)
    
    # morphological closing to fill in the cracks
    for cell_num in range(1, cell_label_im.max()+1):
        cell_mask = cell_label_im==cell_num
        cell_mask = morphology.binary_closing(cell_mask, disk(3))
        cell_label_im[cell_mask] = cell_num
    
    return cell_label_im 


class CellSelectorGUI:
    """
    This class displays a labelled image and allows the user to select
    a region of interest with the mouse. All the labels that are clicked
    on are stored in the list "cell_labels". 
    """
    
    def __init__(self, cell_labels):
        
        cell_mask = np.ma.masked_where(np.ones_like(cell_labels), np.ones(cell_labels.shape))
        
        grid_sz = 7
        self.fig = plt.figure()
        plt.subplot2grid((grid_sz,grid_sz), (0,0), colspan=grid_sz, rowspan=grid_sz-1)
        plt.imshow(cell_labels, cmap='jet')
        self.mask_im = plt.imshow(cell_mask, cmap='gray')
        plt.title('Click to select a cell to track')
        plt.axis('off')
        
        button_ax = plt.subplot2grid((grid_sz,grid_sz ), (grid_sz-1,grid_sz/2))
        self.done_button = plt.Button(button_ax, 'Done')
        self.done_button.on_clicked(self.on_button_press)
        self.cid = self.fig.canvas.mpl_connect('button_press_event', self.on_mouse_click)
        
        self.cell_labels = cell_labels
        self.selected_cell_labels = []
        self.cell_mask = None
        self.fig.canvas.start_event_loop(timeout=-1)
        
    def on_mouse_click(self, event):
        try:
            i, j = int(round(event.ydata)), int(round(event.xdata))
            label = self.cell_labels[i,j]
            if label != 0:
                self.cell_mask = self.cell_labels==label
                self.cell_mask = morphology.binary_dilation(self.cell_mask)
                cell_mask = np.ma.masked_where(self.cell_mask==0, 255*np.ones(self.cell_mask.shape))
                self.mask_im.set_data(cell_mask)
                plt.draw()
                self.selected_cell_labels.append(label)
        except TypeError:  
            pass  # clicked outside axes
        
    def on_button_press(self, event):
        plt.close()
        self.fig.canvas.stop_event_loop()
        

def parse_command_line_args():
    # TODO: additional arguments: alpha, beta, gamma, spacing of boundary points.
    # TODO: additional arguments: some of the sigmas and/or thresholds?

    description_str = 'Select a cell from a movie and track it across frames.'
    parser = argparse.ArgumentParser(description=description_str)
    parser.add_argument('tiff_movie', metavar='tiff movie', type=str, help='an imagej-style tiff movie')
    args = parser.parse_args()
    return args.tiff_movie




if __name__ == '__main__':

    ### parse command line arguments
    tiff_fn = parse_command_line_args()

    ### load raw movie frames
    print 'Loading %s...' % tiff_fn,
    sys.stdout.flush()
    frames = tiff_to_ndarray(tiff_fn).astype(float)
    print 'done.'
    sys.stdout.flush()


    ### Compute the big mask (contains all cells, same for all frames)
    print 'Computing global mask...',
    sys.stdout.flush()
    mask_frame = frames[0,:,:]
    mask = create_mask(mask_frame)
    print 'done.'
    sys.stdout.flush()


    ### Segment first frame via ridge detection + watershed
    print 'Computing initial segmentation...',
    sys.stdout.flush()
    frame = frames[0]
    cell_labels = segment_cells(frame, mask)
    print 'done.'
    sys.stdout.flush()


    ### show the GUI and select a cell for scrutiny
    plt.ion()
    cell_selector = CellSelectorGUI(cell_labels)
    selected_label = cell_selector.selected_cell_labels[-1]
    cell_mask = cell_selector.cell_mask
    print 'You selected cell %i.' % selected_label
    sys.stdout.flush()
    # FIXME: this gives some mysterious warnings
    # FIXME: catch the case where cell_selector.selected_cell_labels is empty
    # FIXME: alpha (and possibly beta) should scale with point spacing


    ### Compute the snake-based contour for each frame

    # initial boundary points
    boundary_pts = mask_to_boundary_pts(cell_mask, pt_spacing=6)

    # allocate array
    all_boundary_pts = np.empty((len(frames), boundary_pts.shape[0], boundary_pts.shape[1]))

    # Frame-by-frame snake fit. This is fairly slow; takes ~80 sec on my laptop.
    tsta = time.clock()
    print 'Tracking cell %i across %i frames...' % (selected_label, len(frames))
    for frame_num, frame in enumerate(frames):

        # print progress
        if frame_num%10 == 0:
            print 'Frame %i of %i' % (frame_num, len(frames))
            sys.stdout.flush()

        # compute distance transforms and fit snake
        edge_dist, corner_dist = frame_to_distance_images(frame)
        boundary_pts = fit_snake(boundary_pts, edge_dist, corner_dist,
                                 alpha=0.1, beta=0.1, gamma=0.8,
                                 nits=20)

        # TODO: resample the points along the curve to maintain contant spacing
        # store results in big array
        all_boundary_pts[frame_num,:,:] = boundary_pts

    print 'elapsed time:', time.clock() - tsta


    ### write boundary points to file
    outdir = '.'
    out_fn = 'cell%i_boundary_points.npy' % selected_label
    out_fn = os.path.join(outdir, out_fn)
    print 'Saving boundary points to', out_fn
    np.save(out_fn, all_boundary_pts)
    
    
    

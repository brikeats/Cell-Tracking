#!/bin/bash

BOUNDARY_FILENAME=cell_boundary_points.npy 
./track_cell.py cell_membranes.tiff -o ${BOUNDARY_FILENAME}
./visualize_tracking.py cell_membranes.tiff ${BOUNDARY_FILENAME} -o tracking_visualization.avi
./create_cell_movie.py cell_membranes.tiff ${BOUNDARY_FILENAME} -o cell6.avi
./plot_cell_trajectory.py cell_membranes.tiff ${BOUNDARY_FILENAME}


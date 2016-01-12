#!/bin/bash

BOUNDARY_FILENAME=cell_boundary_points.npy 

echo ----- Running track_cell.py ------
./track_cell.py cell_membranes.tiff -o ${BOUNDARY_FILENAME}
echo

echo ----- Running visualize_tracking.py ------
./visualize_tracking.py cell_membranes.tiff ${BOUNDARY_FILENAME} -o tracking_visualization.avi
echo

echo ----- Running create_cell_movie.py ------
./create_cell_movie.py cell_membranes.tiff ${BOUNDARY_FILENAME} -o cell_zoom.avi
echo

echo ----- Running plot_cell_trajectory.py ------
./plot_cell_trajectory.py cell_membranes.tiff ${BOUNDARY_FILENAME}
echo

# Cell Tracking

Python scripts for tracking cells in fluorescent microscopy, for [Ed Munro's group](http://munrolab.bsd.uchicago.edu/).

## Prerequisites

* Python 2.7
* Python scientific stack (`numpy`, `scipy`, `matplotlib`)
* `scikit-image`
* `pykalman`
* `pims` for reading tiff stacks
* `ffmpeg` or `avconv` command line tools for creating movies

Everything is pure python except for the function `write_movie` in `BKlib.py`, which uses `ffmpeg`/`avconv`. One of these will probably be installed on Linux machines; it can be installed on Mac using the binaries [here](http://ffmpegmac.net/). The scripts should work on Windows, except for the function `write_movie`, which depends on `avconv`.


## Scripts

All of the scripts take a tiff movie as input. You can see help instructions by typing the script name without any arguments. I've included a sample movie, `cell_membranes.tiff`.

`track_cell.py` must be run first. It takes a tiff file as input, does a simple watershed-based segmentation, and presents a GUI to the user to select a region of interest (i.e., a single segmented cell). This region is then tracked across subsequent frames using an active contour/snakes model. Finally, the control points from the contours are saved to an `npy` file for futher analysis. This output `npy` file is the main output; take note of its name, since it is a required argument for the other scripts.

`visualize_tracking.py` shows the movie with the tracking results overlaid. Use the `-o` flag to specify an output avi filename in order to save the visualization as a movie.

`create_cell_movie.py` extracts a small window around the cell of interest and saves it to a movie, so that the motion of the cell is "frozen". Note that the tiff can be a different channel from that used for cell tracking, i.e., you can use a channel that displays cell boundaries in `track_cell.py`, but another channel with this script. 

`plot_cell_trajectory.py` computes the centroids of the cell boundaries, smooths the trajectory with a Kalman filter, and creates a few plots to quantify motion.




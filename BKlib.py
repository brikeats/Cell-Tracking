import time
import itertools
import platform
import subprocess
from functools import partial
from scipy import optimize, ndimage
from scipy.integrate import simps
from scipy.interpolate import splev, splprep
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from numpy import ma
from matplotlib.colors import ListedColormap
from pykalman import KalmanFilter
from skimage import feature, filters, measure
from skimage.external import tifffile

try:
    from subprocess import DEVNULL # py3k
except ImportError:
    import os
    DEVNULL = open(os.devnull, 'wb')

"""
Some functions that I've found helpful. I'm sure this is reinventing the wheel,
but whatever.
"""

def isplit(iterable, splitters):
    """
    Splits a list about a particular element into a list-of-lists
    thanks to this guy: http://stackoverflow.com/questions/4322705/split-a-list-into-nested-lists-on-a-value
    """
    return [list(g) for k,g in itertools.groupby(iterable,lambda x:x in splitters) if not k]
    

def partition(list_, num):
    # Partition list as evenly as possible.
    part_sizes = [len(list_) / int(num) for _ in range(num)]
    remainder = len(list_) % num    
    for part_num in range(remainder):
        part_sizes[part_num] += 1    
    end_inds = np.cumsum(part_sizes)
    sta_inds = [end - size for end, size in zip(end_inds, part_sizes)]
    return [list_[sta_ind:end_ind] for sta_ind, end_ind in zip(sta_inds, end_inds)]


def partition_indices(list_, num):
    # Partition list as evenly as possible, return zipped indices.
    part_sizes = [len(list_) / int(num) for _ in range(num)]
    remainder = len(list_) % num    
    for part_num in range(remainder):
        part_sizes[part_num] += 1    
    end_inds = np.cumsum(part_sizes)
    sta_inds = [end - size for end, size in zip(end_inds, part_sizes)]
    return zip(sta_inds, end_inds)


def read_config_file(fn):
    """
    Read a simple config file. More complex configs should be in xml or yaml.
    Values should be in format "key=value" or "key value". Values are converted
    to int's or float's if possible; if not, it's a string.
    """
    config = dict()
    with open(fn) as f:
        for line in f.readlines():
            line = line.rstrip()  # remove newline character
            
            # skip blank lines and comment lines
            if not line or line[0]=='#':
                continue
            
            # remove inline comments
            ind = line.find('#')
            if ind != -1:
                line = line[:ind].strip()
            
            # parse
            if '=' in line:
                line_parts = line.split('=')
            else:
                line_parts = line.split(' ')
            if len(line_parts) != 2:
                print('Could not parse line', line, ', skipping...')
                continue
            
            # cast to appropriate type
            key, val_str = line_parts[0].strip(), line_parts[1].strip()
            try:
                config[key] = int(val_str)
            except ValueError:
                try:
                    config[key] = float(val_str)
                except ValueError:
                    config[key] = val_str.replace('"','')

    return config            
    

def print_image_properties(im):
    if not isinstance(im, (np.ndarray)):
        raise TypeError('print_image_properties only handles 2D or 3D numpy arrays')
    try:
        nchan = im.shape[3]
    except IndexError:
        nchan = 1
    print()
    print('image size:    %i x %i' % (im.shape[0], im.shape[1]))
    print('num. channels: %i' % nchan)
    print('dtype: %s' % im.dtype)
    print('min, max: %.1f, %.1f' % (np.min(im), np.max(im)))
    print('mean, stdev: %.1f, %.1f' % (np.mean(im), np.std(im)))
    

def vidshow(frames, start_frame=0, end_frame=-1, fps=10, **kwargs):
    # similar to imshow, but for arrays with a time dimension
    if not isinstance(frames, np.ndarray):
        raise TypeError('vidshow requires a 3D or 4D numpy array')
    if len(frames.shape) == 3:
        is_color = False
    elif len(frames.shape) == 4:
        is_color = True
        if frames.shape[3] != 3:
            raise IndexError('vidshow only knows how to display 3-channel frames')
    else:
        raise IndexError('vidshow requires a 3D or 4D numpy array')
    
    frames = frames[start_frame:end_frame]
    
    plt.gray()
    im = plt.imshow(frames[0], **kwargs)
    for frame_num, frame in enumerate(frames):
        im.set_data(frame)        
        plt.pause(1./fps)
    plt.show()
    
    
def flip_dim(a, axis=0): 
    # like numpy.fliplr or numpy.flipud but works on arbitrary dimension
    idx = [slice(None)]*len(a.shape)
    idx[axis] = slice(None, None, -1)
    return a[idx]


def tiff_to_ndarray(fn):
    """
    Load a tiff stack as 3D numpy array.
    You must have enough RAM to hold the whole movie in memory.
    """
    return tifffile.imread(fn)


def imshow_overlay(im, mask, alpha=0.5, color='red', **kwargs):
    """Show semi-transparent red mask over an image"""
    mask = mask > 0
    mask = ma.masked_where(~mask, mask)        
    plt.imshow(im, **kwargs)
    plt.imshow(mask, alpha=alpha, cmap=ListedColormap([color]))

#
#class AviReader:
#
#    """Read a file as an immutable, iterable, sliceable  sequence of frames."""
#
#    def __init__(self, fn):
#        self.cap = cv2.VideoCapture(fn)
#        self.first_frame = 0
#        self.last_frame = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
#        self.fn = fn
#        self.num_frames = len(self)
#        self.frame_rate = int(self.cap.get(cv2.CAP_PROP_FPS))
#        self.frame_size = (self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT),
#                           self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#
#    def __len__(self):
#        return self.last_frame - self.first_frame
#
#    def __iter__(self):
#        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.first_frame)
#        return self
#
#    def next(self):
#        current_frame = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
#        if current_frame >= self.last_frame:
#            raise StopIteration
#        else:
#            _, frame = self.cap.read()
#            return frame
#
#    def __str__(self):
#        repr_str = 'AviReader instance from '+self.fn+': '
#        repr_str += str(len(self))+' frames of shape '+str(self.frame_size())
#        repr_str += ', ' +str(self.frame_rate())+' fps'
#        return repr_str
#
#    def __getitem__(self, index):
#        # FIXME: doesn't handle step (stride), nor negative slice indices
#        if isinstance(index, int):  # single frame
#            if index < 0:
#                index = len(self) + index
#            if index + self.first_frame > self.last_frame:
#                raise IndexError
#            self.cap.set(cv2.CAP_PROP_POS_FRAMES, index + self.first_frame)
#            ok, frame = self.cap.read()
#            if ok:
#                return frame
#            else:
#                raise IndexError
#        elif isinstance(index, slice):  # slice
#            self.first_frame = index.start
#            # if index.stop is not None:  # FIXME: why doesn't this work with [sta:] indexing?
#            if index.stop <= self.last_frame:
#                self.last_frame = index.stop
#            return self
#        else:
#            raise TypeError('Avi indices should be integer or slices')



class TifReader:

    """An immutable, iterable sequence of frames."""

    def __init__(self, fn):
        self.fn = fn
        self.im = Image.open(fn)
        self.first_frame = 0
        idx = 0
        while True:        
            try:
                self.im.seek(idx)
            except EOFError:
                self.last_frame = idx
                break
            idx += 1
        self.num_frames = self.last_frame
        self._total_frames = self.num_frames
        self.frame_size = self.im.size

    def __len__(self):
        return self.last_frame - self.first_frame

    def __iter__(self):
        self.iter_frame = 0
        return self

    def next(self):
        if self.iter_frame + self.first_frame >= self.last_frame:
            raise StopIteration
        else:
            self.im.seek(self.first_frame + self.iter_frame)
            self.iter_frame += 1        
        return np.array(self.im)

    def __str__(self):
        repr_str = self.__class__.__name__+' instance from '+self.fn+': '
        repr_str += str(len(self))+' frames of shape '+str(self.frame_size)
        return repr_str

    def __getitem__(self, index):
        # FIXME: doesn't handle step (stride), nor slicing w negative indices
        if isinstance(index, int):  # single frame
            if index < 0:
                index = len(self) + index
            if index + self.first_frame > self.last_frame:
                raise IndexError
            self.im.seek(index + self.first_frame)
            return np.array(self.im)
        elif isinstance(index, slice):  # slice
            self.first_frame = index.start
            # if index.stop is not None:  # FIXME: why doesn't this work with [sta:] indexing?
            if index.stop <= self.last_frame:
                self.last_frame = index.stop
            return self
        else:
            raise TypeError('Avi indices should be integer or slices')
    
    @property
    def shape(self):
        sz = self[0].shape
        return (sz[0], sz[1], len(self))
    


def write_video(frames, filename, fps=20):
    """ 
    Uses avconv to write a 3D numpy array to a video file. 
    Currently only supports grayscale arrays.    
    """
    
    # On Mac systems, copy ffmeg binaries to your PATH (http://ffmpegmac.net/)
    
    if platform.system() == 'Windows':
        err_str = 'Don\'t know how to write a movie for %s platform' % platform.system()
        raise NotImplementedError(err_str)

    
    if len(frames.shape) == 4:
        pix_fmt = 'rgb24'
    else:
        pix_fmt = 'gray'
    
    # normalize
    max_pix_val = np.percentile(frames, 99.9)
    if frames.dtype in (np.bool, bool):
        frames = frames.astype(np.uint8)
    frames -= frames.min()
    frames[frames>max_pix_val] = max_pix_val
    if max_pix_val > 0:
            frames *= 255. / max_pix_val
    frames = frames.astype(np.uint8)
    
    # figure out which av program is installed
    program_name = ''
    try:
        subprocess.check_call(['avconv', '-h'], stdout=DEVNULL, stderr=DEVNULL)
        program_name = 'avconv'
    except OSError:
        try:
            subprocess.check_call(['ffmpeg', '-h'], stdout=DEVNULL, stderr=DEVNULL)
            program_name = 'ffmpeg'
        except OSError:
            pass
    if not program_name:
        raise OSError('Can\'t find avconv or ffmpeg')
    
    # prepare pipe to av converter program
    size_str = '%ix%i' % (frames.shape[1], frames.shape[2])
    cmd = [program_name,
            '-y', # (optional) overwrite output file if it exists
            '-f', 'rawvideo',
            '-vcodec','rawvideo',
            '-s', size_str, # size of one frame
            '-pix_fmt', pix_fmt,
            '-r', str(fps), # frames per second
            '-i', '-', # input comes from a pipe
            '-an',     # no audio
            '-qscale', '1',
            '-vcodec','mjpeg',
            filename]
    
    pipe = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=DEVNULL, stderr=subprocess.STDOUT)
    
    # write frames            
    for frame in frames:
        frame = np.fliplr(frame)
        pipe.stdin.write(frame.tostring())
    pipe.stdin.close()
    pipe.wait()
    

def label_im_to_color(im, cmap='jet'):
    im = im.astype(float)
    im -= np.min(im)
    im /= np.max(im)
    cmap = plt.cm.get_cmap(cmap)
    return cmap(im)



class KalmanSmoother2D:
    
    def __init__(self, x_noise, y_noise, smoothness_x=1, smoothness_y=1):
        
        dt = 1
        
        # model
        F = np.eye(4)
        F[0, 2] = dt
        F[1, 3] = dt

        H = np.zeros((2, 4))
        H[0, 0] = 1
        H[1, 1] = 1

        R = np.zeros((2, 2))
        R[0, 0] = x_noise * x_noise
        R[1, 1] = y_noise * y_noise

        sigma_ax, sigma_ay = 1, 1
        G = np.zeros((4, 1))
        G[2] = sigma_ax*dt
        G[3] = sigma_ay*dt

        Q = np.transpose(G)*G
        Q[0, 1] = 0; Q[1, 0] = 0
        Q[0, 3] = 0; Q[3, 0] = 0
        Q[1, 2] = 0; Q[2, 1] = 0
        Q[2, 3] = 0; Q[3, 2] = 0

        # initialize filter
        self.kf = KalmanFilter()
        self.kf.transition_matrices = F
        self.kf.observation_matrices = H
        self.kf.transition_covariance = Q
        self.kf.observation_covariance = R

        # default initial state
        # TODO: maybe use first measurement as default?
        self.kf.initial_state_mean = np.zeros((4,))
        self.kf.initial_state_covariance = np.zeros((4, 4))
        
    # TODO: get innovations?
        
    def set_initial_state(self, initial_mean, initial_covariance=np.zeros((4,4))):
        if initial_mean.shape[0] == 2:
            print('initial velocity unspecified, assuming v0 = 0')
            initial_mean = np.array([initial_mean[0], initial_mean[1], 0, 0])
        self.kf.initial_state_mean = initial_mean
        self.kf.initial_state_covariance = initial_covariance
        
    def set_measurements(self, measurements):
        self.smooth_means, self.smooth_covs = self.kf.smooth(measurements)
        
    def get_smoothed_measurements(self):
        return self.smooth_means[:,0:2]
    
    def get_velocities(self):
        return self.smooth_means[:,2:]
        
    def get_covariances(self):
        return self.smooth_covs


def gray2rgb(im):
    im = im.astype(np.float)
    im /= im.max()
    im = np.round(255*im)
    im = im.astype(np.uint8)
    
    w, h = im.shape
    ret = np.empty((w, h, 3), dtype=np.uint8)
    ret[:, :, 0] = im
    ret[:, :, 1] = im
    ret[:, :, 2] = im
    return ret
    

def enhance_ridges(frame, mask=None):
    """Detect ridges (larger hessian eigenvalue)"""
    blurred = filters.gaussian_filter(frame, 2)
    Hxx, Hxy, Hyy = feature.hessian_matrix(blurred, sigma=4.5, mode='nearest')
    ridges = feature.hessian_matrix_eigvals(Hxx, Hxy, Hyy)[0]

    return np.abs(ridges)



def mask_to_boundary_pts(mask, pt_spacing=10):
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

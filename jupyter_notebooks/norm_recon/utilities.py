import math
import dxchange
import numpy as np
import pandas as pd
from scipy.signal import medfilt2d
import matplotlib.pyplot as plt
import cv2 as cv
import os
import glob
import tomopy
import random
import h5py as h5f
# import svmbir
from tqdm import tqdm
import bm3d_streak_removal as bm3d
from bm3d_streak_removal_cuda import multiscale_streak_removal, normalize_data, extreme_streak_attenuation, default_horizontal_bins, full_streak_pipeline
from skimage.measure import block_reduce
from PIL import Image
import timeit
import algotom.rec.reconstruction as rec

# slit_box_corners = np.array([[ None,  None], [ None, None], [None, None], [None,  None]])

BM_eff = 7.59e-6
BM_eff_wave = 1.8
pix_bin_size_default = 1
pix_bin_func_default = np.sum
pix_bin_dtype_default = np.uint16
img_per_ang_default = 1

def get_deg(fname: str):
    _split = fname.split('_')
    ang = _split[-3] + '.' + _split[-2]
    return ang


def get_fname_df(name_list: list, golden_ratio=False):
    ind = []
    ang_deg = []
    ang_rad = []
    fname_df = pd.DataFrame()
    for e_name in name_list:
        _split = e_name.split('_')
        _index_tiff = _split[-1]
        _index = _index_tiff.split('.')[0]
        if golden_ratio:
            _ang = _split[-4] + '.' + _split[-3]
        else:
            _ang = _split[-3] + '.' + _split[-2]
        index = int(_index)
        angle = float(_ang)
        ind.append(index)
        ang_deg.append(angle)
        ang_rad.append(math.radians(angle))
    fname_df['fname'] = name_list
    fname_df['ang_deg'] = ang_deg
    fname_df['ang_rad'] = ang_rad
    fname_df['idx'] = ind
    return fname_df

def group_fname_df(name_list: list, scan_dir:str):
    ind = []
    sample = []
    position = []
    fpath_list = []
    fname_df = pd.DataFrame()
    for e_name in name_list:
        _split = e_name.split('_')
        _index_tiff = _split[-1]
        _index = _index_tiff.split('.')[0]
        _sample = _split[0] + '_' + _split[1]
        _position = _split[2]
        _fpath = os.path.join(scan_dir, e_name)
        index = int(_index)
        ind.append(index)
        sample.append(_sample)
        position.append(_position)
        fpath_list.append(_fpath)
    fname_df['fname'] = name_list
    fname_df['sample'] = sample
    fname_df['position'] = position
    fname_df['idx'] = ind
    fname_df['fpath'] = fpath_list
    return fname_df


def get_exposure_list(fname_list: list):
    exposure = []
    for e_name in fname_list:
        _split = e_name.split('_')
        _exposure = '_' + _split[-2] + '_'
        exposure.append(_exposure)
    return sorted(list(set(exposure)))

def get_name_list(fname_list: list):
    name = []
    for e_name in fname_list:
        _split = e_name.split('_')
        _name = '_'.join(_split[0:-2])
        name.append(_name)
    return sorted(list(set(name)))

def get_list_by_idx(name_list: list, golden_ratio=False):
    fname_df = get_fname_df(name_list, golden_ratio)
    fname_df.sort_values('idx', inplace=True)
    fname = fname_df['fname'].to_list()
    ang_deg = fname_df['ang_deg'].to_list()
    ang_rad = fname_df['ang_rad'].to_list()
    ind = fname_df['idx'].to_list()
    return fname, ang_deg, ang_rad, ind


def get_list_by_ang(name_list: list, golden_ratio=False):
    fname_df = get_fname_df(name_list, golden_ratio)
    fname_df.sort_values('ang_deg', inplace=True)
    fname = fname_df['fname'].to_list()
    ang_deg = fname_df['ang_deg'].to_list()
    ang_rad = fname_df['ang_rad'].to_list()
    ind = fname_df['idx'].to_list()
    return fname, ang_deg, ang_rad, ind


# def get_fname_ind_str(name_list: list):
#     ind = []
#     ind_dict_random = {}
#     ind_dict_sorted = {}
#     for e_name in name_list:
#         _split = e_name.split('_')
#         _index_tiff = _split[-1]
#         _index = _index_tiff.split('.')[0]
#         index = int(_index)
#         ind.append(index)
#         ind_dict_random[index] = e_name
#     ind = sorted(ind)
#     for n, e_ind in enumerate(ind):
#         ind_dict_sorted[n] = ind_dict_random[e_ind]

#     return list(ind_dict_sorted.values()), ind

def get_idx_num(fname: str):
    _split = fname.split('_')
    _index_tiff = _split[-1]
    _index = _index_tiff.split('.')[0]
    index = int(_index)
    return index

def sort_by_idx(fname_list: list):
    ind = []
    ind_dict_random = {}
    ind_dict_sorted = {}
    for e_name in fname_list:
        index = get_idx_num(e_name)
        ind.append(index)
        ind_dict_random[index] = e_name
    ind = sorted(ind)
    for n, e_ind in enumerate(ind):
        ind_dict_sorted[n] = ind_dict_random[e_ind]

    return list(ind_dict_sorted.values())

def get_list(name_list: list):
    ind = range(len(name_list))
    return sorted(name_list), ind

def hdf5_to_sample_name(hdf5_name:str):
    _name_str_list = hdf5_name.split('_')
    _name_str_list.pop(-1)
    _name_str_list.pop(-1)
    sample_name = '_'.join(_name_str_list)
    return sample_name

def _init_arr_from_stack(fname, number_of_files, slc=None, pixel_bin_size=pix_bin_size_default, func=pix_bin_func_default, dtype=pix_bin_dtype_default):
    """
    Initialize numpy array from files in a folder.
    """
    if fname.split('.')[-1] == 'fits':
        _arr = dxchange.read_fits(fname)
        f_type = 'fits'
    elif fname.split('.')[-1] in 'tiff':
        _arr = dxchange.read_tiff(fname, slc)
        f_type = 'tif'
    else:
        raise ValueError("'{}', only '.tif/.tiff' and '.fits' are supported.".format(fname))
    if pixel_bin_size > 1:
        _arr = bin_pix(_arr, pixel_bin_size=pixel_bin_size, func=func, dtype=dtype)
    size = (number_of_files, _arr.shape[0], _arr.shape[1])
    return np.empty(size, dtype=_arr.dtype), f_type


def read_tiff_stack(fdir, fname: list, pixel_bin_size=pix_bin_size_default, func=pix_bin_func_default, dtype=pix_bin_dtype_default):
    arr, f_type = _init_arr_from_stack(os.path.join(fdir, fname[0]), len(fname), pixel_bin_size=pixel_bin_size, func=func, dtype=dtype)
    print(len(fname))
    for m, name in tqdm(enumerate(fname), leave=False):
        _arr = dxchange.read_tiff(os.path.join(fdir, name))
        if pixel_bin_size > 1:
            _arr = tomopy.misc.corr.remove_outlier(_arr, 20).astype(np.ushort) # apply gamma filter before pixel bin
            _arr = bin_pix(_arr, pixel_bin_size=pixel_bin_size, func=func, dtype=dtype)
        arr[m] = _arr[:]
    return arr

def read_tiff_stack_wo_tqdm(fdir, fname: list, pixel_bin_size=pix_bin_size_default, func=pix_bin_func_default, dtype=pix_bin_dtype_default):
    arr, f_type = _init_arr_from_stack(os.path.join(fdir, fname[0]), len(fname), pixel_bin_size=pixel_bin_size, func=func, dtype=dtype)
    for m, name in enumerate(fname):
        _arr = dxchange.read_tiff(os.path.join(fdir, name))
        if pixel_bin_size > 1:
            _arr = tomopy.misc.corr.remove_outlier(_arr, 20).astype(np.ushort) # apply gamma filter before pixel bin
            _arr = bin_pix(_arr, pixel_bin_size=pixel_bin_size, func=func, dtype=dtype)
        arr[m] = _arr[:]
    return arr

def read_img_stack(fdir, fname: list, fliplr=False, flipud=False):
    arr, f_type = _init_arr_from_stack(os.path.join(fdir, fname[0]), len(fname))
    if f_type == 'tif':
        for m, name in tqdm(enumerate(fname)):
            _arr = dxchange.read_tiff(os.path.join(fdir, name))
            if fliplr:
                _arr = np.fliplr(_arr)
            if flipud:
                _arr = np.flipud(_arr)
            arr[m] = _arr
    elif f_type == 'fits':
        for m, name in tqdm(enumerate(fname)):
            _arr = dxchange.read_fits(os.path.join(fdir, name))
            if fliplr:
                _arr = np.fliplr(_arr)
            if flipud:
                _arr = np.flipud(_arr)
            arr[m] = _arr
    return arr

def load_tiff(file_name):
    """load tiff image
    Parameters:
    -----------
       full file name of tiff image
    """
    try:
        _image = Image.open(file_name)
        metadata = dict(_image.tag_v2)
        data = np.asarray(_image)
        _image.close()
        return [data, metadata]
    except OSError as e:
        raise OSError(f"Unable to read the TIFF file provided!: {e}")

def make_tiff(data=[], metadata=[], file_name=""):
    """create tiff file"""
    new_image = Image.fromarray(data)
    new_image.save(file_name, tiffinfo=metadata)

def restore_metadata(src_fname, tgt_fname):
    _src = load_tiff(src_fname)
    _src_metadata = _src[1]
    _tgt = load_tiff(tgt_fname)
    _tgt_data = _tgt[0]
    make_tiff(data=_tgt_data, metadata=_src_metadata, file_name=tgt_fname)

# def restore_metadata_dir(src_dir, tgt_dir):
    

def find_proj180_ind(ang_list: list):
    dif = [abs(x - 180) for x in ang_list]
    difmin = min(dif)
    ind180 = dif.index(difmin)
    return (ind180, ang_list[ind180])


def find_idx_by_ang(ang_list: list, ang):
    dif = [abs(x - ang) for x in ang_list]
    difmin = min(dif)
    ind = dif.index(difmin)
    return (ind, ang_list[ind])


def shrink_window(corners, size):
    corners[0][0] = corners[0][0] + size
    corners[0][1] = corners[0][1] + size
    corners[1][0] = corners[1][0] + size
    corners[1][1] = corners[1][1] - size
    corners[2][0] = corners[2][0] - size
    corners[2][1] = corners[2][1] - size
    corners[3][0] = corners[3][0] - size
    corners[3][1] = corners[3][1] + size
    return corners


def set_roi(corners, xmin, ymin, xmax, ymax):
    corners[0][0] = xmin
    corners[0][1] = ymin
    corners[1][0] = xmin
    corners[1][1] = ymax
    corners[2][0] = xmax
    corners[2][1] = ymax
    corners[3][0] = xmax
    corners[3][1] = ymin
    return corners


################################################ Added on 8/24/2022

def is_routine_ct(ct_dir):
    re_list = []
    for each in ["raw*", "ct*", "ob*", "OB*", "dc*", "DC*", "df*", "DF*"]:
        re = len(glob.glob(ct_dir + "/" + each)) == 0
        re_list.append(re)
    if False in re_list:
        return False
    else:
        return True


def get_name_and_idx(fdir):
    fname_list = os.listdir(fdir)
    fname, idx_list = get_list(fname_list)
    return fname, idx_list


def load_ct(fdir, ang1=0, ang2=360, name="raw*", filter_name=None, pixel_bin_size=pix_bin_size_default, func=pix_bin_func_default, dtype=pix_bin_dtype_default, img_per_ang=img_per_ang_default, mars_ct=True, golden_ratio=False):
    if mars_ct:
        print("Normal CT naming convention")
        ct_list = os.listdir(fdir)
        if filter_name is not None:
            ct_list = filter_list(ct_list, filter_name)
        # ct_name, ang_deg, theta, idx_list = get_ind_list(ct_list)
#         if img_per_ang >1:
#             ct_list = ct_list[num_img_per_ang-1::num_img_per_ang]
        ct_name, ang_deg, ang_rad, idx_list = get_list_by_ang(ct_list, golden_ratio)
        img_per_ang = ang_deg.count(ang_deg[0])
    else:
        print("Different CT naming convention, please verify 'img_per_ang = {}'".format(img_per_ang))
        ct_list = glob.glob(fdir + "/" + name)
        ct_name, idx_list = get_list(ct_list)
        ang_rad = tomopy.angles(len(idx_list), ang1=ang1, ang2=ang2)  # Default 360 degree rotation
        ang_deg = np.rad2deg(ang_rad)
    
    proj = read_tiff_stack(fdir=fdir, fname=ct_name, pixel_bin_size=pixel_bin_size, func=func, dtype=dtype)
    if img_per_ang > 1:
        print("{} images per angle, image stack binning...".format(img_per_ang))
        ct_name = ct_name[img_per_ang-1::img_per_ang]
        ang_deg = ang_deg[img_per_ang-1::img_per_ang]
        ang_rad = ang_rad[img_per_ang-1::img_per_ang]
        idx_list = idx_list[img_per_ang-1::img_per_ang]
        if img_per_ang < 3:
            func_stack = np.mean
            dtype_stack = np.float16
        else:
            func_stack = np.median
            dtype_stack = np.uint16
        proj = block_reduce(proj, block_size=(img_per_ang, 1, 1), func=func_stack)#, func_kwargs={'dtype': dtype_stack})
    proj360_ind = find_idx_by_ang(ang_deg, 360)
    proj180_ind = find_idx_by_ang(ang_deg, 180)
    proj000_ind = find_idx_by_ang(ang_deg, 0)
    print('Found index of 180 degree projections: {} of angle {}'.format(proj180_ind[0], proj180_ind[1]))
    print('Found index of 0 degree or first projections: {} of angle {}'.format(proj000_ind[0], proj000_ind[1]))
    print('Found index of 360 degree or last projections: {} of angle {}'.format(proj360_ind[0], proj360_ind[1]))
    print('Loading {} CT projections...'.format(len(ct_name)))
    print('{} CT projections loaded!'.format(len(ct_name)))
    print('Shape: {}'.format(proj.shape))
    return proj, ang_deg, ang_rad, proj000_ind[0], proj180_ind[0], proj360_ind[0], ct_name


# def load_ob(fdir, name="ob*"):
#     if is_routine_ct(fdir):
#         ob_name, idx_list = get_name_and_idx(fdir)
#         print("Normal CT naming convention")
#     else:
#         ob_list = glob.glob(fdir + "/" + name)
#         ob_name, idx_list = get_list(ob_list)
#         print("Different CT naming convention")
#     print("Loading {} Open Beam (OB) images...".format(len(ob_name)))
#     ob = read_tiff_stack(fdir=fdir, fname=ob_name)
#     print("{} Open Beam (OB) images loaded!".format(len(ob_name)))
#     print('Shape: {}'.format(ob.shape))
#     return ob


# def load_dc(fdir, name="dc*"):
#     if is_routine_ct(fdir):
#         dc_name, idx_list = get_name_and_idx(fdir)
#     else:
#         dc_list = glob.glob(fdir + "/" + name)
#         dc_name, idx_list = get_list(dc_list)
#     print("Loading {} Dark Current (DC) images...".format(len(dc_name)))
#     dc = read_tiff_stack(fdir=fdir, fname=dc_name)
#     print("{} Dark Current (DC) images loaded!".format(len(dc_name)))
#     print('Shape: {}'.format(dc.shape))
#     return dc

def load_ob(fdir, name="OB*", pixel_bin_size=pix_bin_size_default, func=pix_bin_func_default, dtype=pix_bin_dtype_default):
    if is_routine_ct(fdir):
        ob_name, idx_list = get_name_and_idx(fdir)
    else:
        ob_list = glob.glob(fdir + "/" + name)
        ob_name, idx_list = get_list(ob_list)
    print("Loading {} Open Beam (OB) images...".format(len(ob_name)))
    ob = read_tiff_stack(fdir=fdir, fname=ob_name, pixel_bin_size=pixel_bin_size, func=func, dtype=dtype)
    print("{} Open Beam (OB) images loaded!".format(len(ob_name)))
    print('Shape: {}'.format(ob.shape))
    return ob


def load_dc(fdir, name="DC*", pixel_bin_size=pix_bin_size_default, func=pix_bin_func_default, dtype=pix_bin_dtype_default):
    if is_routine_ct(fdir):
        dc_name, idx_list = get_name_and_idx(fdir)
    else:
        dc_list = glob.glob(fdir + "/" + name)
        dc_name, idx_list = get_list(dc_list)
    print("Loading {} Dark Current (DC) images...".format(len(dc_name)))
    dc = read_tiff_stack(fdir=fdir, fname=dc_name, pixel_bin_size=pixel_bin_size, func=func, dtype=dtype)
    print("{} Dark Current (DC) images loaded!".format(len(dc_name)))
    print('Shape: {}'.format(dc.shape))
    return dc


##########################

# def load_static(fdir, name="dc*", diff="20"):
#     if is_routine_ct(fdir):
#         dc_name, idx_list = get_name_and_idx(fdir)
#     else:
#         dc_list = glob.glob(fdir + "/" + name)
#         dc_name, idx_list = get_list(dc_list)
#     dc = read_tiff_stack(fdir=fdir, fname=dc_name)
#     if dc.shape[0] == 1:
#         dc_med = dc[:]
#         print("Only 1 file loaded.")
#     else:
#         dc = tomopy.misc.corr.remove_outlier(dc, diff)
#         dc_med = np.median(dc, axis=0).astype(np.ushort)
#     return dc_med


##########################

# def remove_ring(proj, algorithm="Vo"):
#     if algorithm == "Vo":
#         proj_rmv = tomopy.prep.stripe.remove_all_stripe(proj)
#     elif algorithm == "bm3d":
#         proj_norm = bm3d_rmv.extreme_streak_attenuation(proj)
#         proj_rmv = bm3d_rmv.multiscale_streak_removal(proj_norm)
#     return proj_rmv


# def recon(proj, theta, rot_center, algorithm="gridrec"):
#     if algorithm == "svMBIR":
#         # T, P, sharpness, snr_db: parameters of reconstruction, usually keep fixed. (Can be played with)
#         T = 2.0
#         p = 1.2
#         sharpness = 0.0
#         snr_db = 30.0
#         center_offset= -(proj.shape[2]/2 - rot_center)
#         recon = svmbir.recon(proj, angles=theta, weight_type='transmission',
#                              center_offset=center_offset,
#                              snr_db=snr_db, p=p, T=T, sharpness=sharpness,
#                              positivity=False, max_iterations=100,
#                              num_threads= 112, verbose=0) # verbose: display of reconstruction: 0 is minimum, 1 is regular
#     else:
#         recon = tomopy.recon(proj, theta, center=rot_center, algorithm=algorithm, sinogram_order=False)
#     recon = tomopy.circ_mask(recon, axis=0, ratio=1)
#     return recon

################################################ Added on 11/01/2022

def filter_list(name_list:list, pattern=None):
    if pattern is not None:
        filtered_list = []
        for _e in name_list:
            if pattern in _e:
                filtered_list.append(_e)
    else:
        filtered_list = name_list[:]
    return filtered_list


def add_idx_to_front(old: str, index_min=0):
    old_index = get_idx_num(old)
    new_idx_num = old_index - index_min
    new_idx_str = f'{new_idx_num:04}'
    new = new_idx_str + "_" + old
    return new


def get_last_str(fname: str):
    _split = fname.split('_')
    idx_tiff = _split[-1]
    _idx_tiff_split = idx_tiff.split('.')
    idx = _idx_tiff_split[0]
    return idx


def remove_1st_str(fname: str):
    _split = fname.split('_')
    _split.pop(0)
    new_fname = "_".join(_split)
    return new_fname


def remove_last_str(fname: str):
    _split = fname.split('_')
    last_ext = _split[-1]
    _last_ext_split = last_ext.split('.')
    ext = _last_ext_split[-1]
    _split.pop(-1)
    new_fname = "_".join(_split) + '.' + ext
    return new_fname


def normalize(proj, ob, dc):
    assert len(proj.shape) <= 3
    assert len(ob.shape) <= 3
    if dc is not None:
        assert len(dc.shape) <= 3
    proj_mi_dc, ob_mi_dc, dc_med = subtract_dc(proj, ob, dc)
    proj_norm = np.true_divide(proj_mi_dc, ob_mi_dc, dtype=np.float32)
    print("Normalization Done!")
    return proj_norm, proj_mi_dc, ob_mi_dc, dc_med

def subtract_dc(proj, ob, dc):
    if len(ob.shape) == 2:
        ob_med = ob[:]
        print("Only 1 OB loaded.")
    elif len(ob.shape) == 3:
        if ob.shape[0] == 1:
            ob_med = np.squeeze(ob, axis=0)
            print("OB squeezed.")
            print("Only 1 OB loaded.")
        else:
            ob_med = np.median(ob, axis=0).astype(np.ushort)
            print("OB stack combined by median.")
    if dc is not None:
        if len(dc.shape) == 2:
            dc_med = dc[:]
            print("Only 1 DC loaded.")
        elif len(dc.shape) == 3:
            if dc.shape[0] == 1:
                dc_med = np.squeeze(dc, axis=0)
                print("DC squeezed.")
                print("Only 1 DC loaded.")
            else:
                dc_med = np.median(dc, axis=0).astype(np.ushort)
                print("DC stack combined by median.")
        ob_out = ob_med - dc_med
        proj_out = proj - dc_med
        dc_out = dc_med[:]
    else:
        ob_out = ob_med[:]
        proj_out = proj[:]
        dc_out = None
    return proj_out, ob_out, dc_out

def crop(stack, crop_left, crop_right, crop_top, crop_bottom, crop=True):
    if len(stack.shape) == 3:
        if crop:
            new_stack = stack[:, crop_top:crop_bottom, crop_left:crop_right]
        else:
            new_stack = stack[:]
    elif len(stack.shape) == 2:
        if crop:
            new_stack = stack[crop_top:crop_bottom, crop_left:crop_right]
        else:
            new_stack = stack[:]
    else:
        print("Not a image, no cropping is done")
        new_stack = None
    return new_stack


def log(history: dict, event: str, info):
    history[event] = info
    return history

def show_progress(current_iteration, end_iteration):
    print(f"Registering {current_iteration} of {end_iteration} images")

def composite_images(imgs, equalize=False, aggregator=np.mean):

    if equalize:
        imgs = [exposure.equalize_hist(img) for img in imgs]

    imgs = [img / img.max() for img in imgs]

    if len(imgs) < 3:
        imgs += [np.zeros(shape=imgs[0].shape)] * (3-len(imgs))

    imgs = np.dstack(imgs)

    return imgs

def overlay_images(imgs, equalize=False, aggregator=np.mean):

    if equalize:
        imgs = [exposure.equalize_hist(img) for img in imgs]

    imgs = np.stack(imgs, axis=0)

    return aggregator(imgs, axis=0)

def remove_fnames(fname_list:list, to_rmv:list):
    print(len(fname_list))
    for each in to_rmv:
        fname_list.remove(each)
    print(len(fname_list))
    return fname_list

def find_txrm(loc:str, incl_xrm=False):
    fname_list = os.listdir(loc)
    print("Pool:", fname_list)
    txrm_list = []
    for ename in fname_list:
        if '.txrm' in ename:
            txrm_list.append(ename)
        if '.txm' in ename:
            txrm_list.append(ename)
        if incl_xrm:
            if '.xrm' in ename:
                txrm_list.append(ename)
    txrm_list = sorted(txrm_list)
    print("Found:", txrm_list)
    return txrm_list

def txm2tiff(path, fname):
    if '_Drift.txrm' in fname:
        name = 'drift'
        h5_name = "metadata_drift.h5"
        sub_dir = 'proj'
    elif '.txm' in fname:
        name = 'recon'
        h5_name = "metadata_recon.h5"
        sub_dir = 'recon'
    elif 'FrontScoutImage.xrm' in fname:
        name = 'FrontScout'
        h5_name = "metadata_FrontScout.h5"
        sub_dir = 'proj'
    elif 'SideScoutImage.xrm' in fname:
        name = 'SideScout'
        h5_name = "metadata_SideScout.h5"
        sub_dir = 'proj'
    elif 'reference_Side.xrm' in fname:
        name = 'ref_Side'
        h5_name = "metadata_reference_Side.h5"
        sub_dir = 'proj'
    else:
        name = 'proj'
        h5_name = "metadata_proj.h5"
        sub_dir = 'proj'
    fpath = os.path.join(path, fname)
    print("Loading: {}".format(fpath))
    data, metadata = dxchange.read_txrm(fpath)
    save_to = os.path.join(path, sub_dir)
    print("Saving to: {}".format(save_to))
    dxchange.write_tiff_stack(data, fname=save_to + "/" + name, overwrite=True, digit=4)
    metadata_hdf5 = save_to + "/" + h5_name
    if not os.path.exists(save_to):
        os.mkdir(save_to)
    with h5f.File(metadata_hdf5, mode='a') as f:
        f.create_group('tomo/info')
        f.create_dataset('tomo/info/metadata', data=(str(metadata),))
    return metadata

def bin_pix(img, pixel_bin_size=pix_bin_size_default, func=pix_bin_func_default, dtype=pix_bin_dtype_default):
    if pixel_bin_size > 1:
        if len(img.shape) == 2:
            _block_size = (pixel_bin_size, pixel_bin_size)
        elif len(img.shape) == 3:
            _block_size = (1, pixel_bin_size, pixel_bin_size)
        img_binned = block_reduce(img, block_size=_block_size, func=func, func_kwargs={'dtype': dtype})
        return img_binned
    else:
        print("Pixel_bin_size = 1, no binning")
        return img

def estimate_noise_free_sinogram(sino_in: np.ndarray):
    """
    Estimate the noise-free sinogram from a noisy sinogram using BM3D.
    """
    min_org, max_org = np.min(sino_in), np.max(sino_in)
    tmp = sino_in - np.median(sino_in, axis=0)
    tmp = medfilt2d(tmp, kernel_size=3)
    tmp = (tmp - np.min(tmp)) / (np.max(tmp) - np.min(tmp)) * (max_org - min_org) + min_org
    return tmp

def remove_ring(proj_mlog, ring_algo, ncore=None):
    if ring_algo == 'Vo':
        # nchunk = int(proj_mlog.shape[0]/ncore) + 1
        # print("Chunk size: ", nchunk)
        proj_mlog_rmv = tomopy.remove_all_stripe(proj_mlog, 
                                                 ncore=ncore, 
    #                                              nchunk=nchunk
                                                )
    if ring_algo == 'bm3d':
        print("Perform 'extreme streak attenuation' (detection + median filter) on a 3-D stack of projections. First dimension should be angle.")
        proj_mlog_bm3d = bm3d.extreme_streak_attenuation(proj_mlog)
        print("Remove sinogram (after log transform) streak noise using multi-scale BM3D-based denoising procedure.")
        proj_mlog_rmv = bm3d.multiscale_streak_removal(proj_mlog_bm3d)

    if ring_algo == 'bm3dgpu':
        print("Perform 'extreme streak attenuation' (detection + median filter) on a 3-D stack of projections. First dimension should be angle.")
        proj_mlog_bm3d = extreme_streak_attenuation(proj_mlog)
        print("Remove sinogram (after log transform) streak noise using multi-scale BM3D-based denoising procedure.")
        proj_mlog_rmv = multiscale_streak_removal(proj_mlog_bm3d)
    
    #if ring_algo == 'bm3dornl':
    #    sino_mlog_rmv = bm3d_ring_artifact_removal(sino_mlog_tilt[slice_num])
    
    if ring_algo is None:
        proj_mlog_rmv = proj_mlog[:]
    return proj_mlog_rmv

def remove_nan(array, val, ncore):
    if np.isnan(array).any():
        print("Found NaN")
        t0 = timeit.default_timer()
        array = tomopy.misc.corr.remove_nan(array, val=val, ncore=ncore)
        t1 = timeit.default_timer()
        print("Remove NaN Time: {} s".format(t1-t0))
    return array

def remove_neg(array, val, ncore):
    if array.any() < 0:
        print("Found Negtives")
        t0 = timeit.default_timer()
        array = tomopy.misc.corr.remove_neg(array, val=val, ncore=ncore)
        t1 = timeit.default_timer()
        print("Remove Negatives Time: {} s".format(t1-t0))
    return array

def remove_by_idx(idx_list:list, proj_raw, ang_deg, ang_rad, fname_sorted):
    for ea_idx in tqdm(idx_list):
        proj_raw = np.delete(proj_raw, ea_idx, axis=0)
        ang_deg.pop(ea_idx)
        ang_rad.pop(ea_idx)
        fname_sorted.pop(ea_idx)
    return proj_raw, ang_deg, ang_rad, fname_sorted

def remove_by_slicing(idx, proj_raw, ang_deg, ang_rad, fname_sorted):
    proj_raw = proj_raw[idx:]
    ang_deg = ang_deg[idx:]
    ang_rad = ang_rad[idx:]
    fname_sorted = fname_sorted[idx:]
    ang_deg = [round(ea_deg-ang_deg[0], 2) for ea_deg in ang_deg]
    ang_rad = [ea_rad-ang_rad[0] for ea_rad in ang_rad]
    return proj_raw, ang_deg, ang_rad, fname_sorted

def remove_zinger(sino_mlog):
    for z_idx, z_mlog in tqdm(enumerate(sino_mlog)):
        _sino_mlog_clean = rem.remove_zinger(z_mlog, 0.005, size=2)
        sino_mlog[z_idx] = _sino_mlog_clean
    return sino_mlog

def plot_image(img, figsize_in_inches=(5,5)):
    fig, ax = plt.subplots(figsize=figsize_in_inches)
    ax.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
    plt.show()
    
def plot_images(imgs, figsize_in_inches=(5,5)):
    fig, axs = plt.subplots(1, len(imgs), figsize=figsize_in_inches)
    for col, img in enumerate(imgs):
        axs[col].imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
    plt.show()

def plot_imgs_from_stack(img_stack, idx_list, vmin=0.8, vmax=1.2, fig_per_row=4, figsize=(20,20)):
    num_of_row = int(len(idx_list)/fig_per_row)
    if len(idx_list)%fig_per_row != 0:
        num_of_row = num_of_row + 1
    f, ax = plt.subplots(num_of_row, fig_per_row, figsize=figsize)
    for m, ea in enumerate(idx_list):
        _loc = int(m/fig_per_row)
        ax[_loc][m-(_loc*fig_per_row)].imshow(img_stack[ea], vmin=vmin, vmax=vmax)
        ax[_loc][m-(_loc*fig_per_row)].set_title('Index={}'.format(ea))

def generate_randint_list(num_of_ele, range_min, range_max):
    # Create an empty list
    _list = []
    # Generate 10 random numbers between 0 and 200
    for _ in range(num_of_ele):
        _list.append(random.randint(range_min, range_max))
    _list.sort()
    return _list

def recon_full_volume(proj_mlog_to_recon, rot_center, ang_rad, start_ang_idx, end_ang_idx, recon_algo, ncore, svmbir_path, pix_um=None, num_iter=100, apply_log=False):
    t0 = timeit.default_timer()
    ####################### tomopy algorithms (gridrec and fbp are faster than algotom) ##########################
    if recon_algo in ['art', 'bart', 'fbp', 'gridrec',
                      'mlem', 'osem', 'ospml_hybrid', 'ospml_quad',
                      'pml_hybrid', 'pml_quad', 'sirt', 'tv', 'grad', 'tikh']:
        recon = tomopy.recon(proj_mlog_to_recon[start_ang_idx:end_ang_idx,:,:], ang_rad[start_ang_idx:end_ang_idx], center=rot_center,
                             algorithm=recon_algo,
                             ncore=ncore, 
    #                          nchunk=nchunk
                            )
    ################################################ algotom algorithms ##########################################
        #### ASTRA
    if recon_algo in ['FBP', 'SIRT', 'SART', 'ART', 'CGLS', 'FBP_CUDA', 'SIRT_CUDA', 'SART_CUDA', 'CGLS_CUDA']:
        recon = rec.astra_reconstruction(proj_mlog_to_recon[start_ang_idx:end_ang_idx,:,:], 
                                         rot_center, 
                                         angles=ang_rad[start_ang_idx:end_ang_idx],
                                         apply_log=apply_log,
                                         method=recon_algo,
                                         ratio=1.0,
                                         filter_name='hann',
                                         pad=None,
                                         num_iter=num_iter,
                                         ncore=ncore
                                        )
        recon = np.moveaxis(recon, 1, 0) 
        #### gridrec from algotom
    if recon_algo == 'gridrec_algo':
        recon = rec.gridrec_reconstruction(proj_mlog_to_recon[start_ang_idx:end_ang_idx,:,:],
                                           rot_center, 
                                           angles=ang_rad[start_ang_idx:end_ang_idx], 
                                           apply_log=apply_log,
                                           ratio=1.0,
                                           filter_name='shepp',
                                           pad=100,
                                           ncore=ncore
                                          )
        recon = np.moveaxis(recon, 1, 0)
        #### FBP from algotom
    if recon_algo == 'fbp_algo':
        recon = rec.fbp_reconstruction(proj_mlog_to_recon[start_ang_idx:end_ang_idx,:,:], 
                                       rot_center, 
                                       angles=ang_rad[start_ang_idx:end_ang_idx], 
                                       apply_log=apply_log,
                                       ramp_win=None,
                                       filter_name='hann',
                                       pad=None,
                                       pad_mode='edge',
                                       ncore=ncore,
                                       gpu=False,
    #                                   gpu=True, block=(16, 16), # Version error 7.8, current version 7.5
                                      )
        recon = np.moveaxis(recon, 1, 0)
    ################################################### MBIR #####################################################
    if recon_algo == 'svmbir':
        T = 2.0
        p = 1.2
        sharpness = 0.0
        snr_db = 30.0
        center_offset= -(proj_mlog_to_recon.shape[2]/2 - rot_center)
        recon = svmbir.recon(
            proj_mlog_to_recon[start_ang_idx:end_ang_idx,:,:],
            angles=np.array(ang_rad)[start_ang_idx:end_ang_idx], # In radians
            weight_type='transmission', 
            center_offset=center_offset, 
            snr_db=snr_db, p=p, T=T, sharpness=sharpness, 
            positivity=False,
            max_iterations=num_iter,
            num_threads= 112,
            verbose=1,# verbose: display of reconstruction: 0 is minimum, 1 is regular
            svmbir_lib_path = svmbir_path,
        )
        recon = np.fliplr(np.rot90(recon, k=1, axes=(1,2)))
    ##################################
    if pix_um is not None:
        pix_cm = pix_um/10000
        recon = recon/pix_cm
    t1 = timeit.default_timer()
    print("Time cost {} min".format((t1-t0)/60))
    return recon

def recon_slice_by_slice(sino_to_recon, proj_mlog_to_recon, rot_center, ang_rad, start_ang_idx, end_ang_idx, recon_algo, ncore, svmbir_path, save_to, apply_log,
                         recon_crop=False, recon_crop_roi_dict=None, pix_um=None, num_iter=100):
    # Only run this cell if the previous one failed. This cell will recon and save slice by slice
    print('Slice by slice saving to: {}'.format(save_to))
    t0 = timeit.default_timer()
    for h_idx in tqdm(range(sino_to_recon.shape[0])):
        _rec_slice = recon_a_slice(sino_to_recon[h_idx,start_ang_idx:end_ang_idx,:], proj_mlog_to_recon[start_ang_idx:end_ang_idx,h_idx,:], 
                                   rot_center, ang_rad[start_ang_idx:end_ang_idx], recon_algo, ncore, svmbir_path, num_iter=num_iter, apply_log=apply_log)
        _rec_slice = crop(_rec_slice, recon_crop_roi_dict['left'], recon_crop_roi_dict['right'], recon_crop_roi_dict['top'], recon_crop_roi_dict['bottom'], recon_crop)
        if pix_um is not None:
            pix_cm = pix_um/10000
            _rec_slice = _rec_slice/pix_cm
        _slice_name = save_to + "/recon_" + f'{h_idx:05d}'
        dxchange.write_tiff(_rec_slice, fname=_slice_name, overwrite=True)
    t1 = timeit.default_timer()
    print("Time cost {} min".format((t1-t0)/60))

def recon_a_slice(sino_to_recon, proj_mlog_to_recon, rot_center, ang_rad, recon_algo, ncore, svmbir_path, apply_log, num_iter=100):
    if recon_algo == 'gridrec':
        _rec_slice = rec.gridrec_reconstruction(sino_to_recon, rot_center, angles=ang_rad, apply_log=apply_log,
                                                ncore=ncore,
                                                ratio=1.0,
                                                filter_name='shepp',
                                                pad=100,
                                               )
    if recon_algo == 'fbp':
        _rec_slice = rec.fbp_reconstruction(sino_to_recon, rot_center, angles=ang_rad, apply_log=apply_log,
                                            ncore=ncore,
                                            ramp_win=None,
                                            filter_name='hann',
                                            pad=None,
                                            pad_mode='edge',
                                            gpu=False,
                                            # gpu=True, block=(16, 16), # Version error 7.8, current version 7.5
                                           )
    if recon_algo in ['FBP', 'SIRT', 'SART', 'ART', 'CGLS', 'FBP_CUDA', 'SIRT_CUDA', 'SART_CUDA', 'CGLS_CUDA']:
        _rec_slice = rec.astra_reconstruction(sino_to_recon, rot_center, angles=ang_rad, apply_log=apply_log,
                                              method=recon_algo,
                                              num_iter=num_iter,
                                              ncore=ncore,
                                              ratio=1.0,
                                              filter_name='hann',
                                              pad=None,
                                             )
    if recon_algo == 'svmbir':
        T = 2.0
        p = 1.2
        sharpness = 0.0
        snr_db = 30.0
        center_offset= -(proj_mlog_to_recon.shape[2]/2 - rot_center)
        _rec_mbir = svmbir.recon(proj_mlog_to_recon,
                                  angles=np.array(ang_rad), # In radians
                                  weight_type='transmission', 
                                  center_offset=center_offset, 
                                  snr_db=snr_db, p=p, T=T, sharpness=sharpness, 
                                  positivity=False,
                                  max_iterations=num_iter,
                                  num_threads= 112,
                                  verbose=1,# verbose: display of reconstruction: 0 is minimum, 1 is regular
                                  svmbir_lib_path = svmbir_path
                                 )
        _rec_slice = np.flipud(np.rot90(_rec_mbir[0]))
    return _rec_slice
# def recon_a_slice(sino_mlog_to_recon, proj_mlog_to_recon, h_idx, rot_center, ang_rad, start_ang_idx, end_ang_idx, recon_algo, ncore, svmbir_path, num_iter=100, apply_log=False):
#     if recon_algo == 'gridrec':
#         _rec_slice = rec.gridrec_reconstruction(sino_mlog_to_recon[h_idx,start_ang_idx:end_ang_idx,:], rot_center, angles=ang_rad[start_ang_idx:end_ang_idx], apply_log=apply_log,
#                                                 ncore=ncore,
#                                                 ratio=1.0,
#                                                 filter_name='shepp',
#                                                 pad=100,
#                                                )
#     if recon_algo == 'fbp':
#         _rec_slice = rec.fbp_reconstruction(sino_mlog_to_recon[h_idx,start_ang_idx:end_ang_idx,:], rot_center, angles=ang_rad[start_ang_idx:end_ang_idx], apply_log=apply_log,
#                                             ncore=ncore,
#                                             ramp_win=None,
#                                             filter_name='hann',
#                                             pad=None,
#                                             pad_mode='edge',
#                                             gpu=False,
#                                             # gpu=True, block=(16, 16), # Version error 7.8, current version 7.5
#                                            )
#     if recon_algo in ['FBP', 'SIRT', 'SART', 'ART', 'CGLS', 'FBP_CUDA', 'SIRT_CUDA', 'SART_CUDA', 'CGLS_CUDA']:
#         _rec_slice = rec.astra_reconstruction(sino_mlog_to_recon[h_idx,start_ang_idx:end_ang_idx,:], rot_center, angles=ang_rad[start_ang_idx:end_ang_idx], apply_log=apply_log,
#                                               method=recon_algo,
#                                               num_iter=num_iter,
#                                               ncore=ncore,
#                                               ratio=1.0,
#                                               filter_name='hann',
#                                               pad=None,
#                                              )
#     if recon_algo == 'svmbir':
#         T = 2.0
#         p = 1.2
#         sharpness = 0.0
#         snr_db = 30.0
#         center_offset= -(proj_mlog_to_recon.shape[2]/2 - rot_center)
#         _rec_mbir = svmbir.recon(proj_mlog_to_recon[start_ang_idx:end_ang_idx,h_idx,:],
#                                   angles=np.array(ang_rad)[start_ang_idx:end_ang_idx], # In radians
#                                   weight_type='transmission', 
#                                   center_offset=center_offset, 
#                                   snr_db=snr_db, p=p, T=T, sharpness=sharpness, 
#                                   positivity=False,
#                                   max_iterations=num_iter,
#                                   num_threads= 112,
#                                   verbose=1,# verbose: display of reconstruction: 0 is minimum, 1 is regular
#                                   svmbir_lib_path = svmbir_path
#                                  )
#         _rec_slice = np.flipud(np.rot90(_rec_mbir[0]))
#     return _rec_slice
################ change save path for your own
# save_to = "/HFIR/CG1D/IPTS-"+ipts+"/shared/autoreduce/rockit/" + sample_name# + "_vo"
# save_to = "/HFIR/CG1D/IPTS-"+ipts+"/shared/processed_data/rockit/" + sample_name + "_all"
    
    

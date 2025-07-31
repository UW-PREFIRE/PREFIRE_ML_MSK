import sys
import os
import numpy as np
import netCDF4
import xarray as xr
import tensorflow as tf
from PREFIRE_PRD_GEN.file_creation import write_data_fromspec
from PREFIRE_PRD_GEN.file_read import load_all_atts_of_nc4group
import PREFIRE_PRD_GEN.bitflags as bitflags
import PREFIRE_ML_MSK.filepaths as ML_MSK_fpaths
from PREFIRE_tools.utils.bitflags import apply_bit_to_bitflags_v


_FillValue_int8 = -99
_FillValue_float32 = -9.999e3

class_thresholds = [0, 0.2, 0.4, 0.6, 0.8, 1.01]

class_labels = { 0 : "confident clear",
                 1 : "probable clear",
                 2 : "uncertain",
                 3 : "probable cloud",
                 4 : "confident cloud",
                }


def make_obs_masks(dsg_Radiance):
    #this is our radiance observation quality (original shape = (atrack,))
    tmp_shape = dsg_Radiance.radiance_quality_flag.shape

    tmp = np.broadcast_to(dsg_Radiance.observation_quality_flag[:],
                          (tmp_shape[1], tmp_shape[0]))
    obs_quality = np.transpose(tmp)
    return (obs_quality == 0, obs_quality == 1, obs_quality == 2)


def cloudprob2classlabel(cloudprob):
    #digitize assigns indices depending on bins, starts with index 1 notation, hence -1
    return np.digitize(cloudprob, bins = class_thresholds) - 1
    

def run_ML_cmask(L1bfile, AuxMetfile, modelfiles, CMASK_path, n_xtrack,
                 return_rather_than_write_dat=False):
    """
    
    Classifies scenes as clear versus cloudy using pre-trained neural networks
    cloudy is defined as scenes which have a column cloud optical depth > 0.25.
    Note that NNs are unique for each x-track
    
    Parameters
    ----------
    L1bfile : str
        Path to TIRS L1B netCDF file.
    AuxMetfile : str
        Path to corresponding Aux-Met netCDF file.
    modelfiles : list, shape = (n_xtrack,)
        A list of paths where the pretrained neural networks are stored
    CMASK_path : str
        A path to where the resulting mask file should be stored
    n_xtrack : int
        Number of cross-track scenes in a TIRS frame
    return_rather_than_write_dat : bool
        (optional) Return 'dat' instead of writing its values to a file?

    Returns
    -------
    None.

    """

    batch_size=10000

    with netCDF4.Dataset(L1bfile, 'r') as ds:
        sensor_num = int(ds.sensor_ID[-1])
        
    dsg_Radiance = xr.open_dataset(L1bfile, group="Radiance")
    
    ds_radiances = np.array(dsg_Radiance.spectral_radiance)
    
    detector_bf = dsg_Radiance.detector_bitflags.astype("uint16")
    # current channel subset used in training uses all detectors that are not
    # masked or marked as noisy. Unreliable calibration channels are still used.
    # So, remove any channel if any of bits (0,1,2,3) are set.
    ValidChanArr = np.full(detector_bf.shape, True, dtype=bool)
    check_bits = (0, 1, 2, 3)
    for s, c in np.ndindex(detector_bf.shape):
        bit_checks = [not bitflags.bit_meaning(detector_bf[s,c], b) for b in check_bits]
        ValidChanArr[s,c] = np.all(bit_checks)
        
    if sensor_num == 1:  # TIRS1
        # We will mask similar channels to those tested with TIRS2, however, additional FIR1 channel included
        # to cover same wavelength space as TIRS2
        #  (0-62 channel convention), to avoid striping issue found during IOC.
        ValidChanArr[:,:9] = False
        ValidChanArr[:,16:18] = False
        ValidChanArr[:,30:] = False
        
    else:  # TIRS2
        # (2024-07-11) update from BK: use only channels 9-15 & 18-28
        #  (0-62 channel convention), to avoid striping issue found during IOC.
        ValidChanArr[:,:9] = False
        ValidChanArr[:,16:18] = False
        ValidChanArr[:,29:] = False


    dsg_AuxMet = xr.open_dataset(AuxMetfile, group="Aux-Met")

    skinT = np.array(dsg_AuxMet.skin_temp)  # [K]
    tcwv = np.array(dsg_AuxMet.total_column_wv)  # [kg_H2O/m^2]
    
    cloud_probs= np.empty(skinT.shape) # cloud_probs contains the probability
                                       #  of there being a cloud
    N_ary_preds = np.empty(skinT.shape, dtype="int8") # assigned integer values
                                                      # depending on cloud_probs
    mask_qflag = np.full(skinT.shape, 0, dtype="int8")
    mask_qc_bitflags = np.full(skinT.shape, 0, dtype="uint16")

    #we must evaluate each xtrack independently
    for xx in range(n_xtrack):
        #what's required as inputs to our model
        radiances = ds_radiances[:,xx,ValidChanArr[xx,:]]
        radiances = np.hstack((radiances, skinT[:,xx].reshape(-1,1)))
        radiances = np.hstack((radiances,tcwv[:,xx].reshape(-1,1)))
        
        #the actual prediction by our NN
        model = tf.keras.models.load_model(modelfiles[xx])
        y_preds = model.predict(radiances, batch_size=batch_size)
        
        cloud_probs[:,xx]=y_preds[:,1]
    
    # Assigns 0-4 (confident clear to confident cloud) depending on cloud_probs
    N_ary_preds[...] = cloudprob2classlabel(cloud_probs)

    # Mask out bad radiance flags:
    best_obs_mask, uc_obs_mask, bad_obs_mask = make_obs_masks(dsg_Radiance)
    cloud_probs[bad_obs_mask] = _FillValue_float32
    N_ary_preds[bad_obs_mask] = _FillValue_int8
    mask_qflag[bad_obs_mask] = _FillValue_int8

    # Set bit0 ("based on best-quality radiances"):
    mask_qc_bitflags = apply_bit_to_bitflags_v(0, best_obs_mask,
                                               mask_qc_bitflags)

    # Set bit1 ("based on uncategorized radiances"):
    mask_qc_bitflags = apply_bit_to_bitflags_v(1, uc_obs_mask,
                                               mask_qc_bitflags)

    # Set bit2 ("cloud mask determination not attempted due to
    #            radiance_quality_flag value"):
    mask_qc_bitflags = apply_bit_to_bitflags_v(1, bad_obs_mask,
                                               mask_qc_bitflags)

    #writing out data
    # Nested dictionary -- will contain "Geometry" data from L1B file as dict
    dat = {}
    dat['Geometry'] = xr.open_dataset(L1bfile, group = 'Geometry')
    dat["Geometry_Group_Attributes"] = (load_all_atts_of_nc4group("Geometry",
                                                     netCDF4.Dataset(L1bfile)))

    cmask_data = {}
    cmask_data["cloud_mask"] = N_ary_preds
    cmask_data["cldmask_probability"] = cloud_probs
    cmask_data["msk_quality_flag"] = mask_qflag
    cmask_data["msk_qc_bitflags"] = mask_qc_bitflags

    dat['Msk'] = cmask_data
    tmp = [os.sep.join(x.split(os.sep)[-2:]) for x in modelfiles]
    dat["Msk_Group_Attributes"] = {"cldmask_NN_model_paths": ", ".join(tmp)}

    if return_rather_than_write_dat:
        return dat
    else:
        # Generate CMASK output file name
        tmp_fn = os.path.basename(L1bfile)
        L1B_types = ["1B-RAD", "1B-NLRAD", "1B-GEOM"]
        for this_type in L1B_types:
            if this_type in tmp_fn:
                L1B_type = this_type
                break
        CMASK_fname = tmp_fn.replace(L1B_type, "2B-MSK-ML")
        CMASK_fpath = os.path.join(CMASK_path, CMASK_fname)

        # Use generic PREFIRE NetCDF writer to produce CMASK output file:
        product_specs_fpath = os.path.join(
                                      ML_MSK_fpaths.package_ancillary_data_dir,
                                           "product_filespecs.json")
        write_data_fromspec(dat, CMASK_fpath, product_specs_fpath,
                            verbose=True)


if __name__=="__main__":

    #usage:
    # python ML_MSK.py L1bfile AuxMetfile ModelFileFullMoniker CMASK_path
    #

    L1bfile = sys.argv[1]
    AuxMetfile = sys.argv[2]

    # For this type of invocation, use model files in local ancillary dir:
    ModelFileLoc = os.path.join(ML_MSK_fpaths.package_ancillary_data_dir,
                                sys.argv[3])

    with netCDF4.Dataset(L1bfile, 'r') as ds:
        sensor_ID_str = ds.sensor_ID[-1]
        n_xtrack = ds.dimensions["xtrack"].size

    if os.path.isdir(ModelFileLoc):  # Assume this is a dir with multiple files
        ModelFiles = ([os.path.join(ModelFileLoc,
               f"SAT{sensor_ID_str}_xscene{x+1:1d}") for x in range(n_xtrack)])
    else:
        ModelFiles = [ModelFileLoc]*n_xtrack  # Assume this is a single filepath

    CMASK_path = sys.argv[4]

    run_ML_cmask(L1bfile, AuxMetfile, ModelFiles, CMASK_path, n_xtrack)

#!/usr/bin/env bash

## IMPORTANT: Only run this script from the directory it resides in, i.e. with
##             ./run.sh    OR    bash run.sh

##===========================================================================##
## This script contains hardwired information necessary for this algorithm's
##  delivery to and testing within the SDPS (Science Data Processing System).
##
## ** In general, do not push changes to this file to its primary git
##     repository (exceptions include adding a new environment var for
##     algorithm config) **
##
## ++ Instead, make a LOCAL copy of this script (e.g., my_run.sh; do not
##     push that local copy to the primary git repository either) and modify
##     and run that for general algorithm testing and development.
##===========================================================================##

absfpath() {
  # Generate absolute filepath from a relative (or even an absolute) filepath.
  #
  # Based on (circa Oct 2023) https://stackoverflow.com/questions/3915040/how-to-obtain-the-absolute-path-of-a-file-via-shell-bash-zsh-sh
  # 
  # $1     : a relative (or even an absolute) filepath
  # Returns the corresponding absolute filepath.
  if [ -d "$1" ]; then
    # dir
    (cd "$1"; pwd)
  elif [ -f "$1" ]; then
    # file
    if [[ $1 = /* ]]; then
      echo "$1"
    elif [[ $1 == */* ]]; then
      echo "$(cd "${1%/*}"; pwd)/${1##*/}"
    else
      echo "$(pwd)/$1"
    fi
  fi
}

activate_conda_env () {
  . "$1"/bin/activate;
}

deactivate_conda_env () {
  . "$1"/bin/deactivate;
}

set -ve;  # Exit on the first error, and print out commands as we execute them
#set -e;  # Exit on the first error

# Determine the absolute path of the current working directory:
#  (this is typically the package test/ directory)
readonly base_dir="$(absfpath ".")";

hn=`hostname -s`;  # Hostname

# NOTE: Set the input/output directories to absolute paths (relative to the
#        current working directory, 'base_dir').

non_SDPS_hostname="longwave";

input_dir="${base_dir}/inputs";

cfg_str1="ATRACK_IDXRANGE_0BASED_INCLUSIVE:0:END,${input_dir}/PREFIRE_SAT1_1B-RAD_R01_P00_20241007075724_01877.nc||${input_dir}/PREFIRE_SAT1_AUX-MET_R01_P00_20241007075724_01877.nc";

# Specify that numpy, scipy, et cetera should not use more than one thread or
#  process):
MKL_NUM_THREADS=1;
NUMEXPR_NUM_THREADS=1;
OMP_NUM_THREADS=1;
VECLIB_MAXIMUM_THREADS=1;
OPENBLAS_NUM_THREADS=1;
export MKL_NUM_THREADS NUMEXPR_NUM_THREADS OMP_NUM_THREADS;
export VECLIB_MAXIMUM_THREADS OPENBLAS_NUM_THREADS;

# Some environment vars that convey configuration info to the algorithm:

this_top_dir="$(absfpath "${base_dir}/..")";

PACKAGE_TOP_DIR="${this_top_dir}";
ANCILLARY_DATA_DIR="${this_top_dir}/dist/ancillary";

NN_MODEL_MONIKER="VIIRS";
NN_MODEL_SUBV="03";
export NN_MODEL_MONIKER NN_MODEL_SUBV;

OUTPUT_DIR=${base_dir}/outputs;

# Make required environment vars available:
export PACKAGE_TOP_DIR ANCILLARY_DATA_DIR OUTPUT_DIR;

# Check if output file directory exists; if not, bail:
tmpdir="${OUTPUT_DIR}";
test -d "${tmpdir}" || { echo "Output directory does not exist: ${tmpdir}"; exit 1; }

# If custom conda environment files exist, activate that conda environment:
conda_env_dir="${this_top_dir}/dist/c_env_for_PREFIRE_MSK";
if [ -d "${conda_env_dir}" ]; then
   activate_conda_env "${conda_env_dir}";
fi

PYTHONPATH=${PYTHONPATH}:${this_top_dir}/source;
export PYTHONPATH

# If custom conda environment files exist, DEactivate that conda environment:
if [ -d "${conda_env_dir}" ]; then
   deactivate_conda_env "${conda_env_dir}";
fi

for cfg_str in ${cfg_str1}
do
   ATRACK_IDX_RANGE_0BI=${cfg_str%,:*};

   tmp_str=${cfg_str##*,};
   L1B_RAD_FILE=${tmp_str%||*};
   AUX_MET_FILE=${tmp_str##*||};

   if [[ ${L1B_RAD_FILE} =~ '_SAT1_1B' ]]; then
      S_STR="SAT1";
   else
      S_STR="SAT2";
   fi
   NN_model_name=${NN_MODEL_MONIKER}-${S_STR}-${NN_MODEL_SUBV};

   export ATRACK_IDX_RANGE_0BI L1B_RAD_FILE AUX_MET_FILE;

   # Execute primary driver:
   python "${this_top_dir}"/source/PREFIRE_ML_MSK/ML_MSK.py ${L1B_RAD_FILE} ${AUX_MET_FILE} ${NN_model_name} ${OUTPUT_DIR};
done

# If custom conda environment files exist, DEactivate that conda environment:
if [ -d "${conda_env_dir}" ]; then
   deactivate_conda_env "${conda_env_dir}";
fi

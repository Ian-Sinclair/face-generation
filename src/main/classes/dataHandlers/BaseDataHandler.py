"""

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %%%%  Base Data Handler  %%%%%%
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    Summary
        - Data handlers process and manage data
          used for training models.
        - There exists at least one data handler for each
          dataset needed through the model training lifecycle.
        
        - BASE MODEL
            -   Contains util or basic functions that are shared
                between all data handlers -> i.e. is inherited by
                all other data handlers
"""

from src.main.classes.managerUtils import DirectoryUtil
import numpy as np
import h5py
import os

class BaseDataHandler():
    @staticmethod
    def store_hdf5(data, meta, filePath, fileName) -> None:
        """ Stores an array of images to HDF5.
            Parameters:
            ---------------
            data        data tuple to be stored
            meta        meta data to be stored
        """
        if DirectoryUtil.isProtectedFile(filePath,fileName):
            print(f"WARNING: Overwritting protected file at [{os.path.join(filePath, fileName)}]")
            
        # Create a new HDF5 file
        file = h5py.File(f"{os.path.join(filePath, fileName)}", "w")

        # Create a dataset in the file
        dataset = file.create_dataset(
            "data", np.shape(data), h5py.h5t.STD_U8BE, data=data
        )
        meta_set = file.create_dataset(
            "meta", np.shape(meta), h5py.h5t.STD_U8BE, data=meta
        )
        file.close()

    @staticmethod
    def read_hdf5(filePath, fileName) -> tuple[list,list]:
        """ Reads image from HDF5.
            Parameters:
            ---------------
            num_images   number of images to read

            Returns:
            ----------
            data        Data tuples retrieved
            meta        associated meta data, int label
        """
        data, meta = [], []

        # Open the HDF5 file
        file = h5py.File(f"{os.path.join(filePath, fileName)}", "r+")

        data = np.array(file["/data"]).astype("uint8")
        meta = np.array(file["/meta"]).astype("uint8")

        return data, meta


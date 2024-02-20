from src.main.classes.dataHandlers import BaseDataHandler as Base
from src.main.classes.managerUtils import DirectoryUtil

import numpy as np
import os
from PIL import Image
from typing_extensions import Self

class AnimeFacesDataHandler(Base):
    DATA_PATH = "data/animeFaces/"

    TRAINING_DATA_NAME = "trainingData.h5"
    TEST_DATA_NAME = "testData.h5"

    IMAGE_PATH = "images/"

    def __init__(self, image_size : tuple, collection_size : int):
        self.x_train = None
        self.y_train = None

        self.x_test = None
        self.y_test = None

        self.attributesDict = None
        self.images_list_full = None
        self.image_size = image_size
        self.collectionSize = collection_size
    
    def collect_data(self, collectionSize = None) -> Self:
        def load_image(file_path):
            return np.array(Image.open(file_path).resize(self.image_size))
        
        def process_path(file_name):
            img = load_image(self.images_dir + file_name)
            return img
        
        collectionSize = collectionSize or self.collectionSize
        self.images_dir = AnimeFacesDataHandler.DATA_PATH + AnimeFacesDataHandler.IMAGE_PATH

        if type(self.images_list_full) == type(None):
            self.images_list_full = np.array(list(os.listdir(self.images_dir)))
            print(f"{len(self.images_list_full)} image files detected in {self.images_dir}")

        images_list = np.random.choice(self.images_list_full, size=collectionSize, replace=False)

        self.x_train, self.y_train = [],[]
        self.x_test, self.y_test = [],[]

        for i,file_name in enumerate(images_list):
            print(f"Collecting Image: {i+1}/{len(images_list)} \t\t File Name: {file_name}", end='\r' if i+1 != len(images_list) else None)  # Display the current progress
            img = process_path(file_name)
            self.x_train.append(img)

        self.x_train = np.array(self.x_train)

        return self
    
    def process_data(self) -> Self:
        self.x_train = self.x_train.astype('float32') / 255.0

        return self
    
    def clear_data(self) -> Self:
        self.x_train = None
        self.y_train = None
        self.x_test = None
        self.y_test = None

        return self
    
    def get_processed_data(self) -> Self:
        return (self.x_train, self.y_train), (self.x_test, self.y_test)
    
    def save_data(self, path) -> Self:
        """
        Save the processed data to the specified path.

        Args:
            path (str): The path to save the data.

        Returns:
            Self: The MnistDataHandler instance.
        """
        if not DirectoryUtil.isValidDirectory(path):
            print(f"Directory Not Found at [{path}]")
            DirectoryUtil.promptToCreateDirectory(path) # Throws value error
        
        Base.store_hdf5(
            data=self.x_train,
            meta=self.y_train,
            filePath=path,
            fileName=AnimeFacesDataHandler.TRAINING_DATA_NAME
            )
        Base.store_hdf5(
            data=self.x_test,
            meta=self.y_test,
            filePath=path,
            fileName=AnimeFacesDataHandler.TEST_DATA_NAME
            )
        
        return self
    
    def read_local_data(self, file_path) -> Self:
        """
        Read the processed data from the specified file path.
        Writes to 
            [x_train, y_train, x_test, y_test]

        Args:
            file_path (str): The path to read the data from.

        Returns:
            Self: The MnistDataHandler instance.
        """
        self.clear_data()

        if DirectoryUtil.isValidFile(file_path, AnimeFacesDataHandler.TRAINING_DATA_NAME):
            self.x_train, self.y_train = Base.read_hdf5(filePath=file_path,
                                                        fileName=AnimeFacesDataHandler.TRAINING_DATA_NAME
                                                        )
        else :
            print(f"File [{AnimeFacesDataHandler.TRAINING_DATA_NAME}] Not Found at [{file_path}].")
        
        if DirectoryUtil.isValidFile(file_path, AnimeFacesDataHandler.TEST_DATA_NAME):
            self.x_test, self.y_test = Base.read_hdf5(filePath=file_path,
                                                      fileName=AnimeFacesDataHandler.TEST_DATA_NAME
                                                      )
        else:
            print(f"File [{AnimeFacesDataHandler.TEST_DATA_NAME}] Not Found at [{file_path}].")
        
        return self

if __name__ == "__main__": 
    raise SystemError(f"Class File [{AnimeFacesDataHandler.__name__}] Cannot Be Called As Main")


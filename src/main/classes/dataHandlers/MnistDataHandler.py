"""

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %%%%  MNIST Data Handler  %%%%%
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    Summary
        - Data handler class for the MNIST dataset
        
        Manages collecting dataset, pre-processing data, 
        and saving / loading datasets as h5 files
"""


from src.main.classes.dataHandlers import BaseDataHandler as Base
from src.main.classes.managerUtils import DirectoryUtil

from keras.datasets import mnist
from keras.utils import to_categorical
import numpy as np

from typing import Tuple
from typing_extensions import Self


class MnistDataHandler(Base):
    """
    Data handler class for processing and managing MNIST dataset.
    - Manages internal parameters
            [x_train, y_train, x_test, y_test]
    """

    TRAINING_DATA_NAME = "trainingData.h5"
    TEST_DATA_NAME = "testData.h5"

    NUM_LABELS = 10

    def __init__(self):
        self.x_train : np.ndarray = None
        self.y_train : np.ndarray = None

        self.x_test : np.ndarray = None
        self.y_test : np.ndarray = None

    def collect_data(self) -> Self:
        """
        Load the MNIST dataset.
        
        Returns:
            Self: The MnistDataHandler instance.
        """
        (self.x_train, self.y_train), (self.x_test, self.y_test) = mnist.load_data()

        return self
    
    def process_data(self) -> Self:
        """
        Preprocess the loaded data and rewrites
            [x_train, y_train, x_test, y_test]
        with the proprocessed data

        Returns:
            Self: The MnistDataHandler instance.
        """
        self.x_train = (self.x_train.astype('float32') / 255.0).reshape(self.x_train.shape + (1,))
        self.x_test = (self.x_test.astype('float32') / 255.0).reshape(self.x_test.shape + (1,))

        self.y_train = to_categorical(self.y_train)
        self.y_test = to_categorical(self.y_test)

        return self
    
    def clear_data(self) -> Self:
        """
        Clears the internal parameters.
            [x_train, y_train, x_test, y_test]

        Returns:
            Self: The MnistDataHandler instance.
        """
        self.x_train = None
        self.y_train = None

        self.x_test = None
        self.y_test = None

        return self
    
    def get_processed_data(self) -> Tuple[tuple, tuple]:
        """
        Get the processed data.
            (x_train, y_train), (x_test, y_test)

        Returns:
            Tuple[tuple, tuple]: A tuple containing training and testing data.
        """
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
            fileName=MnistDataHandler.TRAINING_DATA_NAME
            )
        Base.store_hdf5(
            data=self.x_test,
            meta=self.y_test,
            filePath=path,
            fileName=MnistDataHandler.TEST_DATA_NAME
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

        if DirectoryUtil.isValidFile(file_path + MnistDataHandler.TRAINING_DATA_NAME):
            self.x_train, self.y_train = Base.read_hdf5(filePath=file_path,
                                                        fileName=MnistDataHandler.TRAINING_DATA_NAME
                                                        )
        else :
            print(f"File [{MnistDataHandler.TRAINING_DATA_NAME}] Not Found at [{file_path}].")
        
        if DirectoryUtil.isValidFile(file_path + MnistDataHandler.TEST_DATA_NAME):
            self.x_test, self.y_test = Base.read_hdf5(filePath=file_path,
                                                      fileName=MnistDataHandler.TEST_DATA_NAME
                                                      )
        else:
            print(f"File [{MnistDataHandler.TEST_DATA_NAME}] Not Found at [{file_path}].")
        
        return self

if __name__ == "__main__": 
    raise SystemError(f"Class File [{MnistDataHandler.__name__}] Cannot Be Called As Main")


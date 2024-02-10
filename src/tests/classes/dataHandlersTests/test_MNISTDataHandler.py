import unittest
import random
import numpy as np
from unittest.mock import MagicMock, patch
from src.main.classes.dataHandlers import MnistDataHandler

class TestMnistDataHandler(unittest.TestCase):
    def setUp(self):
        self.data_handler = MnistDataHandler()

    @patch('keras.datasets.mnist.load_data')
    def test_collect_data(self, mock_load_data):
        # Mock the mnist.load_data() function
        mock_load_data.return_value = (
            (mock_x_train, mock_y_train),
            (mock_x_test, mock_y_test)
        ) = (MagicMock(), MagicMock()), (MagicMock(), MagicMock())

        # Call collect_data method
        self.data_handler.collect_data()

        # Check if data is assigned correctly
        self.assertEqual(self.data_handler.x_train, mock_x_train)
        self.assertEqual(self.data_handler.y_train, mock_y_train)
        self.assertEqual(self.data_handler.x_test, mock_x_test)
        self.assertEqual(self.data_handler.y_test, mock_y_test)

    def test_process_data(self):
        def generate_random_list_y_data():
            return np.random.randint(1, 11, size=(10,))
        def generate_random_list_x_data():
            # Define image dimensions
            image_width = 28
            image_height = 28
            num_images = 10
            return np.random.randint(0, 256, size=(num_images, image_height, image_width))

        # Create a MagicMock object
        mock_object_y = MagicMock()
        mock_object_x = MagicMock()

        # Set up the side effect to generate a list of random integers
        mock_object_y.side_effect = generate_random_list_y_data
        mock_object_x.side_effect = generate_random_list_x_data

        # Assign some mock data for processing
        self.data_handler.x_train = mock_object_x()
        self.data_handler.y_train = mock_object_y()
        self.data_handler.x_test = mock_object_x()
        self.data_handler.y_test = mock_object_y()

        # Call process_data method
        self.data_handler.process_data()

        # Check if data is processed correctly
        self.assertTrue(self.data_handler.x_train.any())
        self.assertTrue(self.data_handler.y_train.any())
        self.assertTrue(self.data_handler.x_test.any())
        self.assertTrue(self.data_handler.y_test.any())

    def test_clear_data(self):
        # Assign some mock data
        self.data_handler.x_train = MagicMock()
        self.data_handler.y_train = MagicMock()
        self.data_handler.x_test = MagicMock()
        self.data_handler.y_test = MagicMock()

        # Call clear_data method
        self.data_handler.clear_data()

        # Check if data is cleared
        self.assertIsNone(self.data_handler.x_train)
        self.assertIsNone(self.data_handler.y_train)
        self.assertIsNone(self.data_handler.x_test)
        self.assertIsNone(self.data_handler.y_test)
    
    @patch('src.main.classes.managerUtils.DirectoryUtil.isValidDirectory', return_value=True)
    @patch('src.main.classes.dataHandlers.BaseDataHandler.store_hdf5')
    def test_save_data(self, mock_store_hdf5, mock_is_valid_directory):
        # Assign some mock data
        self.data_handler.x_train = MagicMock()
        self.data_handler.y_train = MagicMock()
        self.data_handler.x_test = MagicMock()
        self.data_handler.y_test = MagicMock()

        # Call save_data method
        self.data_handler.save_data('/path/to/save')

        # Check if the directory is validated
        mock_is_valid_directory.assert_called_once_with('/path/to/save')

        # Check if store_hdf5 is called with the correct arguments
        mock_store_hdf5.assert_any_call(data=self.data_handler.x_train,
                                         meta=self.data_handler.y_train,
                                         filePath='/path/to/save',
                                         fileName=MnistDataHandler.TRAINING_DATA_NAME)
        mock_store_hdf5.assert_any_call(data=self.data_handler.x_test,
                                         meta=self.data_handler.y_test,
                                         filePath='/path/to/save',
                                         fileName=MnistDataHandler.TEST_DATA_NAME)

    @patch('src.main.classes.managerUtils.DirectoryUtil.isValidFile', return_value=True)
    @patch('src.main.classes.dataHandlers.BaseDataHandler.read_hdf5', return_value=(MagicMock(), MagicMock()))
    def test_read_local_data(self, mock_read_hdf5, mock_is_valid_file):
        # Call read_local_data method
        self.data_handler.read_local_data('/path/to/read')

        # Check if the data is read correctly
        mock_is_valid_file.assert_any_call('/path/to/read' + MnistDataHandler.TRAINING_DATA_NAME)
        mock_is_valid_file.assert_any_call('/path/to/read' + MnistDataHandler.TEST_DATA_NAME)

        # Check if read_hdf5 is called with the correct arguments
        mock_read_hdf5.assert_any_call(filePath='/path/to/read',
                                         fileName=MnistDataHandler.TRAINING_DATA_NAME)
        mock_read_hdf5.assert_any_call(filePath='/path/to/read',
                                         fileName=MnistDataHandler.TEST_DATA_NAME)


if __name__ == '__main__':
    unittest.main()

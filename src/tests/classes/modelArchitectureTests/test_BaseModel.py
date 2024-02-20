import unittest
from unittest.mock import MagicMock, patch
import os
from src.main.classes.managerUtils import DirectoryUtil
from src.main.classes.modelArchitectures import BaseModel

from keras import Model, models

class TestBaseModel(unittest.TestCase):

    def setUp(self):
        # Create a mock Keras model
        self.mock_model = MagicMock()

        # Initialize BaseModel with the mock model
        self.base_model = BaseModel(model=self.mock_model)

        # Mock path for testing
        self.test_path = "/path/to/test"

    def tearDown(self):
        pass

    def test_save_model(self):
        # Mocking the required functions of DirectoryUtil
        with patch.object(DirectoryUtil, 'isValidDirectory', return_value=True):
            with patch.object(DirectoryUtil, 'isValidFile', return_value=False):
                with patch.object(DirectoryUtil, 'isProtectedFile', return_value=False):
                    with patch("builtins.open", unittest.mock.mock_open(read_data='')):
                        # Call save_model method
                        result = self.base_model.save_model(self.test_path)

                        # Assert that the model is saved successfully
                        self.assertEqual(result, self.base_model)

    @patch('keras.models.model_from_json')
    def test_load_architecture(self, model_from_json):
        model_from_json.return_value = MagicMock()
        # Mocking the required functions of DirectoryUtil
        with patch.object(DirectoryUtil, 'isValidFile', return_value=True):
            with patch("builtins.open", unittest.mock.mock_open(read_data='{"config": "model_config"}')):
                # Call load_architecture method
                try :
                    result = self.base_model.load_architecture(self.test_path)
                except ValueError as e :
                    self.fail(e)

                # Assert that the model architecture is loaded successfully
                self.assertEqual(result, self.base_model)

    def test_load_model(self):
        # Mocking the required functions of DirectoryUtil
        with patch.object(BaseModel, 'load_architecture', return_value=self.base_model):
            with patch.object(BaseModel, 'load_weights', return_value=self.base_model):
                # Call load_model method
                result = self.base_model.load_model(self.test_path)
                # Assert that the model is loaded successfully
                self.assertEqual(result, self.base_model)
                BaseModel.load_architecture.assert_called_once_with(self.test_path)
                BaseModel.load_weights.assert_called_once_with(self.test_path)

    def test_load_weights(self):
        # Mocking the required functions of DirectoryUtil
        with patch.object(DirectoryUtil, 'isValidFile', return_value=True):
            # Call load_weights method
            result = self.base_model.load_weights(self.test_path)

            # Assert that the weights are loaded successfully
            self.assertEqual(result, self.base_model)

    
    def plot_model_patch(*args, **kwargs):
        return MagicMock()
    @patch('keras.utils.plot_model', new=plot_model_patch)
    def test_save_architecture_diagram(self):
        # Mocking the required functions of DirectoryUtil
        with patch.object(DirectoryUtil, 'isValidDirectory', return_value=True):
            with patch.object(DirectoryUtil, 'isProtectedFile', return_value=False):
                # Call save_architecture_diagram method
                result = self.base_model.save_architecture_diagram(self.test_path, "model_architecture.png")

                # Assert that the model architecture diagram is saved successfully
                self.assertEqual(result, self.base_model)

if __name__ == '__main__':
    unittest.main()

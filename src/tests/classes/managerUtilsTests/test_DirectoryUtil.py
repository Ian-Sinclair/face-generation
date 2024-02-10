import unittest
from unittest.mock import patch
from src.main.classes.managerUtils import DirectoryUtil

class TestDirectoryUtil(unittest.TestCase):
    def test_isValidDirectory_existing_directory(self):
        # Test when directory exists
        with patch('os.path.exists', return_value=True), patch('os.path.isdir', return_value=True):
            self.assertTrue(DirectoryUtil.isValidDirectory('/path/to/existing/directory'))
    
    def test_isValidDirectory_non_existing_directory(self):
        # Test when directory doesn't exist
        with patch('os.path.exists', return_value=False):
            self.assertFalse(DirectoryUtil.isValidDirectory('/path/to/non/existing/directory'))
    
    def test_isValidFile_existing_file(self):
        # Test when file exists
        with patch('os.path.exists', return_value=True), patch('os.path.isdir', return_value=True):
            self.assertTrue(DirectoryUtil.isValidFile('/path/to/existing/','file'))
    
    def test_isValidFile_non_existing_file(self):
        # Test when file doesn't exist
        with patch('os.path.exists', return_value=False), patch('os.path.isdir', return_value=True):
            self.assertFalse(DirectoryUtil.isValidFile('/path/to/non/existing/','file'))
    
    @patch('builtins.input', return_value='y')
    def test_promptToCreateDirectory_yes_input(self, mock_input):
        # Test user input 'y'
        with patch('os.makedirs') as mock_makedirs:
            self.assertTrue(DirectoryUtil.promptToCreateDirectory('/path/to/new/directory'))
            mock_makedirs.assert_called_once_with('/path/to/new/directory')
    
    @patch('builtins.input', return_value='n')
    def test_promptToCreateDirectory_no_input(self, mock_input):
        # Test user input 'n'
        with self.assertRaises(ValueError) as context:
            DirectoryUtil.promptToCreateDirectory('/path/to/new/directory')
        self.assertEqual(str(context.exception), "Directory creation declined for '/path/to/new/directory'.")

if __name__ == '__main__':
    unittest.main()

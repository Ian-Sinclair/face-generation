"""

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %%%%%  Directory Utils  %%%%%%%
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    Summary
        - Utility functions to manage internal project directories
        
        Including validating directory paths and files, along with
        creating new directories.
"""

import os

class DirectoryUtil():
    protected_files : list = []

    @staticmethod
    def isValidDirectory(path : str) -> bool:
        """
        Check if the given path is a valid directory.

        Args:
            path (str): The path to check.

        Returns:
            bool: True if the path is a valid directory, False otherwise.
        """
        if not os.path.exists(path):
            print(f"Directory Not Found at [{path}]")
            return False
        elif not os.path.isdir(path):
            return False
        
        return True

    @staticmethod
    def isValidFile(path : str, file_name : str) -> bool:
        """
        Check if the given path is a valid file.

        Args:
            path (str): The path to check.

        Returns:
            bool: True if the path is a valid file, False otherwise.
        """
        if not DirectoryUtil.isValidDirectory(path):
            return False
        if not os.path.exists(os.path.join(path, file_name) ):
            print(f"File Not Found at [{os.path.join(path, file_name)}]")
            return False
        
        return True
    
    @classmethod
    def isProtectedFile(cls, path : str, file_name : str) -> bool:
        """
        Check if a file is listed as protected.

        Args:
            path (str): The directory path.
            file_name (str): The file name.

        Returns:
            bool: True if the file is protected, False otherwise.
        """
        return os.path.join(path, file_name) in cls.protected_files
    
    @classmethod
    def setProtectedFilesList(cls, protected_files : list) -> bool :
        """
        Set the list of protected files.

        Args:
            protected_files (list): List of protected file paths.

        Returns:
            bool: True if the list is set successfully.
        """
        cls.protected_files = protected_files
        return True


    @staticmethod
    def promptToCreateDirectory(path : str) -> bool:
        """
        Prompt the user to create a directory if it doesn't exist.

        Args:
            path (str): The path where the directory should be created.

        Returns:
            bool: True if the directory was created or already exists, False otherwise.
        
        Raises:
            ValueError: If the user declines to create the directory.
        """
        if DirectoryUtil.isValidDirectory(path): return True
        user_input : str = input(
            f"""
            REQUIRED: Would you like to create the 
            directory at [{path}] Y/N\nconfirm? """
            )
        create_dir : bool = user_input.lower() == "y" or user_input.lower() == "yes"

        if create_dir : 
            os.makedirs(path)
            print(f"Directory '{path}' created successfully.")
            return True
        else :
            raise ValueError(f"Directory creation declined for '{path}'.")
        

        

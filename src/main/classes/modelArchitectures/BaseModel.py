"""

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %%%%%  Base Model Architecture  %%%%%
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    Summary
        -   Base model - wraps keras.Models to provide universal
            functions for all model architecture classes.
        
        Manages saving, loading and creating architecture visualizations
        for each model
"""

from src.main.classes.managerUtils import DirectoryUtil

from keras import Model, models, utils
import os

from typing_extensions import Self

class BaseModel():
    """
    Base class for handling model operations like saving, loading, and diagram creation.
    """
    MODEL_ARCHITECTURE_FILENAME = "model_architecture.json"
    MODEL_WEIGHTS_FILENAME = "model_weights.h5"

    def __init__(self, model : Model = None) :
        """
        Initialize the BaseModel instance.

        Args:
            model (Model, optional): The Keras model. Defaults to None.
        """
        self.model : Model = model
    
    def __call__(self, *args) :
        return self.model(args)
    
    def __str__(self) -> str:
        """
        String representation of the model.

        Returns:
            str: String representation of the model.
        """
        if self.model is None :
            return "Model [None]"
        stringSummary = []
        self.model.summary(print_fn=lambda x: stringSummary.append(x))
        return "\n".join(stringSummary)

    def save_model(self, path : str) -> Self:
        """
        Save the model to the specified path.

        Args:
            path (str): The directory path where the model will be saved.
            Model will be saved as an architecture JSON and weights h5 file

        Returns:
            BaseModel: The BaseModel instance.
        """
        if self.model is None :
            print(f"model is [{self.model}] cannot save model")
            return self
        if not DirectoryUtil.isValidDirectory(path):
            print(f"Directory Not Found at [{path}]")
            try : 
                DirectoryUtil.promptToCreateDirectory(path) # Throws value error
            except ValueError as e:
                print(e)
                return Self
        
        if DirectoryUtil.isProtectedFile(path,BaseModel.MODEL_ARCHITECTURE_FILENAME) :
            print(f"ABORTING: File [{os.path.join(path, BaseModel.MODEL_ARCHITECTURE_FILENAME)}] is protected - cannot overwrite file")
            return self
        
        if DirectoryUtil.isProtectedFile(path,BaseModel.MODEL_WEIGHTS_FILENAME) :
            print(f"ABORTING: File [{os.path.join(path, BaseModel.MODEL_WEIGHTS_FILENAME)}] is protected - cannot overwrite file")
            return self
        
        # Save model architecture to JSON and weights to HDF5
        model_json = self.model.to_json()
        with open(os.path.join(path, BaseModel.MODEL_ARCHITECTURE_FILENAME), 'w') as json_file:
            json_file.write(model_json)
        
        self.model.save_weights(os.path.join(path, BaseModel.MODEL_WEIGHTS_FILENAME))
        return self
    
    def load_model(self, path : str) -> Self :
        """
        Load the model from the specified path.

        Args:
            path (str): The directory path where the model is saved.

        Returns:
            BaseModel: The BaseModel instance.
        """
        try :
            self.load_architecture(path)
        except ValueError as e :
            print(e)
            return self
        try :
            self.load_weights(path)
        except ValueError as e :
            print(e)
        return self
    
    def load_architecture(self, path : str) -> Self :
        """
        Load the model architecture from the specified path.

        Args:
            path (str): The directory path where the model architecture is saved.

        Returns:
            BaseModel: The BaseModel instance.
        """
        if not DirectoryUtil.isValidFile(os.path.join(path + BaseModel.MODEL_ARCHITECTURE_FILENAME)):
            print(f"REQUIRED ELEMENT FAILURE: Directory Not Found at [{os.path.join(path + BaseModel.MODEL_ARCHITECTURE_FILENAME)}]")
            print(f"Cannot load model architecture")
            raise ValueError(f"REQUIRED ELEMENT FAILURE: Directory Not Found at [{os.path.join(path + BaseModel.MODEL_ARCHITECTURE_FILENAME)}]\n Cannot Load Model Architecture")
        
        # Load model architecture from JSON
        with open(os.path.join(path + BaseModel.MODEL_ARCHITECTURE_FILENAME), 'r') as json_file:
            loaded_model_json = json_file.read()
        
        self.model = models.model_from_json(loaded_model_json)
        return self
    
    def load_weights(self, path : str) -> Self :
        """
        Load the model weights from the specified path.

        Args:
            path (str): The directory path where the model weights are saved.

        Returns:
            BaseModel: The BaseModel instance.
        """
        if self.model is None : 
            print(f"REQUIRED ELEMENT FAILURE: Model is [{self.model}] cannot load weights")
            return self
        
        if not DirectoryUtil.isValidFile(path=path, file_name=BaseModel.MODEL_WEIGHTS_FILENAME):
            print(f"REQUIRED ELEMENT FAILURE: Directory Not Found at [{os.path.join(path + BaseModel.MODEL_WEIGHTS_FILENAME)}]")
            print(f"Cannot load model weights")
            raise ValueError(f"REQUIRED ELEMENT FAILURE: Directory Not Found at [{os.path.join(path + BaseModel.MODEL_WEIGHTS_FILENAME)}]\n Cannot Load Model Weights")
        
        self.model.load_weights(path + BaseModel.MODEL_WEIGHTS_FILENAME)
        return self
    
    def save_architecture_diagram(self, path : str, file_name : str) -> Self :
        """
        Save the model architecture diagram to the specified path.

        Args:
            path (str): The directory path where the diagram will be saved.
            file_name (str): The name of the file to save.

        Returns:
            BaseModel: The BaseModel instance.
        """
        if self.model is None : 
            print(f"model is [{self.model}] cannot create model architecture diagram")
            return self
        
        if not DirectoryUtil.isValidDirectory(path):
            print(f"Directory Not Found at [{path}]")
            try : 
                DirectoryUtil.promptToCreateDirectory(path) # Throws value error
            except ValueError as e:
                print(e)
                return Self
        
        if DirectoryUtil.isProtectedFile(path,file_name) :
            print(f"ABORTING: File [{os.path.join(path, file_name)}] is protected - cannot overwrite file")
            return self
        
        utils.plot_model(
            self.model, 
            to_file=os.path.join(path, file_name), 
            show_shapes = True, 
            show_layer_names = True,
            expand_nested=True,
            show_layer_activations=True,
            show_trainable=True
            )
        return self
        

if __name__ == "__main__": 
    raise SystemError(f"Class File [{BaseModel.__name__}] Cannot Be Called As Main")
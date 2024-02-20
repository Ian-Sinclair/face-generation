from src.main.classes.modelArchitectures import BaseModel

from keras import Model, Input, layers
from keras.initializers import RandomNormal

from typing_extensions import Self
from typing import Literal, Callable

class Critic(BaseModel):
    """Evaluates the realism of generated images."""
    def __init__(
        self,
            input_size : tuple[int, int, int],
            filters : list[int],
            kernels : list[int],
            strides : list[int],
            activation_functions : list[Callable | str | Literal["linear", "relu", "sigmoid", "tanh", "softmax"]] = ['relu'],
            use_drop_out : bool = True,
            drop_out_rate: float = 0.25
    ):
        """
        Initialize the Critic.

        Args:
            input_size (tuple[int, int, int]): Input image dimensions.
            filters (list[int]): Number of filters for each convolutional layer.
            kernels (list[int]): Kernel size for each convolutional layer.
            strides (list[int]): Stride size for each convolutional layer.
            activation_functions (list[Callable | str]): Activation functions for each layer.
            use_drop_out (bool): Whether to use dropout.
            drop_out_rate (float): Dropout rate.
        """
        self.input_size = input_size
        self.filters = filters
        self.kernels = kernels
        self.strides = strides
        self.activation_functions = activation_functions
        self.use_drop_out = use_drop_out
        self.drop_out_rate = drop_out_rate

        self.model_depth = min(len(lst) for lst in [self.filters, self.kernels, self.strides])

        self.input_layer = None
        self.output_layer = None

        model : Model = self._build_critic()

        super().__init__(model)

    def _build_critic(self) -> Model :
        """
        Build the critic model.

        Returns:
            Model: The critic model.
        """
        
        # -------------------------
        # ----Input Validation ----
        # -------------------------
        if not (len(self.filters) == len(self.kernels) == len(self.strides)) :
            print(f"WARNING: Layer Diminsionality Is Un-Even - number of each layer = filters - [{len(self.filters)}, kernels - [{len(self.kernels)}], strides - [{len(self.strides)}]]")
            print(f"WARNING: Reconfiguring Layers")
            self.filters = self.filters[self.model_depth:]
            self.kernels = self.kernels[self.model_depth:]
            self.strides = self.strides[self.model_depth:]
            print(f"filter_sizes = [{self.filters}, kernel_sizes = [{self.kernels}], strides = [{self.strides}]]")
        
        if len(self.activation_functions) != self.model_depth :
            print(f"NOTICE: Extending Activation Function Array")
            self.activation_functions = self.activation_functions + [self.activation_functions[-1]]*(self.model_depth-len(self.activation_functions))

            print(f"Activation Function Length: [{len(self.activation_functions)}]")
        
        # --------------------------
        # --- Model Construction ---
        # --------------------------

        self.input_layer = Input(shape=self.input_size, name="Critic_Input")
        x = self.input_layer

        for i in range(self.model_depth) :
            conv_layer = layers.Conv2D(
                filters=self.filters[i],
                kernel_size=self.kernels[i],
                strides=self.strides[i],
                padding="same",
                #kernel_initializer=RandomNormal(mean=0., stddev=0.02),
                name = f'{self.__class__.__name__}_Conv2D_{i}'
            )
            x = conv_layer(x)
            x = layers.Activation(self.activation_functions[i])(x)

            if self.use_drop_out : 
                x = layers.Dropout(rate = self.drop_out_rate)(x)
        
        x = layers.Flatten()(x)

        self.output_layer = layers.Dense(
            units=1, 
            activation=None, 
            #kernel_initializer=RandomNormal(mean=0., stddev=0.02),
            name = f'{self.__class__.__name__}_output_layer'
        )(x)

        return Model(self.input_layer, outputs=self.output_layer, name="Critic")
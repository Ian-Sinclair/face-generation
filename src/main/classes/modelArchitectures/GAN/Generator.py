from src.main.classes.modelArchitectures import BaseModel

from keras import Model, Input, layers
from keras import backend as K
from keras.initializers import RandomNormal
import numpy as np

from typing_extensions import Self
from typing import Literal, Callable

class Generator(BaseModel) :
    """Generates images from random noise."""
    MODEL_NAME = "Generator"
    def __init__(
            self,
            latent_dim : int,
            output_size : tuple[int, int, int],
            filters : list[int],
            kernels : list[int],
            upsample_sizes : list[int],
            activation_functions : list[Callable | str | Literal["linear", "relu", "sigmoid", "tanh", "softmax"]] = ["relu"],
            use_batch_norm : bool = True,
            batch_norm_momentum : float = 0.9,
            use_drop_out : bool = True,
            drop_out_rate : float = 0.25
    ):
        """
        Initialize the Generator.

        Args:
            latent_dim (int): Dimensionality of the latent space.
            output_size (tuple[int, int, int]): Output image dimensions.
            filters (list[int]): Number of filters for each convolutional layer.
            kernels (list[int]): Kernel size for each convolutional layer.
            upsample_sizes (list[int]): Upsample size for each layer.
            activation_functions (list[Callable | str]): Activation functions for each layer.
            use_batch_norm (bool): Whether to use batch normalization.
            batch_norm_momentum (float): Momentum for batch normalization.
            use_drop_out (bool): Whether to use dropout.
            drop_out_rate (float): Dropout rate.
        """
        self.latent_dim = latent_dim
        self.output_size = output_size
        self.filters = filters
        self.kernels = kernels
        self.upsample_sizes = upsample_sizes
        self.activation_functions = activation_functions
        self.use_batch_norm = use_batch_norm
        self.batch_norm_momentum = batch_norm_momentum
        self.use_drop_out = use_drop_out
        self.drop_out_rate = drop_out_rate

        # Determine model depth
        self.model_depth = min(len(lst) for lst in [self.filters, self.kernels])

        # Initialize layers
        self.input_layer = None
        self.output_layer = None

        # Build generator model
        model = self._build_generator()

        super().__init__(model)

    def _build_generator(self) :
        """
        Build the generator model.

        Returns:
            Model: The generator model.
        """
        # -------------------------
        # ----Input Validation ----
        # -------------------------

        if not (len(self.filters) == len(self.kernels) == len(self.upsample_sizes)) :
            print(
                f"""WARNING: Layer Diminsionality Is Un-Even 
                - number of each layer - 
                filters - [{len(self.filters)}, 
                kernels - [{len(self.kernels)}], 
                upsamples - [{len(self.upsample_sizes)}]]
                """)
            print(f"WARNING: Reconfiguring Layers")
            self.filters = self.filters[self.model_depth:]
            self.kernels = self.kernels[self.model_depth:]
            self.upsample_sizes = self.upsample_sizes[self.model_depth:]
            print(f"filter_sizes = [{self.filters}, kernel_sizes = [{self.kernels}], upsample_sizes = [{self.upsample_sizes}]]")

        if len(self.activation_functions) != self.model_depth+1 :
            # Extend activation function array
            self.activation_functions = self.activation_functions + [self.activation_functions[-1]]*(self.model_depth+1-len(self.activation_functions))
        
        # --------------------------
        # --- Model Construction ---
        # --------------------------

        h,w,c = self.output_size
        upsample = np.prod(self.upsample_sizes)
        self.init_conv_size = (int(h/upsample),int(w/upsample),64)

        self.input_layer = Input(
            shape=self.latent_dim, 
            name="Generator_Input"
            )
        
        x = layers.Dense(
            np.prod(self.init_conv_size),
            #kernel_initializer=RandomNormal(mean=0., stddev=0.02)
            )(self.input_layer)
        
        if self.use_batch_norm : 
            x = layers.BatchNormalization(momentum=self.batch_norm_momentum)(x)

        init_activation_function = self.activation_functions.pop(0)
        x = layers.Activation(init_activation_function)(x)

        x = layers.Reshape(self.init_conv_size)(x)

        if self.use_drop_out :
            x = layers.Dropout(rate=self.drop_out_rate)(x)
        
        for i in range(self.model_depth) :
            x = layers.UpSampling2D(size=self.upsample_sizes[i])(x)
            conv_layer = layers.Conv2D(filters=self.filters[i],
                                           kernel_size=self.kernels[i],
                                           strides=1,
                                           padding="same",
                                           #kernel_initializer=RandomNormal(mean=0., stddev=0.02),
                                           name = f'{self.__class__.__name__}_Conv2D_{i}'
                                           )
            x = conv_layer(x)

            if i < self.model_depth-1:
                if self.use_batch_norm : 
                    x = layers.BatchNormalization()(x)
                
                x = layers.Activation(self.activation_functions[i])(x)

                if self.use_drop_out : 
                    x = layers.Dropout(rate = self.drop_out_rate)(x)
            
            else :
                x = layers.Activation("sigmoid")(x)
        
        self.output_layer = x

        if K.int_shape(self.output_layer)[1:] != self.output_size :
            print(
                f"""
                {Generator.MODEL_NAME} WARNING: Original feature dimension of shape [{self.output_size}] 
                does not match dimension of decoded space [{K.int_shape(self.output_layer)[1:]}]
                """
                )
            
        return Model(self.input_layer, outputs=self.output_layer, name=Generator.MODEL_NAME)
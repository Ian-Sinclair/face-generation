from src.main.classes.modelArchitectures import BaseModel
from src.main.classes.modelArchitectures.GAN import Critic, Generator, GANLosses, GANVisualizations
from src.main.classes.managerUtils import DirectoryUtil

from keras import Model
from keras.optimizers import legacy as optimizers
from keras import backend as K
import tensorflow as tf
import numpy as np
import time
import pandas as pd
from dataclasses import dataclass
import os
import pickle

from typing_extensions import Self
from typing import Literal, Callable, Tuple

@dataclass
class loss_metrics:
    critic_real_loss : tf.Tensor | float = None
    critic_synthetic_loss : tf.Tensor | float = None
    critic_interpolated_loss : tf.Tensor | float = None
    generator_loss : tf.Tensor | float = None

    def reduce_tensors(self) -> Self:
        if isinstance(self.critic_real_loss, tf.Tensor):
            self.critic_real_loss = tf.math.reduce_mean(self.critic_real_loss).numpy()
        if isinstance(self.critic_synthetic_loss, tf.Tensor):
            self.critic_synthetic_loss = tf.math.reduce_mean(self.critic_synthetic_loss).numpy()
        if isinstance(self.critic_interpolated_loss, tf.Tensor):
            self.critic_interpolated_loss = tf.math.reduce_mean(self.critic_interpolated_loss).numpy()
        if isinstance(self.generator_loss, tf.Tensor):
            self.generator_loss = tf.math.reduce_mean(self.generator_loss).numpy()
        return self


    def toJSON(self) :
        return {
            "Critic Real Loss" : self.critic_real_loss,
            "Critic Synthetic Loss" : self.critic_synthetic_loss,
            "Critic Interpolated Loss" : self.critic_interpolated_loss,
            "Generator Synthetic Loss" : self.generator_loss
        }


@dataclass
class time_metrics:
    critic_step_time : float = None
    generator_step_time : float = None
    overhead_step_time : float = None
    total_runtime : float = None

    def toJSON(self) :
        return {
            "Critic Training Time Per Step" : self.critic_step_time,
            "Generator Training Time Per Step" : self.generator_step_time,
            "Overhead Time Per Step" : self.overhead_step_time,
            "Total Runtime" : self.total_runtime
        }

class WGAN_GP(BaseModel):
    """
    WGAN_GP class implementing the Wasserstein GAN with Gradient Penalty (WGAN-GP) architecture.

    This class inherits from BaseModel and provides functionalities to train the WGAN-GP model.

    Args:
        image_size (Tuple[int, int, int]): Size of the input images (height, width, channels).
        latent_space_dim (int): Dimension of the latent space.
        generator_filter_sizes (list[int]): List of filter sizes for the generator layers.
        generator_kernel_sizes (list[int]): List of kernel sizes for the generator layers.
        generator_upsample_sizes (list[int]): List of upsample sizes for the generator layers.
        critic_filter_sizes (list[int]): List of filter sizes for the critic layers.
        critic_kernel_sizes (list[int]): List of kernel sizes for the critic layers.
        critic_strides (list[int]): List of strides for the critic layers.
        generator_activation_functions (list[Callable | str | Literal["linear", "relu", "sigmoid", "tanh", "softmax"]]): List of activation functions for the generator layers.
        generator_use_batch_norm (bool): Whether to use batch normalization in the generator.
        generator_batch_norm_momentum (float): Momentum for the batch normalization in the generator.
        generator_use_drop_out (bool): Whether to use dropout layers in the generator.
        generator_drop_out_rate (float): Dropout rate for the dropout layers in the generator.
        critic_activation_functions (list[Callable | str | Literal["linear", "relu", "sigmoid", "tanh", "softmax"]]): List of activation functions for the critic layers.
        critic_use_drop_out (bool): Whether to use dropout layers in the critic.
        critic_drop_out_rate (float): Dropout rate for the dropout layers in the critic.
    """
    CRITIC_NAME = "critic"
    GENERATOR_NAME = "generator"
    MODEL_NAME = "complete"
    CLASS_FILENAME = "WGAN_GP.pkl"
    TRAINING_METRICS_FILENAME = "training_metric.csv"

    def __init__(
                self,
                image_size : Tuple[int, int, int],

                latent_space_dim : int,
                
                generator_filter_sizes : list[int],
                generator_kernel_sizes : list[int],
                generator_upsample_sizes : list[int],

                critic_filter_sizes : list[int],
                critic_kernel_sizes : list[int],
                critic_strides : list[int],

                generator_activation_functions : list[Callable | str | Literal["linear", "relu", "sigmoid", "tanh", "softmax"]] = ['relu'],
                generator_use_batch_norm : bool = True,
                generator_batch_norm_momentum : float = 0.9,
                generator_use_drop_out : bool = True,
                generator_drop_out_rate : float = 0.25,

                critic_activation_functions : list[Callable | str | Literal["linear", "relu", "sigmoid", "tanh", "softmax"]] = ['relu'],
                critic_use_drop_out : bool = True,
                critic_drop_out_rate: float = 0.25
        ) :
        # Initialize metric classes
        self.loss_metrics = loss_metrics()
        self.time_metrics = time_metrics()

        # Initialize the model parameters
        self._image_size = image_size

        self._latent_space_dim = latent_space_dim

        self._generator_filter_sizes     = generator_filter_sizes
        self._generator_kernel_sizes     = generator_kernel_sizes
        self._generator_upsample_sizes   = generator_upsample_sizes

        self._critic_filter_sizes        = critic_filter_sizes
        self._critic_kernel_sizes        = critic_kernel_sizes
        self._critic_strides             = critic_strides

        self._generator_activation_functions     = generator_activation_functions
        self._generator_use_batch_norm           = generator_use_batch_norm
        self._generator_batch_norm_momentum      = generator_batch_norm_momentum
        self._generator_use_drop_out             = generator_use_drop_out
        self._generator_drop_out_rate            = generator_drop_out_rate

        self._critic_activation_functions        = critic_activation_functions
        self._critic_use_drop_out                = critic_use_drop_out
        self._critic_drop_out_rate               = critic_drop_out_rate

        
        # Initialize the generator and critic models
        self.generator = Generator(
            latent_dim = self._latent_space_dim,
            output_size = self._image_size,
            filters = self._generator_filter_sizes,
            kernels = self._generator_kernel_sizes,
            upsample_sizes = self._generator_upsample_sizes,
            activation_functions = self._generator_activation_functions,
            use_batch_norm=self._generator_use_batch_norm,
            batch_norm_momentum=self._generator_batch_norm_momentum,
            use_drop_out=self._generator_use_drop_out,
            drop_out_rate=self._generator_drop_out_rate
        )

        self.critic = Critic(
            input_size = self._image_size,
            filters = self._critic_filter_sizes,
            kernels = self._critic_kernel_sizes,
            strides = self._critic_strides,
            activation_functions=self._critic_activation_functions,
            use_drop_out=self._critic_use_drop_out,
            drop_out_rate=self._critic_drop_out_rate
        )

        # Define input and output layers for the model
        self.input  = self.generator.input_layer

        self.synthetic_layer = self.generator.model(self.input)

        self.belief_layer = self.critic.model(self.synthetic_layer)

        model = Model(self.input, self.belief_layer, name="WGAN_Complete_Model")

        # Call parent class constructor
        super().__init__(model)

    def train(
            self,
            x_train : np.ndarray,
            batch_size : int,
            epochs : int,
            save_folder : str = None
        ) -> Self:
        """
        Trains the WGAN-GP model.

        Args:
            x_train (np.ndarray): Training dataset.
            batch_size (int): Batch size for training.
            epochs (int): Number of training epochs.
            save_folder (str, optional): Folder to save training metrics. Defaults to None.
        """
        num_batches = x_train.shape[0]//batch_size

        generator_optimizer = optimizers.RMSprop(learning_rate=0.00005)
        critic_optimizer = optimizers.RMSprop(learning_rate=0.00005)

        synthetic_images_plot_handler = GANVisualizations\
            .Generator_Images_Plot_Handler()\
                .initialize_random_latent_samples(shape=(batch_size, self.generator.latent_dim))
        
        loss_plot_handler = GANVisualizations.Line_Plot_Handler(
            x_label="Step",
            y_labels=self.loss_metrics.toJSON().keys()
            )

        total_runtime_start_time = time.time()

        for epoch in range(epochs) :
            for step in range(num_batches) :
                overhead_start_time = time.time()
                critic_start_time = time.time()

                for _ in range(5):
                    self.loss_metrics.critic_real_loss, \
                    self.loss_metrics.critic_synthetic_loss, \
                    self.loss_metrics.critic_interpolated_loss = self.train_critic(
                        x_train=x_train, 
                        batch_size=batch_size,
                        optimizer=critic_optimizer
                        )
                self.time_metrics.critic_step_time = time.time() - critic_start_time

                generator_start_time = time.time()
                self.loss_metrics.generator_loss = self.train_generator(batch_size=batch_size, optimizer=generator_optimizer)
                self.time_metrics.generator_step_time = time.time() - generator_start_time

                loss_plot_handler\
                    .update_plot(x_data=(epoch+1)*(step+1), y_data=self.loss_metrics.toJSON())\
                    .save_plot(save_path=os.path.join(save_folder,"viz/"))

                #  Generates synthetic images plot
                if step % 50 == 0:
                    synthetic_images_plot_handler\
                        .generate_images(model=self.generator.model)\
                        .plot_and_save_images(save_folder=os.path.join(save_folder,"viz/"))
                    
                

                self.time_metrics.overhead_step_time = (time.time() - overhead_start_time) - (self.time_metrics.generator_step_time + self.time_metrics.critic_step_time)
                self.time_metrics.total_runtime = time.time() - total_runtime_start_time

                # save metrics
                self.save_metrics(epoch=epoch, step=step, save_folder=save_folder, initialize_file=(epoch==0 and step==0))
                # Print training summary
                self.print_training_summary(epoch, epochs, step, num_batches)
                
        return self

    def save_metrics(self, epoch, step, save_folder : str, initialize_file = False):
        metrics_path = os.path.join(save_folder, "metrics/")
        if not DirectoryUtil.isValidDirectory(metrics_path):
            print(f"Directory Not Found at [{metrics_path}]")
            try : 
                DirectoryUtil.promptToCreateDirectory(metrics_path) # Throws value error
            except ValueError as e:
                print(e)
                return self
        
        if DirectoryUtil.isProtectedFile(metrics_path,WGAN_GP.TRAINING_METRICS_FILENAME) :
            print(f"ABORTING: File [{os.path.join(metrics_path, WGAN_GP.TRAINING_METRICS_FILENAME)}] is protected - cannot overwrite file")
            return self
        
        metrics = [{"Epoch" : epoch, "Step" : step, **self.loss_metrics.reduce_tensors().toJSON(), **self.time_metrics.toJSON()}]

        df = pd.DataFrame(metrics)

        csv_file = os.path.join(metrics_path, WGAN_GP.TRAINING_METRICS_FILENAME)
        # Save DataFrame to CSV with headers
        if initialize_file :
            df.to_csv(csv_file, index=False)
        else : 
            df.to_csv(csv_file, mode='a', header=False, index=False)

    
    def print_training_summary(self, epoch, epochs, step, steps):
        summary : str = f"""
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %%%%%%%% Training Summary %%%%%%%%%%%
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %
        %   Epoch:  {epoch + 1}/{epochs}
        %   Step:   {step+1}/{steps}
        %   
        %%%%   Critic Losses   %%%%
        %   Real Loss:          {tf.math.reduce_mean(self.loss_metrics.critic_real_loss)}
        %   Synthetic Loss:     {tf.math.reduce_mean(self.loss_metrics.critic_synthetic_loss)}
        %   Interpolated Loss   {tf.math.reduce_mean(self.loss_metrics.critic_interpolated_loss)}
        %
        %%%%   Generator Losses   %%%%
        %   Synthetic Loss:          {tf.math.reduce_mean(self.loss_metrics.generator_loss)}
        %
        %%%%   Time     %%%%
        %   Critic Time (seconds):    {self.time_metrics.critic_step_time}
        %   Generator Time (seconds): {self.time_metrics.generator_step_time}
        %   Overhead Time (seconds):  {self.time_metrics.overhead_step_time}
        %
        %   Total Runtime (hours):  {self.time_metrics.total_runtime / 3600}
        """

        if epoch == 0 and step == 0 : print("\n" * (summary.count('\n') + 1), end='', flush=True)
        print("\033[F" * (summary.count('\n') + 1) + summary, flush=True)
        return True


    @tf.function()
    def train_generator(
        self, 
        batch_size : int,
        optimizer : optimizers.Optimizer
    ) -> tf.Tensor:
        """
        Trains the generator model.

        Args:
            batch_size (int): Batch size for training.
            optimizer (optimizers.Optimizer): Optimizer for training.

        Returns:
            tf.Tensor: Loss of the generator.
        """
        latent_batch = tf.random.normal(shape=(batch_size, self.generator.latent_dim))
        y_true = tf.ones(shape=(batch_size, 1))
        with tf.GradientTape() as tape:
            synthetic_images = self.generator.model(latent_batch, training = True)
            belief_tensor = self.critic.model(synthetic_images, training = True)
            loss = GANLosses.wasserstein(y_true=y_true, y_pred=belief_tensor)

            grads = tape.gradient(loss, self.generator.model.trainable_variables)
            optimizer.apply_gradients(zip(grads, self.generator.model.trainable_variables))
        return loss
    
    @tf.function()
    def train_critic(
            self,
            x_train : np.ndarray,
            batch_size : int,
            optimizer : optimizers.Optimizer
    ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        """
        Trains the critic model.

        Args:
            x_train (np.ndarray): Training dataset.
            batch_size (int): Batch size for training.
            optimizer (optimizers.Optimizer): Optimizer for training.

        Returns:
            Tuple[tf.Tensor, tf.Tensor, tf.Tensor]: Tuple containing the real loss, synthetic loss, and interpolated loss.
        """
        def interpolate_images(synthetic_img, real_img):
            nonlocal batch_size
            alpha = K.random_uniform(shape=(batch_size,1,1,1))
            return (alpha * real_img) + ((1-alpha) * synthetic_img)
        
        
        valid = tf.ones(shape=(batch_size, 1))
        fake = -tf.ones(shape=(batch_size, 1))

        idx = tf.random.uniform(shape=(batch_size,), minval=0, maxval=x_train.shape[0], dtype=tf.int32)
        real_images = tf.gather(x_train, idx)

        latent_batch = tf.random.normal(shape=(batch_size, self.generator.latent_dim))
        synthetic_images = self.generator.model(latent_batch, training = True)

        with tf.GradientTape() as tape:
            real_belief_tensor = self.critic.model(real_images, training = True)
            synthetic_belief_tensor = self.critic.model(synthetic_images, training = True)
            with tf.GradientTape() as g:
                interpolated_images = interpolate_images(synthetic_images, real_images)
                g.watch(interpolated_images)
                interpolated_belief_tensor = self.critic.model(interpolated_images, training = True)
                gradients = g.gradient(interpolated_belief_tensor,interpolated_images)

            real_loss = GANLosses.wasserstein(y_true=valid, y_pred=real_belief_tensor)
            synthetic_loss = GANLosses.wasserstein(y_true=fake, y_pred=synthetic_belief_tensor)
            interpolated_loss = GANLosses.gradient_penalty_loss(gradients)

            
            loss = real_loss + synthetic_loss + interpolated_loss
            grads = tape.gradient(loss, self.critic.model.trainable_variables)
            optimizer.apply_gradients(zip(grads, self.critic.model.trainable_variables))
        
        return real_loss, synthetic_loss, interpolated_loss



    def save_architecture_diagram(self, path: str, file_name: str) -> Self:
        """
        Saves the architecture diagrams of the generator, critic, and complete model.

        Args:
            path (str): Path to the directory to save the diagrams.
            file_name (str): Prefix for the file names of the diagrams.

        Returns:
            Self: The instance of the class.
        """
        self.generator.save_architecture_diagram(path=path, file_name=file_name + "_" + WGAN_GP.GENERATOR_NAME + ".png")
        self.critic.save_architecture_diagram(path=path, file_name=file_name + "_" + WGAN_GP.CRITIC_NAME + ".png")
        return super().save_architecture_diagram(path=path, file_name=file_name+ "_" + WGAN_GP.MODEL_NAME + ".png")

    def save_model(self, path: str) -> Self:
        """
        Saves the model and its components to the specified path.

        Args:
            path (str): Path to the directory to save the model.

        Returns:
            Self: The instance of the class.
        """
        if not DirectoryUtil.isValidDirectory(path):
            print(f"Directory Not Found at [{path}]")
            try : 
                DirectoryUtil.promptToCreateDirectory(path) # Throws value error
            except ValueError as e:
                print(e)
                return Self
        
        if DirectoryUtil.isProtectedFile(path,WGAN_GP.CLASS_FILENAME) :
            print(f"ABORTING: File [{os.path.join(path, WGAN_GP.CLASS_FILENAME)}] is protected - cannot overwrite file")
            return self

        # Serialize and save the instance data as JSON
        with open(os.path.join(path, WGAN_GP.CLASS_FILENAME), 'wb') as file:
            pickle.dump(self, file)

        self.generator.save_model(path=os.path.join(path, WGAN_GP.GENERATOR_NAME + "/"))
        self.critic.save_model(path=os.path.join(path, WGAN_GP.CRITIC_NAME + "/"))
        return super().save_model(path=os.path.join(path, WGAN_GP.MODEL_NAME + "/"))
    
    @staticmethod
    def load_model(path : str) -> Self:
        """
        Loads the model from the specified path.

        Args:
            path (str): Path to the directory containing the saved model.

        Returns:
            Self: The instance of the class.
        """
        file_path = os.path.join(path, WGAN_GP.CLASS_FILENAME)
        with open(file_path, 'rb') as file:
             instance = pickle.load(file)
        
        self = instance
        return self
    
    def __str__(self) -> str:
        """
        Returns a string representation of the WGAN_GP object.

        Returns:
            str: String representation of the object.
        """
        stringSummary = []
        stringSummary.append(str(self.generator) + "\n\n")
        stringSummary.append(str(self.critic) + "\n\n")
        stringSummary.append(super().__str__())
        return "\n".join(stringSummary)

    


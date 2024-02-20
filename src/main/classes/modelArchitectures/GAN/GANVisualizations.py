from src.main.classes.managerUtils import DirectoryUtil

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from keras import Model
import tensorflow as tf
import numpy as np
import os
import math

from typing_extensions import Self
from typing import Tuple

class GANVisualizations():
    class Line_Plot_Handler:
        def __init__(self, x_label : str, y_labels : list[str], y_axis_title = "Loss", plot_title = "Loss Plot"):
            self.fig, self.ax = plt.subplots()
            self.x_label : str = x_label
            self.ax.set_xlabel(self.x_label)
            self.ax.set_ylabel(y_axis_title)
            self.ax.set_title(plot_title)
            self.ax.grid(True)
            self.y_labels = {label : [] for label in y_labels}
            self.data = {self.x_label: [], **self.y_labels}

        def update_plot(self, x_data : int, y_data : dict) -> Self:
            self.ax.clear()  # Clear the plot before updating

            # Update plot data
            self.ax.grid(True)
            self.data[self.x_label].append(x_data)
            for label, item in y_data.items():
                if label in self.y_labels :
                    self.data[label].append(item)
                    self.ax.plot(self.data[self.x_label], self.data[label], label=label, linewidth=0.5)

            self.ax.legend()
            return self

        def show_plot(self) -> Self:
            plt.show()
            return self

        def save_plot(self, save_path : str, file_name : str = "loss_plot.jpg") -> Self:
            if not DirectoryUtil.isValidDirectory(save_path):
                print(f"Directory Not Found at [{save_path}]")
                try : 
                    DirectoryUtil.promptToCreateDirectory(save_path) # Throws value error
                except ValueError as e:
                    print(e)
                    return self
            
            if DirectoryUtil.isProtectedFile(save_path,file_name) :
                print(f"ABORTING: File [{os.path.join(save_path, file_name)}] is protected - cannot overwrite file")
                return self
            
            self.fig.savefig(os.path.join(save_path,file_name))
            plt.close(self.fig)
            return self

    class Generator_Images_Plot_Handler:
        def __init__(self):
            self.latent_sample = None
            self.images = None
        
        def initialize_random_latent_samples(self, shape : Tuple[int,int]) -> Self :
            self.latent_sample = np.random.normal(size=shape)
            return self

        def update_latent_samples(self, samples : tf.Tensor) -> Self :
            self.latent_sample = samples
            return self
        
        def generate_images(self, model : Model) -> Self:
            if not self._validate_latent_sample() :
                print(f'Latent Sample Must Not Be None')
                return self
            self.images = model(self.latent_sample)
            return self
        
        def _validate_latent_sample(self) -> bool:
            if self.latent_sample is None :
                return False
            return True
        def _validate_images(self) -> bool :
            if self.images is None :
                return False
            return True


        # Function to plot and save the generated images
        def plot_and_save_images(self, save_folder : str, file_name : str = "generated_images.jpg") -> Self:
            if not self._validate_images() :
                print(f"Images not found...")
                return self
            if not DirectoryUtil.isValidDirectory(save_folder):
                print(f"Directory Not Found at [{save_folder}]")
                try : 
                    DirectoryUtil.promptToCreateDirectory(save_folder) # Throws value error
                except ValueError as e:
                    print(e)
                    return self
            
            if DirectoryUtil.isProtectedFile(save_folder,file_name) :
                print(f"ABORTING: File [{os.path.join(save_folder, file_name)}] is protected - cannot overwrite file")
                return None
            
            # Calculate the number of rows and columns for the grid layout
            num_images = len(self.images)
            max_columns = math.ceil(num_images**0.5)
            num_rows = math.ceil(num_images / max_columns)  # Adjust the number as needed
            num_cols = min(num_images, max_columns)  # Maximum 4 columns

            # Create subplots with the grid layout
            fig, axes = plt.subplots(num_rows, num_cols, figsize=(10, 10),gridspec_kw={'wspace': 0.1, 'hspace': 0.1})
            for i, img in enumerate(self.images):
                row = i // num_cols
                col = i % num_cols
                axes[row, col].imshow(img)
                axes[row, col].axis('off')

            # Hide any extra empty subplots
            for i in range(num_images, num_rows * num_cols):
                axes.flatten()[i].axis('off')

            
            plt.savefig(os.path.join(save_folder, file_name), bbox_inches='tight')
            plt.close()


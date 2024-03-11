from dataclasses import dataclass
import tensorflow as tf

from typing_extensions import Self

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
class accuracy_metrics:
    critic_real_accuracy : tf.Tensor | float = None
    critic_synthetic_accuracy : tf.Tensor | float = None

    def reduce_tensors(self) -> Self:
        if isinstance(self.critic_real_accuracy, tf.Tensor):
            self.critic_real_accuracy = tf.math.reduce_mean(self.critic_real_accuracy).numpy()
        if isinstance(self.critic_synthetic_accuracy, tf.Tensor):
            self.critic_synthetic_accuracy = tf.math.reduce_mean(self.critic_synthetic_accuracy).numpy()
        return self


    def toJSON(self) :
        return {
            "Critic Real Accuracy" : self.critic_real_accuracy,
            "Critic Synthetic Accuracy" : self.critic_synthetic_accuracy,
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
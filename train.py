from models.Mamba import  Mamba
import tensorflow as tf

mb=Mamba()
mb.generate(tf.constant([1]),tf.constant(15))
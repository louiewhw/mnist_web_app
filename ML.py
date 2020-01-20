from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
import tensorflow as tf


mnist = tf.keras.datasets.mnist.load_data()


print(mnist)

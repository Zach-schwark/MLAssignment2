import tensorflow as tf; 
import numpy as np
import helpers
print(tf.reduce_sum(tf.random.normal([1000, 1000])))
print(tf.config.list_physical_devices('GPU'))

mnist = tf.keras.datasets.mnist
helpers.load_mnist()

(x_train, y_train), (x_test, y_test) = helpers.load_mnist()
x_train, x_test = x_train / 255.0, x_test / 255.0

model = tf.keras.models.Sequential([
  tf.keras.layers.Dense(784),
  tf.keras.layers.Dense(200, activation='relu'),
  tf.keras.layers.Dense(100, activation='relu'),
  #tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10)
])

predictions = model(x_train[:1]).numpy()
predictions
tf.nn.softmax(predictions).numpy()
print(tf.nn.softmax(predictions).numpy())
print(np.shape(predictions))
loss_fn = tf.keras.losses.LogCosh(reduction=tf.keras.losses.Reduction.SUM)
print(np.shape(predictions))
print(np.shape(y_train[:1]))
loss_fn(y_train[:1], predictions).numpy()
model.compile(optimizer='adam',
              loss=loss_fn,
              metrics=['accuracy'])
model.fit(x_train, y_train, epochs=5)
model.evaluate(x_test,  y_test, verbose=2)

from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.callbacks import Callback


class MyCallback(Callback):
    def on_epoch_end(self, epochs, logs=None):
        if logs['accuracy'] > 0.85 and logs['accuracy'] - logs['val_accuracy'] > 0.03:
            print('\ntraining accuracy reached 0.85 '
                  'and difference between training accuracy and validation accuracy reached 0.3')
            self.model.stop_training = True


(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
train_images = train_images / 255.0
test_images = test_images / 255.0

model = Sequential([
    Flatten(input_shape=(28, 28)),
    Dense(units=128, activation='relu'),
    Dense(units=10, activation='softmax')
])
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
callback = MyCallback()
model.fit(train_images, train_labels, validation_data=(test_images, test_labels), epochs=50, callbacks=[callback])
model.evaluate(test_images, test_labels)

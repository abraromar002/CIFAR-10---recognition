

# 🧠 Image Classification using Deep Learning (Basic NN + ResNet50)
This project demonstrates image classification on a dataset with 10 classes using two different deep learning approaches:

A simple neural network built from scratch.

A more advanced model using ResNet50 pretrained on ImageNet, fine-tuned for our task.

# 📂 Project Structure
bash
├── data/
│   ├── train/
│   └── test/
├── models/
│   ├── basic_nn.h5
│   └── resnet50_model.h5
├── notebooks/
│   └── classification.ipynb
├── README.md
🧰 Technologies Used
Python 3

TensorFlow / Keras

NumPy, Matplotlib

Pretrained model: ResNet50

# 🚀 How it Works
1. Preprocessing
Images resized to (32, 32, 3) for the basic model.

Resized to (256, 256, 3) for ResNet50.

Pixel values scaled to range [0, 1].

python

X_train_scaled = X_train / 255.0
X_test_scaled = X_test / 255.0
2. Basic Neural Network
python

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(32,32,3)),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])
Loss: sparse_categorical_crossentropy

Optimizer: Adam

Accuracy ~ X%

3. ResNet50-based Model
Uses pretrained weights from imagenet.

Adds multiple UpSampling layers to match ResNet's input size.

Adds custom dense layers and dropout for regularization.

python

model = Sequential()
model.add(UpSampling2D((2,2)))
model.add(UpSampling2D((2,2)))
model.add(UpSampling2D((2,2)))
model.add(convolutional_base)  # ResNet50 without top
model.add(Flatten())
...
model.add(Dense(10, activation='softmax'))
Final accuracy: ~93.86% on test data

Optimizer: RMSprop with learning rate 2e-5

# 📊 Results
Validation and training accuracy/loss plotted using matplotlib.

Final model achieves high accuracy using transfer learning.

# ✅ Example Accuracy Output:
bash


313/313 [==============================] - 40s 124ms/step - loss: 0.2318 - acc: 0.9386
Test Accuracy = 0.9386
📈 Visualizations


# ✅ Future Work
Add image augmentation to prevent overfitting.

Explore more pretrained architectures (e.g., VGG16, EfficientNet).

Export model as .tflite for mobile deployment.


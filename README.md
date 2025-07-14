# Project Overview
This project is a from-scratch implementation of a simple neural network to classify handwritten digits from the MNIST dataset — using only NumPy, Pandas, and Matplotlib. No machine learning libraries like TensorFlow or PyTorch were used.

## Model Architecture
The model architecture includes:

- An input layer with 784 neurons (one for each pixel in a 28x28 image)  
- Two hidden layers with 10 neurons each  
- An output layer with 10 neurons (representing digits 0 through 9)

## Key Features
- Forward and backward propagation coded manually  
- Softmax in the output layer to turn logits into probabilities  
- One-hot encoding for the labels to work smoothly with the loss function  
- I started with ReLU as the activation function in the hidden layers, but later switched to tanh after seeing better results

## Results and Observations
At first, I used ReLU, which is common in deeper networks. But in this simpler setup, ReLU led to lower accuracy and less stable training. When I swapped it out for tanh, the network’s accuracy improved noticeably — training was smoother, and validation accuracy increased significantly.

With the right hyperparameters (a learning rate of 0.1 and 500 training iterations), the model reached around 80% accuracy on the validation set.

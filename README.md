Performing backpropagation on a neural network involves calculating the gradients of the loss function with respect to each weight and bias in the network and then updating these weights and biases using gradient descent. Below, I'll outline the steps and formulas for backpropagation and gradient descent for a simple neural network with:

- 1 input layer with 2 neurons
- 1 hidden layer with 2 neurons
- 1 output layer with 1 neuron

### Network Architecture
Let's denote:
- \( x_1, x_2 \): Inputs
- \( h_1, h_2 \): Hidden layer neurons
- \( o \): Output neuron
- \( w_{ij} \): Weight from neuron \( i \) to neuron \( j \)
- \( b_i \): Bias for neuron \( i \)

### Step-by-Step Backpropagation

1. **Forward Pass**:
   - Compute activations for the hidden layer:
     \[
     h_1 = \sigma(w_{11}x_1 + w_{12}x_2 + b_1)
     \]
     \[
     h_2 = \sigma(w_{21}x_1 + w_{22}x_2 + b_2)
     \]
   - Compute activation for the output layer:
     \[
     o = \sigma(w_{31}h_1 + w_{32}h_2 + b_3)
     \]

2. **Loss Function**:
   - Use Mean Squared Error (MSE) for simplicity:
     \[
     L = \frac{1}{2}(y - o)^2
     \]
   where \( y \) is the true label.

3. **Backward Pass**:

   - Compute the gradient of the loss with respect to the output activation:
     \[
     \frac{\partial L}{\partial o} = -(y - o)
     \]

   - Compute the gradient of the output activation with respect to the weighted sum of inputs to the output neuron (using the derivative of the sigmoid function):
     \[
     \frac{\partial o}{\partial z_3} = o(1 - o)
     \]
     where \( z_3 = w_{31}h_1 + w_{32}h_2 + b_3 \).

   - Compute the gradient of the loss with respect to the weighted sum of inputs to the output neuron:
     \[
     \frac{\partial L}{\partial z_3} = \frac{\partial L}{\partial o} \cdot \frac{\partial o}{\partial z_3} = -(y - o) \cdot o(1 - o)
     \]

   - Compute the gradients of the loss with respect to the weights and biases of the output neuron:
     \[
     \frac{\partial L}{\partial w_{31}} = \frac{\partial L}{\partial z_3} \cdot h_1
     \]
     \[
     \frac{\partial L}{\partial w_{32}} = \frac{\partial L}{\partial z_3} \cdot h_2
     \]
     \[
     \frac{\partial L}{\partial b_3} = \frac{\partial L}{\partial z_3}
     \]

   - Compute the gradient of the loss with respect to the hidden layer activations:
     \[
     \frac{\partial L}{\partial h_1} = \frac{\partial L}{\partial z_3} \cdot w_{31}
     \]
     \[
     \frac{\partial L}{\partial h_2} = \frac{\partial L}{\partial z_3} \cdot w_{32}
     \]

   - Compute the gradients of the loss with respect to the weights and biases of the hidden neurons:
     \[
     \frac{\partial L}{\partial w_{11}} = \frac{\partial L}{\partial z_1} \cdot x_1
     \]
     \[
     \frac{\partial L}{\partial w_{12}} = \frac{\partial L}{\partial z_1} \cdot x_2
     \]
     \[
     \frac{\partial L}{\partial w_{21}} = \frac{\partial L}{\partial z_2} \cdot x_1
     \]
     \[
     \frac{\partial L}{\partial w_{22}} = \frac{\partial L}{\partial z_2} \cdot x_2
     \]
     \[
     \frac{\partial L}{\partial b_1} = \frac{\partial L}{\partial z_1}
     \]
     \[
     \frac{\partial L}{\partial b_2} = \frac{\partial L}{\partial z_2}
     \]

4. **Gradient Descent Update**:
   - Update the weights and biases using the computed gradients:
     \[
     w_{ij} = w_{ij} - \eta \cdot \frac{\partial L}{\partial w_{ij}}
     \]
     \[
     b_i = b_i - \eta \cdot \frac{\partial L}{\partial b_i}
     \]
     where \( \eta \) is the learning rate.

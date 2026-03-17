from .Layer import Layer
import numpy as np


class Dense(Layer):
  def __init__(self, units: int, inputs:int=0, kernel_initializer: str=None):
    super().__init__()

    self.units = units
    self.in_size = inputs
    self.out_size = units
    self.kernel_initializer = kernel_initializer

    if not kernel_initializer:
      self._set_default_initializers()


  def _set_default_initializers(self):
      self.kernel_initializer = "he"
      """if self.activation == "relu":
          self.initializer = "he"
      elif self.activation in ["sigmoid", "tanh", "softmax", "linear"]:
          self.initializer = "xavier"
      else:
          raise Exception("Invalid activation function")"""

  def build(self, input_size):
      self.in_size = input_size
      self._initialize_weights()

      self.b = np.zeros((1, self.out_size), dtype=np.float32)

      # Optimizer state
      self.vW = np.zeros_like(self.W).astype(np.float32)
      self.vb = np.zeros_like(self.b).astype(np.float32)
      self.mW = np.zeros_like(self.W).astype(np.float32)
      self.mb = np.zeros_like(self.b).astype(np.float32)

  def _initialize_weights(self):

      if self.kernel_initializer == "he":
          # W ~ N(0, √(2/in))
          std = np.sqrt(2.0 / self.in_size)
          self.W = np.random.randn(self.in_size, self.out_size).astype(np.float32) * std

      elif self.kernel_initializer == "xavier":
          # U(-√(6/(in+out)),√(6/(in+out)))
          limit = np.sqrt(6.0 / (self.in_size + self.out_size))
          self.W = np.random.uniform(-limit, limit, (self.in_size, self.out_size)).astype(np.float32)

      else:
          self.W = np.random.randn(self.in_size, self.out_size).astype(np.float32) * 0.01

  def _forward(self, A_prev, training=None):
      if self.W is None:
          self.build(A_prev.shape[1])

      self.A_prev = A_prev
      self.Z = A_prev @ self.W + self.b
      self.A = self.Z

      return self.Z

  def _backward(self, dA, skip_activation=False):

      # dW_i = dZ_i · A_{i}^T
      # Gradient of the loss w.r.t. weights of layer i
      self.dW = self.A_prev.T @ dA

      # dB_i = sum(dZ_i) over the batch
      # Gradient of the loss w.r.t. biases of layer i
      self.db = np.sum(dA, axis=0, keepdims=True)

      # Gradient to pass backward
      # dA_prev_i = W_i · dA_i
      return dA @ self.W.T


  def _update(self, lambda_, lr, beta1, beta2, _eps, optimizer, t):

      if optimizer == "adamW":
          dw = self.dW # pure gradient
      else:
          # Get batch size from the last dZ calculation to scale regularization
          m = self.A.shape[0]
          dw = self.dW + (lambda_ / m) * self.W # L2 regularization

      if optimizer == "momentum":
        self.vW = beta1 * self.vW + dw
        self.vb = beta1 * self.vb + self.db

        update_w = self.vW
        update_b = self.vb

      elif optimizer == "adam" or optimizer == "adamW":

        self.mW = beta1 * self.mW + (1 - beta1) * dw
        self.mb = beta1 * self.mb + (1 - beta1) * self.db

        self.vW = beta2 * self.vW + (1 - beta2) * (dw ** 2)
        self.vb = beta2 * self.vb + (1 - beta2) * (self.db ** 2)

        m_w_hat = self.mW / (1 - beta1 ** t)
        m_b_hat = self.mb / (1 - beta1 ** t)

        v_w_hat = self.vW / (1 - beta2 ** t)
        v_b_hat = self.vb / (1 - beta2 ** t)


        update_w = m_w_hat / (np.sqrt(v_w_hat) + _eps)
        update_b = m_b_hat / (np.sqrt(v_b_hat) + _eps)

      elif optimizer == "rmsprop":
        self.vW = beta2 * self.vW + (1 - beta2) * (dw ** 2)
        self.vb = beta2 * self.vb + (1 - beta2) * (self.db ** 2)

        update_w = dw / (np.sqrt(self.vW) + _eps)
        update_b = self.db / (np.sqrt(self.vb) + _eps)

      else:
        # Classic SGD
        update_w = dw
        update_b = self.db


      # W = W - lr * (dW + λ * W)
      self.W -= lr * update_w
      self.b -= lr * update_b

      # Decouple (adamW) Weight Decay
      if optimizer == "adamW":
        self.W *= (1 - lr * lambda_)  # Decoupled weight decay



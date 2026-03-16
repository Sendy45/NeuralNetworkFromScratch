import numpy as np
from tqdm.auto import tqdm

from . import Flatten
from .layers import Dropout, Activation, BatchNorm, Dense, Conv2D, MaxPooling2D, AveragePooling2D
from .utils import History

class NeuralNetwork:
    def __init__(self, layers:list):

        self.num_classes = 0
        self.layers = layers

        self._eps = 1e-08 # Avoid dividing by zero
        self.lr = 0.001
        self.lambda_ = 0.0
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.loss_type = "cross_entropy"
        self.optimizer = "adam"

    def save(self, path):
        layer_data = []

        for layer in self.layers:
            entry = {"type": type(layer).__name__}

            if isinstance(layer, Dense):
                entry.update({
                    "W": layer.W,
                    "b": layer.b,
                    "in_size": layer.in_size,
                    "out_size": layer.out_size,
                    "initializer": layer.initializer,
                })
            elif isinstance(layer, BatchNorm):
                entry.update({
                    "gamma": layer.gamma,
                    "beta": layer.beta,
                    "running_mean": layer.running_mean,
                    "running_var": layer.running_var,
                    "momentum": layer.momentum,
                })
            elif isinstance(layer, Dropout):
                entry["rate"] = layer.rate
            elif isinstance(layer, Conv2D):
                entry.update({
                    "W": layer.W,
                    "b": layer.b,
                    "initializer": layer.initializer,
                    "filters": layer.filters,
                    "kernel_size": layer.kernel_size,
                    "padding": layer.padding,
                    "_padding_val": layer._padding_val,
                    "strides": layer.strides,
                    "in_size": layer.in_size,
                    "out_size": layer.out_size,
                })
            elif isinstance(layer, (MaxPooling2D, AveragePooling2D)):
                entry.update({
                    "pool_size": layer.pool_size,
                    "strides": layer.strides,
                    "padding": layer.padding,
                    "_padding_val": layer._padding_val,
                })

            # Activation layers (ReLu, Sigmoid, etc.) and Flatten layer need no extra data

            layer_data.append(entry)

        np.savez(
            path,
            layers=np.array(layer_data, dtype=object),
            lr=self.lr,
            lambda_=self.lambda_,
            beta1=self.beta1,
            beta2=self.beta2,
            loss_type=self.loss_type,
            optimizer=self.optimizer,
            num_classes=self.num_classes,
        )


    @staticmethod
    def load(path):
        from .layers import Dense, BatchNorm, Dropout, ReLu, Sigmoid, Softmax, Tanh, Linear

        ACTIVATION_MAP = {
            "ReLu": ReLu,
            "Sigmoid": Sigmoid,
            "Softmax": Softmax,
            "Tanh": Tanh,
            "Linear": Linear,
        }

        data = np.load(path, allow_pickle=True)
        layers = []

        for entry in data["layers"]:
            layer_type = entry["type"]

            if layer_type == "Dense":
                layer = Dense(units=entry["out_size"])
                layer.W = entry["W"]
                layer.b = entry["b"]
                layer.in_size = entry["in_size"]
                layer.out_size = entry["out_size"]
                layer.initializer = entry["initializer"]
                # Restore optimizer states as zeros (not serialized)
                layer.vW = np.zeros_like(layer.W)
                layer.vb = np.zeros_like(layer.b)
                layer.mW = np.zeros_like(layer.W)
                layer.mb = np.zeros_like(layer.b)

            elif layer_type == "BatchNorm":
                layer = BatchNorm(momentum=entry["momentum"])
                layer.gamma = entry["gamma"]
                layer.beta = entry["beta"]
                layer.running_mean = entry["running_mean"]
                layer.running_var = entry["running_var"]

            elif layer_type == "Dropout":
                layer = Dropout(rate=entry["rate"])

            elif layer_type == "Conv2D":
                layer = Conv2D(filters=entry["filters"],
                kernel_size=entry["kernel_size"])
                layer.W = entry["W"]
                layer.b = entry["b"]
                layer.in_size = entry["in_size"]
                layer.out_size = entry["out_size"]
                layer.initializer = entry["initializer"]
                layer.filters = entry["filters"]
                layer.kernel_size = entry["kernel_size"]
                layer.padding = entry["padding"]
                layer._padding_val = entry["_padding_val"]
                layer.strides = entry["strides"]

            elif layer_type == "MaxPooling2D":
                layer = MaxPooling2D(pool_size=entry["pool_size"])
                layer.strides = entry["strides"]
                layer.padding = entry["padding"]
                layer._padding_val = entry["_padding_val"]

            elif layer_type == "AveragePooling2D":
                layer = AveragePooling2D(pool_size=entry["pool_size"])
                layer.strides = entry["strides"]
                layer.padding = entry["padding"]
                layer._padding_val = entry["_padding_val"]

            elif layer_type == "Flatten":
                layer = Flatten()

            elif layer_type in ACTIVATION_MAP:
                layer = ACTIVATION_MAP[layer_type]()

            else:
                raise ValueError(f"Unknown layer type: {layer_type}")

            layers.append(layer)

        model = NeuralNetwork(layers)
        model.lr = data["lr"].item()
        model.lambda_ = data["lambda_"].item()
        model.beta1 = data["beta1"].item()
        model.beta2 = data["beta2"].item()
        model.loss_type = data["loss_type"].item()
        model.optimizer = data["optimizer"].item()
        model.num_classes = data["num_classes"].item()

        return model


    def summary(self):
        ll = 60 # Line Length
        print("=" * ll)
        print("Model Summary")
        print("=" * ll)

        total_params = 0

        for i, layer in enumerate(self.layers):
            layer_type = type(layer).__name__

            if isinstance(layer, Dense):
                params = layer.W.size + layer.b.size if layer.W is not None else 0
                total_params += params
                built = f"{layer.in_size} → {layer.out_size}"
                print(f"[{i+1}] Dense          {built:<20} params: {params}")

            elif isinstance(layer, BatchNorm):
                params = layer.gamma.size + layer.beta.size if layer.gamma is not None else 0
                total_params += params
                print(f"[{i+1}] BatchNorm      momentum={layer.momentum:<13} params: {params}")

            elif isinstance(layer, Dropout):
                print(f"[{i+1}] Dropout        rate={layer.rate}")


            elif isinstance(layer, Conv2D):

                params = layer.W.size + layer.b.size if layer.W is not None else 0
                total_params += params

                k_h, k_w = layer.kernel_size
                s_h, s_w = layer.strides

                built = f"{layer.filters} filters {k_h}x{k_w}"

                print(f"[{i + 1}] Conv2D         {built} stride=({s_h},{s_w}) params: {params}")


            elif isinstance(layer, MaxPooling2D):

                p_h, p_w = layer.pool_size
                s_h, s_w = layer.strides

                print(f"[{i + 1}] MaxPooling2D   pool=({p_h},{p_w}( stride=({s_h},{s_w})")

            elif isinstance(layer, AveragePooling2D):

                p_h, p_w = layer.pool_size
                s_h, s_w = layer.strides

                print(f"[{i + 1}] AveragePooling2D   pool=({p_h},{p_w}( stride=({s_h},{s_w})")

            else:
                print(f"[{i+1}] {layer_type:<15}")

            print("-" * ll)

        print(f"Total trainable parameters: {total_params}")
        print("=" * ll)


    # One hot encoding for y_true - convert format into a matrix for calculations
    def _one_hot_encoding(self, y):
        one_hot_y = np.zeros((y.size, self.num_classes))
        one_hot_y[np.arange(y.size), y] = 1
        return one_hot_y

    """ **********************************************************
    Model Algorithms
    ********************************************************** """

    # Forward function - feed input and get prediction
    def _forward(self, X, training=True):
        # Z = W * A + B
        # A - output (after activation function)
        # Also next layer input
        for layer in self.layers:
            X = layer._forward(X, training=training)
        return X


    # Backward function - locates the origin of the loss and tweaks it
    def _backward(self, y_true):
      m = y_true.size


      dA = self._loss_derivative(self.layers[-1].A, y_true) / m
      dA = self.layers[-1]._backward(dA)

      # Remaining layers
      # Backpropagation: iterate layers in reverse order (from last to first)
      # Compute gradients dW and dB for each layer
      for layer in reversed(self.layers[:-1]):
          # dA = ∂J/∂A_L
          # This is the derivative of the loss w.r.t. the network output
          dA = layer._backward(dA)


    # Loss derivative - for the last layer based on the loss-type
    def _loss_derivative(self, y_pred, y_true):
        one_hot = self._one_hot_encoding(y_true) # y_true formatting
        # Division by m happens in backward function
        if self.loss_type == "cross_entropy":
            # - y_true / y_pred
            return -(one_hot / (y_pred + self._eps))

        elif self.loss_type == "mse":
            # 2 * (y_pred - y_true)
            return 2 * (y_pred - one_hot)

        else:
            raise Exception("Invalid loss function")


    # Compute loss for logging
    def _compute_loss(self, y_pred, y_true):
        one_hot = self._one_hot_encoding(y_true)  # shape: (num_classes, N)
        m = y_true.size

        if self.loss_type == "cross_entropy":
          # -1/m * ∑ (y_true * log(y_pred))
          data_loss = -1 * np.sum(one_hot * np.log(y_pred + self._eps)) / m
        elif self.loss_type == "mse":
          # 1/m * ∑ ((y_pred - y_true)^2)
          data_loss = np.mean((y_pred - one_hot) ** 2)
        else:
          raise Exception("Invalid loss function")

        # L2 regularization term: (λ / 2m) * sum(||W||^2)
        reg_loss = 0.0
        for layer in self.layers:
            if hasattr(layer, 'W') and layer.W is not None:
                reg_loss += np.sum(layer.W ** 2)

        reg_loss = (self.lambda_ / (2 * m)) * reg_loss

        return data_loss + reg_loss


    # Update weights and biases
    # lr = learning rate
    # high lr - impact the model fast, can overshoot
    # low lr - learns slower, wont overshoot
    # Lambda λ = Counter overfitting by punishing big weights
    # Force weights to be small but not zero (w = 0 -> no impact on the model)
    # Beta1 = momentum factor
    # Beta2 = RSMprop factor
    def _update(self, optimizer_t):

      for layer in self.layers:
          layer._update(self.lambda_, self.lr, self.beta1, self.beta2, self._eps, self.optimizer, optimizer_t)

    """ **********************************************************
    Metrics
    ********************************************************** """

    # Calculate model accuracy
    @staticmethod
    def accuracy(predictions, y):
      return np.sum(predictions==y) / y.size


    # Calculate model precision
    @staticmethod
    def precision(predictions, y, num_classes):
      precisions = []

      for c in range(num_classes):
        tp = np.sum((predictions == c) & (y == c))
        fp = np.sum((predictions == c) & (y != c))

        precisions.append(tp / (tp + fp + 1e-8))

      return np.mean(precisions)


    # Calculate model recall
    @staticmethod
    def recall(predictions, y, num_classes):
      recalls = []

      for c in range(num_classes):
        tp = np.sum((predictions == c) & (y == c))
        fn = np.sum((predictions != c) & (y == c))

        recalls.append(tp / (tp + fn + 1e-8))

      return np.mean(recalls)


    # Calculate model f1
    @staticmethod
    def f1(predictions, y, num_classes):
      precision = NeuralNetwork.precision(predictions, y, num_classes)
      recall = NeuralNetwork.recall(predictions, y, num_classes)
      return 2 * (precision * recall) / (precision + recall + 1e-8)


    def calc_metrics(self, history:History, y_pred, y_true, metrics=None):
      if metrics is None:
          metrics = []
      for metric in metrics:
        if metric == "accuracy":
          accuracy = NeuralNetwork.accuracy(y_pred, y_true)
          history.add("accuracy", accuracy)
        if metric == "precision":
          precision = NeuralNetwork.precision(y_pred, y_true, self.num_classes)
          history.add("precision", precision)
        if metric == "recall":
          recall = NeuralNetwork.recall(y_pred, y_true, self.num_classes)
          history.add("recall", recall)
        if metric == "f1":
          f1 = NeuralNetwork.f1(y_pred, y_true, self.num_classes)
          history.add("f1", f1)

      return history

    """ **********************************************************
    Runtime functions
    ********************************************************** """

    # Converts final layer activation to predicted class labels
    @staticmethod
    def _decode_output(output):
      if output.shape[1] == 1:  # Binary classification
          return (output > 0.5).astype(int).flatten()
      else:  # Multi-class (softmax)
          return np.argmax(output, axis=1)

    @staticmethod
    def shuffle_data(x, y):
        perm = np.random.permutation(y.size)
        x = x[perm] if x.ndim > 2 else x[:, perm]
        y = y[perm]
        return x, y

    @staticmethod
    def set_seed(seed):
      np.random.seed(seed)


    def check_gradient(self, X, y):
      assert X.shape[1] == y.size, f"X has {X.shape[1]} samples but y has {y.size}"

      # Use a small batch to avoid numerical issues
      X = X[:, :8].astype(np.float64)  # <-- float64 is critical for numerical grad
      y = y[:8]

      rel_diff = []
      original_lambda = self.lambda_
      self.lambda_ = 0.0
      epsilon = 1e-5  # smaller epsilon can help

      self._forward(X, training=False)
      self._backward(y)

      # Snapshot ALL analytical gradients before any weight perturbation
      analytical_grads = {}
      for idx, layer in enumerate(self.layers):
          if hasattr(layer, 'dW') and layer.dW is not None:
              analytical_grads[idx] = layer.dW.copy()  # <-- copy before perturbation

      for idx, layer in enumerate(self.layers):
          if hasattr(layer, 'W') and layer.W is not None:
              i = np.random.randint(0, layer.W.shape[0])
              j = np.random.randint(0, layer.W.shape[1])

              W_orig = layer.W[i, j]
              grad_analytical = analytical_grads[idx][i, j]

              layer.W[i, j] = W_orig + epsilon
              y_pred = self._forward(X, training=False)
              loss_plus = self._compute_loss(y_pred, y)

              layer.W[i, j] = W_orig - epsilon
              y_pred = self._forward(X, training=False)
              loss_minus = self._compute_loss(y_pred, y)

              grad_numerical = (loss_plus - loss_minus) / (2 * epsilon)
              layer.W[i, j] = W_orig

              numerator = abs(grad_numerical - grad_analytical)
              denominator = abs(grad_numerical) + abs(grad_analytical) + 1e-10
              rel_diff.append(numerator / denominator)

              print(f"Layer {idx} W[{i},{j}]  Numerical: {grad_numerical:.10f}  Analytical: {grad_analytical:.10f}  Rel diff: {rel_diff[-1]:.2e}")

      self.lambda_ = original_lambda
      return rel_diff


    # Compile the model and trains it
    # x - input features
    # y - labels
    # lr - learning rate
    # epochs - num of iterations over the data
    # batch_size - data size before model updating
    # loss_type - loss functions used
    # lambda_ - lambda used for preventing weights overfitting
    def gradient_descent(self, X, y, X_val=None, y_val=None, epochs=10, batch_size=1):
      X = X.astype(np.float32)
      history = History()

      optimizer_t = 0

      stopping_patience = 5

      for ep in range(epochs):
        predictions = []
        epoch_loss = 0

        x_shuffled, y_shuffled = NeuralNetwork.shuffle_data(X, y)

        n_samples = X.shape[0]

        # Batches
        # tqdm - loading animation
        for i in tqdm(range(0, n_samples, batch_size)):

          optimizer_t += 1

          # get batch
          x_batch = x_shuffled[i:i + batch_size] if X.ndim > 2 else x_shuffled[:, i:i + batch_size]
          y_batch = y_shuffled[i:i + batch_size]
          # feed model
          self._forward(x_batch)
          self._backward(y_batch)
          self._update(optimizer_t)

          y_pred = self.layers[-1].A
          # Monitor loss - epoch_loss = Avg(batches_loss)
          predictions.append(self._decode_output(y_pred))
          batch_loss = self._compute_loss(y_pred, y_batch)
          epoch_loss += batch_loss * n_samples / X.shape[1]

          # Check Gradient - make sure backpropagation works well
          #self.check_gradient(x_batch, y_batch)

        predictions = np.concatenate(predictions)

        history.add("epoch", ep+1)
        history.add("loss", epoch_loss)
        history = self.calc_metrics(history, predictions, y_shuffled, metrics=["accuracy", "precision", "recall"])

        # Validation
        if X_val is not None and y_val is not None:
          val_pred = self.predict_proba(X_val)
          val_loss = self._compute_loss(val_pred, y_val)
          val_acc = NeuralNetwork.accuracy(self._decode_output(val_pred), y_val)
          history.add("val_loss", val_loss)
          history.add("val_accuracy", val_acc)

          history.progress()

          if len(history.history["val_loss"]) > 1:
            if history.history["val_loss"][-1] > history.history["val_loss"][-2]:
                stopping_patience -= 1
            if stopping_patience == 0:
                print("Early stopping")
                break
        else:
          history.progress()

      return history


    """ **********************************************************
    Model API functions
    ********************************************************** """

    # Configure training hyperparameters and optimization settings
    def compile(self, loss_type="cross_entropy", optimizer="adam", lr=0.001, lambda_=0.0, beta1=0.9, beta2=0.999):
        self.lr = lr
        self.lambda_ = lambda_
        self.beta1 = beta1
        self.beta2 = beta2
        self.loss_type = loss_type
        self.optimizer = optimizer


    # Train the model
    def fit(self, X, y, X_val=None, y_val=None, epochs=10, batch_size=1):
        """if X.shape[1] != y.size:
            raise ValueError("Mismatch between samples and labels")"""

        # Find last layer - last outsize is num classes
        for layer in reversed(self.layers):
            if hasattr(layer, "out_size"):
                self.num_classes = layer.out_size
                break
        return self.gradient_descent(X, y, X_val=X_val, y_val=y_val, epochs=epochs, batch_size=batch_size)


    # Add layers to model
    def add(self, layer):
        self.layers.append(layer)


    # Return predicted class labels for input data
    def predict(self, X):
        return self._decode_output(self.predict_proba(X))


    # Return raw output activations (probabilities or scores)
    def predict_proba(self, X):
        return self._forward(X, training=False)


    # Evaluate model performance (accuracy) on given dataset
    def evaluate(self, X, y):
        predictions = self.predict(X)
        return NeuralNetwork.accuracy(predictions, y)
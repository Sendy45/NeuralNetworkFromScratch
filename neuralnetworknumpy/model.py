import numpy as np
from tqdm.auto import tqdm
import pickle, os

from .layers import *
from .utils import History

class NeuralNetwork:
    def __init__(self, layers:list):

        self.task = None
        self.num_classes = 0
        self.layers = layers

        self._eps = 1e-08 # Avoid dividing by zero
        self.lr = 0.001
        self.lambda_ = 0.0
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.loss_type = "cross_entropy"
        self.optimizer = "adam"

    """ **********************************************************
        Utils Functions
    ********************************************************** """

    def save(self, path):
        """
          Serialise the full model to a .pkl file.

          pickle writes directly to disk, keeping peak memory proportional to
          the largest single weight tensor rather than the whole model.
        """

        if not path.endswith(".pkl"):
            path = path + ".pkl"

        # Strip large cached training intermediates before saving.
        # These (patches, cols, A_pad, A_prev, etc.) are rebuilt on the next
        # forward pass and can be several times larger than the weights.
        def _strip(layer):
            for attr in ("p2d", "cols", "patches", "A_pad", "A_prev",
                         "x_mu", "X_hat", "Z", "dW", "db", "A"):
                layer.__dict__.pop(attr, None)
            if hasattr(layer, "layers"):
                for sub in layer.layers:
                    _strip(sub)
            if hasattr(layer, "projection") and layer.projection is not None:
                _strip(layer.projection)
            if hasattr(layer, "depthwise"):
                _strip(layer.depthwise)
            if hasattr(layer, "pointwise"):
                _strip(layer.pointwise)

        for layer in self.layers:
            _strip(layer)

        with open(path, "wb") as f:
            pickle.dump({
                "layers": self.layers,
                "lr": self.lr,
                "lambda_": self.lambda_,
                "beta1": self.beta1,
                "beta2": self.beta2,
                "loss_type": self.loss_type,
                "optimizer": self.optimizer,
                "num_classes": self.num_classes,
            }, f, protocol=pickle.HIGHEST_PROTOCOL)

        print(f"Model saved → {path}  ({os.path.getsize(path) / 1e6:.1f} MB)")

    @staticmethod
    def load(path):
        """Load a model saved with model.save()."""
        import pickle

        if not path.endswith(".pkl"):
            path = path + ".pkl"

        with open(path, "rb") as f:
            data = pickle.load(f)

        model = NeuralNetwork(data["layers"])
        model.lr = data["lr"]
        model.lambda_ = data["lambda_"]
        model.beta1 = data["beta1"]
        model.beta2 = data["beta2"]
        model.loss_type = data["loss_type"]
        model.optimizer = data["optimizer"]
        model.num_classes = data["num_classes"]
        return model

    def summary(self):
        ll = 70
        print("=" * ll)
        print("Model Summary")
        print("=" * ll)

        total_params = 0

        def count_layer(i, layer, indent=0):
            nonlocal total_params
            layer_type = type(layer).__name__
            prefix = "  " * indent
            idx = f"[{i + 1}]" if indent == 0 else "   "

            if isinstance(layer, Dense):
                params = layer.W.size + layer.b.size if layer.W is not None else 0
                total_params += params
                print(f"{idx} {prefix}Dense            {layer.in_size} → {layer.out_size:<20} params: {params}")

            elif isinstance(layer, (BatchNorm, BatchNorm2D)):
                params = layer.gamma.size + layer.beta.size if layer.gamma is not None else 0
                total_params += params
                print(f"{idx} {prefix}{layer_type:<20} momentum={layer.momentum:<10} params: {params}")

            elif isinstance(layer, Dropout):
                print(f"{idx} {prefix}Dropout          rate={layer.rate}")

            elif isinstance(layer, Conv2D):
                params = layer.W.size + layer.b.size if layer.W is not None else 0
                total_params += params
                k_h, k_w = layer.kernel_size
                s_h, s_w = layer.strides
                print(
                    f"{idx} {prefix}Conv2D           {layer.filters} filters {k_h}x{k_w} stride=({s_h},{s_w})   params: {params}")

            elif isinstance(layer, GroupConv2D):
                params = layer.W.size + layer.b.size if layer.W is not None else 0
                total_params += params
                k_h, k_w = layer.kernel_size
                print(
                    f"{idx} {prefix}GroupConv2D      {layer.filters} filters {k_h}x{k_w} groups={layer.groups}   params: {params}")

            elif isinstance(layer, DepthwiseSeparableConv2D):
                dw_params = layer.depthwise.W.size + layer.depthwise.b.size if layer.depthwise.W is not None else 0
                pw_params = layer.pointwise.W.size + layer.pointwise.b.size if layer.pointwise.W is not None else 0
                params = dw_params + pw_params
                total_params += params
                print(f"{idx} {prefix}DepthwiseSepConv2D                              params: {params}")
                print(f"     {prefix}  depthwise: {dw_params}  pointwise: {pw_params}")

            elif isinstance(layer, SpatiallySeparableConv2D):
                params = layer.W.size + layer.b.size if layer.W is not None else 0
                total_params += params
                print(f"{idx} {prefix}SpatiallySepConv2D                              params: {params}")

            elif isinstance(layer, DepthwiseConv2D):
                params = layer.W.size + layer.b.size if layer.W is not None else 0
                total_params += params
                k_h, k_w = layer.kernel_size
                print(
                    f"{idx} {prefix}DepthwiseConv2D  {k_h}x{k_w} multiplier={layer.depth_multiplier}   params: {params}")

            elif isinstance(layer, ResidualBlock):
                print(f"{idx} {prefix}ResidualBlock")
                for j, sub in enumerate(layer.layers):
                    count_layer(j, sub, indent=indent + 1)
                if layer.projection is not None:
                    print(f"     {prefix}  projection:")
                    count_layer(0, layer.projection, indent=indent + 2)

            elif isinstance(layer, MaxPooling2D):
                p_h, p_w = layer.pool_size
                s_h, s_w = layer.strides
                print(f"{idx} {prefix}MaxPooling2D     pool=({p_h},{p_w}) stride=({s_h},{s_w})")

            elif isinstance(layer, AveragePooling2D):
                p_h, p_w = layer.pool_size
                s_h, s_w = layer.strides
                print(f"{idx} {prefix}AveragePooling2D pool=({p_h},{p_w}) stride=({s_h},{s_w})")

            elif isinstance(layer, GlobalAveragePooling2D):
                print(f"{idx} {prefix}GlobalAvgPool2D")

            elif isinstance(layer, Flatten):
                print(f"{idx} {prefix}Flatten")

            elif isinstance(layer, Embedding):
                params = layer.W.size
                total_params += params
                print(
                    f"{idx} {prefix}Embedding        vocab={layer.vocab_size} dim={layer.embed_dim:<15} params: {params}")

            elif isinstance(layer, PositionEmbedding):
                params = layer.W.size
                total_params += params
                print(f"{idx} {prefix}PositionEmbedding seq={layer.seq_len} dim={layer.embed_dim:<14} params: {params}")

            elif isinstance(layer, TransformerBlock):
                # Count params inside without printing sub-layers
                attn_params = sum(w.size for w in [
                    layer.attn.W_q, layer.attn.W_k, layer.attn.W_v
                ])
                if layer.attn.W_o is not None:
                    attn_params += layer.attn.W_o.size
                ffn_params = sum(
                    w.size for w in [layer.ffn.W1, layer.ffn.b1, layer.ffn.W2, layer.ffn.b2]
                )
                norm_params = sum(
                    w.size for w in [layer.norm1.gamma, layer.norm1.beta,
                                     layer.norm2.gamma, layer.norm2.beta]
                )
                params = attn_params + ffn_params + norm_params
                total_params += params
                print(
                    f"{idx} {prefix}TransformerBlock d_model={layer.d_model} heads={layer.n_heads} ffn={layer.ffn_dim:<8} params: {params}")

            elif isinstance(layer, (RNN, GRU, LSTM)):
                params = sum(w.size for w in layer.weights())  # assumes a weights() method
                total_params += params
                print(f"{idx} {prefix}{layer_type:<20} hidden={layer.hidden_size:<15} params: {params}")

            elif isinstance(layer, Seq2Seq):
                print(f"{idx} {prefix}Seq2Seq")
                print(f"     {prefix}  encoder:")
                count_layer(0, layer.encoder, indent=indent + 2)
                print(f"     {prefix}  decoder:")
                count_layer(0, layer.decoder, indent=indent + 2)

            elif isinstance(layer, (Softmax, ReLu, Sigmoid, Tanh, Linear)):
                print(f"{idx} {prefix}{layer_type}")

            else:
                print(f"{idx} {prefix}{layer_type}")

            if indent == 0:
                print("-" * ll)

        for i, layer in enumerate(self.layers):
            count_layer(i, layer)

        print(f"Total trainable parameters: {total_params:,}")
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
    def forward(self, X, training=True):
        # Z = W * A + B
        # A - output (after activation function)
        # Also next layer input
        for layer in self.layers:
            X = layer.forward(X, training=training)
        return X


    # Backward function - locates the origin of the loss and tweaks it
    def backward(self, y_pred, y_true):
      m = y_true.shape[0] if hasattr(y_true, 'shape') else y_true.size

      dA = self._loss_derivative(y_pred, y_true) / m
      dA = self.layers[-1].backward(dA)

      # Remaining layers
      # Backpropagation: iterate layers in reverse order (from last to first)
      # Compute gradients dW and dB for each layer
      for layer in reversed(self.layers[:-1]):
          # dA = ∂J/∂A_L
          # This is the derivative of the loss w.r.t. the network output
          dA = layer.backward(dA)


    # Loss derivative - for the last layer based on the loss-type
    def _loss_derivative(self, y_pred, y_true):
        # Division by m happens in backward function

        # - y_true / y_pred
        if self.loss_type == "cross_entropy":
            # (B, T, V)
            if self.task == "language_model":
                B, T, V = y_pred.shape

                # Ensure correct shape
                assert y_true.shape == (B, T), f"y_true shape {y_true.shape} != {(B, T)}"
                assert np.issubdtype(y_true.dtype, np.integer), "y_true must be integer indices"
                assert y_true.max() < V, f"Token id {y_true.max()} out of bounds for vocab size {V}"

                # Flatten batch and time
                y_pred_flat = y_pred.reshape(B * T, V)  # (B*T, V)
                y_true_flat = y_true.reshape(-1)  # (B*T,)

                # Gradient
                d = np.zeros_like(y_pred_flat)    # (B*T, V)

                d[np.arange(B*T), y_true_flat] = -1.0 / (y_pred_flat[np.arange(B*T), y_true_flat] + self._eps)

                # Reshape back
                d = d.reshape(B, T, V)

                return d
            # (B, V)
            elif self.task == "classification":
                d = np.zeros_like(y_pred)
                m = y_true.size
                d[np.arange(m), y_true] = -1.0 / (y_pred[np.arange(m), y_true] + self._eps)
                return d
            else:
                raise Exception("Invalid loss function")

        # 2 * (y_pred - y_true)
        elif self.loss_type == "mse":
            one_hot = self._one_hot_encoding(y_true)
            return 2 * (y_pred - one_hot)
        else:
            raise Exception("Invalid loss function")


    # Compute loss for logging
    def _compute_loss(self, y_pred, y_true):
        m = y_true.size

        # -1/m * ∑ (y_true * log(y_pred))
        if self.loss_type == "cross_entropy":
          # Language Model
          if self.task == "language_model":
              B, T, V = y_pred.shape # (batch_size, seq_len, vocab_size)

              # Trim y_true to match T in case of mismatch
              y_true = y_true[:B, :T]  # <-- add this line

              data_loss = -np.mean(np.log(y_pred[np.arange(B)[:, None], np.arange(T), y_true] + self._eps))

          elif self.task == "classification":
              # data_loss = -1 * np.sum(one_hot * np.log(y_pred + self._eps)) / m
              data_loss = -np.mean(np.log(y_pred[np.arange(m), y_true] + self._eps))

          else:
              raise Exception("Invalid model task")

        elif self.loss_type == "mse":
          # 1/m * ∑ ((y_pred - y_true)^2)
          one_hot = self._one_hot_encoding(y_true)  # shape: (num_classes, N)
          data_loss = np.mean((y_pred - one_hot) ** 2)
        else:
          raise Exception("Invalid loss function")

        if self.lambda_ == 0:
            return data_loss

        else:
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
    def update(self, optimizer_t):

      for layer in self.layers:
          layer.update(self.lambda_, self.lr, self.beta1, self.beta2, self._eps, self.optimizer, optimizer_t)

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
        if output.ndim == 3:
            # language model: (batch, seq_len, vocab_size) -> (batch, seq_len)
            return np.argmax(output, axis=-1)
        if output.shape[1] == 1: # Binary classification
            return (output > 0.5).astype(int).flatten()
        # Multi-class (softmax)
        return np.argmax(output, axis=1)

    @staticmethod
    def shuffle_data(x, y):
        # Handle (src, trg) format for X
        if isinstance(x, tuple):
            perm = np.random.permutation(x[0].shape[0])
            x = tuple(arr[perm] for arr in x)
        else:
            perm = np.random.permutation(x.shape[0])
            x = x[perm]

        y = y[perm]
        return x, y

    @staticmethod
    def set_seed(seed):
      np.random.seed(seed)

    # need reconstruction to work with cnn and lm
    """def check_gradient(self, X, y):
      assert X.shape[0] == y.size, f"X has {X.shape[0]} samples but y has {y.size}"

      # Use a small batch to avoid numerical issues
      X = X[:8].astype(np.float64)  # <-- float64 is critical for numerical grad
      y = y[:8]

      rel_diff = []
      original_lambda = self.lambda_
      self.lambda_ = 0.0
      epsilon = 1e-5  # smaller epsilon can help

      y_pred = self.forward(X, training=False)
      self.backward(y_pred, y)

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
              y_pred = self.forward(X, training=False)
              loss_plus = self._compute_loss(y_pred, y)

              layer.W[i, j] = W_orig - epsilon
              y_pred = self.forward(X, training=False)
              loss_minus = self._compute_loss(y_pred, y)

              grad_numerical = (loss_plus - loss_minus) / (2 * epsilon)
              layer.W[i, j] = W_orig

              numerator = abs(grad_numerical - grad_analytical)
              denominator = abs(grad_numerical) + abs(grad_analytical) + 1e-10
              rel_diff.append(numerator / denominator)

              print(f"Layer {idx} W[{i},{j}]  Numerical: {grad_numerical:.10f}  Analytical: {grad_analytical:.10f}  Rel diff: {rel_diff[-1]:.2e}")

      self.lambda_ = original_lambda
      return rel_diff"""

    # Returns the feature maps of a specific convolutional layer
    def visualize_feature_maps(self, X, layer_index):
        A = X
        for idx, layer in enumerate(self.layers):
            A = layer.forward(A, training=False)
            if idx == layer_index:
                # feature maps shape: (m, H, W, C_out)
                return A
        raise ValueError(f"Layer index {layer_index} out of range")

    # Compile the model and trains it
    # x - input features
    # y - labels
    # lr - learning rate
    # epochs - num of iterations over the data
    # batch_size - data size before model updating
    # loss_type - loss functions used
    # lambda_ - lambda used for preventing weights overfitting
    def gradient_descent(self, X, y, X_val=None, y_val=None, epochs=10, batch_size=1):

      # History - keep track of matrics and loss
      history = History()

      # Optimizer step count
      optimizer_t = 0

      # Number of epochs without val improvement until early stopping
      stopping_patience = 5

      # Main model training loop
      for ep in range(epochs):
        predictions = []
        epoch_loss = 0

        x_shuffled, y_shuffled = NeuralNetwork.shuffle_data(X, y)


        # Handle (src, trg) format - LM
        if isinstance(x_shuffled, tuple):
            n_samples = X[0].shape[0]
        else:
            n_samples = X.shape[0]

        # Batches
        # tqdm - loading animation
        for i in tqdm(range(0, n_samples, batch_size)):

          optimizer_t += 1

          # get batch
          # Handle (src, trg) format - LM
          if isinstance(x_shuffled, tuple):
              x_batch = tuple(arr[i:i + batch_size] for arr in x_shuffled)
          else:
              x_batch = x_shuffled[i:i + batch_size]

          y_batch = y_shuffled[i:i + batch_size]

          # feed model
          y_pred = self.forward(x_batch)
          self.backward(y_pred, y_batch)
          self.update(optimizer_t)

          # Monitor loss - epoch_loss = Avg(batches_loss)
          predictions.append(self._decode_output(y_pred))
          batch_loss = self._compute_loss(y_pred, y_batch)
          epoch_loss += batch_loss

          # Check Gradient - make sure backpropagation works well
          #self.check_gradient(x_batch, y_batch)

        # Average loss over batches
        epoch_loss /= (n_samples // batch_size)

        predictions = np.concatenate(predictions)

        # Update history - epoch num and loss
        history.add("epoch", ep + 1)
        history.add("loss", epoch_loss)

        # Calculate matrics and update history
        if self.task == "language_model":
            # token-level accuracy: predicted next token == actual next token
            token_acc = np.mean(predictions == y_shuffled)
            history.add("token_accuracy", token_acc)
        else:
            history = self.calc_metrics(
                history, predictions,
                y_shuffled[:len(predictions)],
                metrics=["accuracy", "precision", "recall"]
            )


        # Validation
        if X_val is not None and y_val is not None:
          # Divide into batches to save computing power
          val_batch_size = 256
          val_preds_list = []

          # Support tuple X_val for Seq2Seq (src, trg) format
          if isinstance(X_val, tuple):
            n_val = X_val[0].shape[0]
          else:
            n_val = len(X_val)

          for vi in range(0, n_val, val_batch_size):
              # Support tuple X_val for Seq2Seq (src, trg) format
              if isinstance(X_val, tuple):
                vb = tuple(arr[vi:vi + val_batch_size] for arr in X_val)
              else:
                vb = X_val[vi:vi + val_batch_size]

              val_preds_list.append(self.forward(vb, training=False))

          val_pred = np.concatenate(val_preds_list, axis=0)

          # Calculate matrics and update history
          if self.task == "language_model":
              # Adjust val pred and y for correct shape (B, T, V) -> (B*T, V)
              # flatten
              B, T, V = val_pred.shape
              vp_flat = val_pred.reshape(B * T, V)
              vy_flat = y_val[:len(val_preds_list) * val_batch_size].reshape(-1)[:B * T]
              val_loss = self._compute_loss(val_pred, y_val)
              val_acc = np.mean(np.argmax(vp_flat, axis=-1) == vy_flat)
          else:
              val_loss = self._compute_loss(val_pred, y_val)
              val_acc = NeuralNetwork.accuracy(self._decode_output(val_pred), y_val)

          history.add("val_loss", val_loss)
          history.add("val_accuracy", val_acc)

          history.progress()

          # Early stopping check
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
    def compile(self, loss_type="cross_entropy", optimizer="adam", lr=0.001, lambda_=0.0, beta1=0.9, beta2=0.999, task="classification"):
        self.lr = lr
        self.lambda_ = lambda_
        self.beta1 = beta1
        self.beta2 = beta2
        self.loss_type = loss_type
        self.optimizer = optimizer
        self.task = task


    # Train the model
    def fit(self, X, y, X_val=None, y_val=None, epochs=10, batch_size=1):


        # Find last layer - last outsize is num classes
        if self.task == "classification":
            for layer in reversed(self.layers):
                if hasattr(layer, "out_size"):
                    self.num_classes = layer.out_size
                    break
        elif self.task == "language_model":
            for layer in reversed(self.layers):
                if hasattr(layer, "vocab_size"):
                    self.num_classes = layer.vocab_size
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
        return self.forward(X, training=False)


    # Evaluate model performance (accuracy) on given dataset
    def evaluate(self, X, y):
        predictions = self.predict(X)
        return NeuralNetwork.accuracy(predictions, y)

    """*****************************
        wrong implementation
    *****************************"""
    # Generate response for text (language model only)
    def generate(self, prompt_ids, tokenizer, max_new_tokens=50, temperature=1.0, seq_len=16):
        """
        Autoregressively generate token ids from a prompt.

        prompt_ids  : list of integer token ids (from tokenizer.encode)
        temperature : > 1.0 = more random, < 1.0 = more focused, 1.0 = neutral
        seq_len     : context window — must match what the model was trained on
        """
        ids = list(prompt_ids)

        for _ in range(max_new_tokens):
            context = ids[-seq_len:]

            # Pad to seq_len if needed
            if len(context) < seq_len:
                pad_id = tokenizer.special_tokens.get("<PAD>", 0)
                context = [pad_id] * (seq_len - len(context)) + context

            x = np.array(context)[np.newaxis, :]  # (1, seq_len)

            logits = self.forward(x, training=False)  # (1, seq_len, vocab_size)
            last = logits[0, -1].astype(np.float64)  # (vocab_size,) — last position

            # Temperature scaling then softmax
            last /= temperature  # scale first
            last -= last.max()  # then stabilize
            probs = np.exp(last) / np.sum(np.exp(last))

            next_id = self._top_k_sample(probs, k=10)
            ids.append(int(next_id))

        return tokenizer.decode(ids)

    def _top_k_sample(self, probs, k=10):
        # Zero out everything except top k probabilities
        top_k_indices = np.argsort(probs)[-k:]
        filtered = np.zeros_like(probs)
        filtered[top_k_indices] = probs[top_k_indices]
        filtered /= filtered.sum()  # renormalize
        return np.random.choice(len(filtered), p=filtered)
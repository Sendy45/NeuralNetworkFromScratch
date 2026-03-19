import numpy as np

class History:
  def __init__(self):
    self.history = {}

  def add(self, key, value):
    if key not in self.history:
      self.history[key] = []
    self.history[key].append(value)

  def progress(self):
    for key, value in self.history.items():
      print(f"{key}: {value[-1]}")



class Scaler:
    def __init__(self, mode="standard"):
        """
        mode: "standard" — Z-score normalisation (mean=0, std=1 per feature)
              "minmax"   — scales each feature to [0, 1]

        Works with any row-major input shape:
          (N, features)        — dense / flattened data
          (N, H, W, channels)  — image data (statistics per pixel-channel)
        """
        self.mode = mode
        self.mean = None
        self.std = None
        self.min = None
        self.max = None

    def fit(self, X):
        """Compute statistics from X. Call only on training data."""
        if self.mode == "standard":
            # mean/std shape: (1, features) or (1, H, W, C)
            # keepdims=True lets them broadcast against any (N, ...) input
            self.mean = np.mean(X, axis=0, keepdims=True)
            self.std = np.std(X, axis=0, keepdims=True)
            self.std[self.std == 0] = 1e-8  # Avoid division by zero

        elif self.mode == "minmax":
            self.min = np.min(X, axis=0, keepdims=True)
            self.max = np.max(X, axis=0, keepdims=True)
            # Avoid division by zero if all values in a feature are the same
            self.diff = self.max - self.min
            self.diff[self.diff == 0] = 1e-8

    def transform(self, X):
        """Applies scaling to data using fitted parameters."""
        if self.mode == "standard":
            return (X - self.mean) / self.std
        elif self.mode == "minmax":
            return (X - self.min) / self.diff
        raise NotImplementedError(f"Unknown mode: {self.mode}")

    def fit_transform(self, X):
        """Fit on X, then transform and return it. Use only on training data."""
        self.fit(X)
        return self.transform(X)

def split_train_test(X, y, test_ratio=0.2):
  """
    Randomly split (X, y) into train and test sets.
    X shape: (N, ...)  -  any row-major layout.
    Returns: X_train, y_train, X_test, y_test
  """
  m = X.shape[0]
  perm = np.random.permutation(m)

  X = X[perm]
  y = y[perm]

  test_size = int(m * test_ratio)

  X_test = X[:test_size]
  y_test = y[:test_size]

  X_train = X[test_size:]
  y_train = y[test_size:]

  return X_train, y_train, X_test, y_test


def split_train_validation(X, y, val_ratio=0.2):
    """
        Randomly split (X, y) into train and validation sets.
        X shape: (N, ...)  -  any row-major layout.
        Returns: X_train, y_train, X_val, y_val
    """
    m = X.shape[0]
    perm = np.random.permutation(m)
    X = X[perm]
    y = y[perm]

    val_size = int(m * val_ratio)

    X_val = X[:val_size]
    y_val = y[:val_size]

    X_train = X[val_size:]
    y_train = y[val_size:]

    return X_train, y_train, X_val, y_val
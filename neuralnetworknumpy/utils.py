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
        mode: "standard" for Z-score (Mean=0, Std=1)
              "minmax" for range scaling (0 to 1)
        """
        self.mode = mode
        self.mean = None
        self.std = None
        self.min = None
        self.max = None

    def fit(self, X):
        """Calculates parameters from training data. X shape: (features, samples)"""
        if self.mode == "standard":
            self.mean = np.mean(X, axis=1, keepdims=True)
            self.std = np.std(X, axis=1, keepdims=True)
            self.std[self.std == 0] = 1e-8  # Avoid division by zero

        elif self.mode == "minmax":
            self.min = np.min(X, axis=1, keepdims=True)
            self.max = np.max(X, axis=1, keepdims=True)
            # Avoid division by zero if all values in a feature are the same
            self.diff = self.max - self.min
            self.diff[self.diff == 0] = 1e-8

    def transform(self, X):
        """Applies scaling to data using fitted parameters."""
        if self.mode == "standard":
            return (X - self.mean) / self.std
        elif self.mode == "minmax":
            return (X - self.min) / self.diff
        else:
            raise NotImplementedError

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

def split_train_test(X, y, test_ratio=0.2):
  m = X.shape[1]
  perm = np.random.permutation(m)

  X = X[:, perm]
  y = y[perm]

  test_size = int(m * test_ratio)

  X_test = X[:, :test_size]
  y_test = y[:test_size]

  X_train = X[:, test_size:]
  y_train = y[test_size:]

  return X_train, y_train, X_test, y_test


def split_train_validation(X, y, val_ratio=0.2):
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
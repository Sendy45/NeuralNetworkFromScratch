from neuralnetworknumpy.backend import np

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
from neuralnetworknumpy.backend import np

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
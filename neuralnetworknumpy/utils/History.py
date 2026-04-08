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
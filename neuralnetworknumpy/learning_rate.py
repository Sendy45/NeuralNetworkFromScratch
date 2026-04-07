from .backend import np

class LearningRate:
    def __call__(self, step):
        raise NotImplementedError()


class SequentialLR(LearningRate):
    def __init__(self, schedules, boundaries):
        self.schedules = schedules
        self.boundaries = boundaries

    def __call__(self, step):
        for i, boundary in enumerate(self.boundaries):
            if step < boundary:
                start = self.boundaries[i - 1] if i > 0 else 0
                return self.schedules[i](step - start)  # pass local step
        start = self.boundaries[-1]
        return self.schedules[-1](step - start)  # pass local step to last schedule


class LinearWarmup(LearningRate):
    def __init__(self, warmup_steps, max_lr):
        self.warmup_steps = warmup_steps
        self.max_lr = max_lr

    def __call__(self, step):
        if step >= self.warmup_steps:
            return self.max_lr
        return self.max_lr * (step / self.warmup_steps)


class StepDecay(LearningRate):
    def __init__(self, drop_rate, step_size, base_lr):
        self.drop_rate = drop_rate
        self.step_size = step_size
        self.base_lr = base_lr

    def __call__(self, step):
        return self.base_lr * (self.drop_rate ** (np.floor(step / self.step_size)))

class ExponentialDecay(LearningRate):
    def __init__(self, drop_rate, base_lr):
        self.drop_rate = drop_rate
        self.base_lr = base_lr

    def __call__(self, step):
        return self.base_lr * np.e ** (-1 * self.drop_rate * step)

class CosineDecay(LearningRate):
    def __init__(self, max_steps, base_lr):
        self.max_steps = max_steps
        self.base_lr = base_lr

    def __call__(self, step):
        step = min(step, self.max_steps)
        return self.base_lr * 0.5 * (1 + np.cos(np.pi * step / self.max_steps))
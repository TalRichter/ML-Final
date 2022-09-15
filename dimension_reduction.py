import time


class DimReduce:

    def __init__(self) -> None:
        self.total_runtime = -1

    def __call__(self, x, y):
        start_time = time.time()
        scores = self.run(x, y)
        self.total_runtime = time.time() - start_time
        return scores

    def run(self, x, y):
        pass

    def set_n_features(self, k):
        pass

    def __repr__(self) -> str:
        return type(self).__name__

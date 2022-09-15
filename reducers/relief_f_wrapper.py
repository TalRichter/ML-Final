from skfeature.function.similarity_based.reliefF import reliefF

from dimension_reduction import DimReduce


class ReliefF(DimReduce):
    def __init__(self, alpha=0.1) -> None:
        super().__init__()
        self.alpha = alpha

    def run(self, x, y) -> tuple:
        return reliefF(x,y, mode="raw")


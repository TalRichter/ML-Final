import sys
from io import StringIO

import numpy as np

from dimension_reduction import DimReduce
import matlab
import matlab.engine


class DUFS(DimReduce):
    def __init__(self) -> None:
        super().__init__()

    def run(self,x,y):
        eng = matlab.engine.start_matlab()
        out = StringIO()
        err = StringIO()
        data = matlab.double(x.tolist())

        # try:
        index = eng.dufs(data, len(x), 1.0, 1.0, 1.0, nargout=1)
        # except:
        #     pass
        # print('matlab out')
        # print(out.getvalue())
        # print(err.getvalue())
        index = np.asarray(index)
        index = [int(item) for sublist in index for item in sublist]
        return np.array(index)
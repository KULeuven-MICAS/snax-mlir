import numpy as np

from util.gendata import create_data, create_header

np.random.seed(0)
data = np.random.randint(-128, 128, size=(8, 640), dtype=np.int8)

create_header("data", {"DATA_SIZE": data.size}, {"data": data})
create_data("data", {"data": data})

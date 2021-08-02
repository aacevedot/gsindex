# Geometrical Separability Index (GSI)

This is a python wrapper of the Geometrical Separability Index (GSI) based on the MATLAB code proposed in [1] and originally proposed by Chris Thornton in [2].

> [1] Greene, J. (2001). Feature subset selection using thornton’s separability index and its applicability to a number of sparse proximity-based classifiers. In Proceedings of Annual Symposium of the Pattern Recognition Association of South Africa.
> 
> [2] Thornton, C. (1998). Separability is a learner’s best friend. In 4th Neural Computation and Psychology Workshop, London, 9–11 April 1997 (pp. 40-46). Springer, London.

## Installation

Run the following to install:

```python
pip install gsindex
```

## Usage

```python
import numpy as np
from gsindex import geometrical_separability_index

matrix = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [10, 11], [12, 13], [14, 15], [16, 17]])
labels = np.array(['sample1', 'sample1', 'sample1', 'sample1', 'sample2', 'sample2', 'sample2', 'sample2'])

gsi = geometrical_separability_index(matrix, labels)

print(gsi)
```

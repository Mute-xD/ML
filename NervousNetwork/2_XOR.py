"""
异或 是非线性的，无法用单层感知器解决

调不了包就设为根目录+改名
"""
import singleLayerPerceptron
import numpy as np


x = np.array([[1, 0, 0],
              [1, 0, 1],
              [1, 1, 0],
              [1, 1, 1]])
y = np.array([[1], [0], [0], [1]])

slp = singleLayerPerceptron.SingleLayerPerceptron(x, y)
slp.plot()

import numpy as np
from scipy.special import softmax
import math
from random import choice

# x = np.array([choice([0, 1.,2.]) for _ in range(12)])
x = np.array([1., 0, 1, 0, 0, 2, 0, 2, 1, 1, 1, 1])
x = x.reshape(3,4)

query_w = np.array([1., 0, 1., 1, 0, 0, 0, 0, 1, 0, 1, 1])
query_w = query_w.reshape(4, 3)


key_w = np.array([0, 0, 1., 1, 1, 0, 0, 1, 0, 1, 1, 0])
key_w = key_w.reshape(4, 3)

value_w = np.array([0, 2, 0, 0, 3, 0, 1, 0, 3, 1, 1, 0])
value_w = value_w.reshape(4, 3)

Q = x @ query_w
K = x @ key_w
V = x @ value_w

k_d = int(math.sqrt(3))
print('k_d:', k_d)

attention_score = (Q @ K.transpose())/k_d
print(attention_score)

attention_score = softmax(attention_score, axis=1) # taking softmax across the rows
print("attention_score: ")
print(attention_score)

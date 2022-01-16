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


print('Contenating')

import numpy as np

print("Assuming the training on the 8 heads of the attention sub-layer is done.")
z0h1=np.random.random((3, 64))
z1h2=np.random.random((3, 64))
z2h3=np.random.random((3, 64))
z3h4=np.random.random((3, 64))
z4h5=np.random.random((3, 64))
z5h6=np.random.random((3, 64))
z6h7=np.random.random((3, 64))
z7h8=np.random.random((3, 64))
print("shape of one head", z0h1.shape, "dimension of 8 heads", 64*8)

print("Concatenation of heads 1 to 8 to obtain the original 8x64=512 output dimension of the model")
output_attention=np.hstack((z0h1,z1h2,z2h3,z3h4,z4h5,z5h6,z6h7,z7h8))
print('final output shape', output_attention.shape)

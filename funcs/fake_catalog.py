import numpy as np
import itertools

length = np.linspace(0,500,21)

full_array = np.zeros((21**3,4))
i = 0

for p in itertools.product(length,repeat=3):
    full_array[i,1:4] = p
    i+=1

for i in range(len(full_array)):
    full_array[i,0] = np.random.rand() * 12


print(len(full_array))

np.savetxt('fake_catalog', full_array/0.6751)






"""
Plot the loglikelihood function used by the insect to make sure it's all good.
"""

import matplotlib.pyplot as plt
plt.ion()

from insect import Insect

from config.ins_loglike import *

# create insect
ins = Insect(env=ENV, dt=DT)
ins.w = W
ins.r = R
ins.d = D
ins.a = A
ins.tau = TAU
ins.loglike_function = LOGLIKE

# initialize insect
ins.initialize()

plt.matshow(ins.loglike_map[0][:, :, 0].T, origin='lower')
plt.draw()

print 'Env shape:'
print ins.env.shape
print 'Loglikelihood map shape:'
print ins.loglike_map.shape
raw_input()
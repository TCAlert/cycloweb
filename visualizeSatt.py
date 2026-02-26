import numpy as np
import matplotlib.pyplot as plt
from satcmaps import wvtables, irtables 

X, Y = np.meshgrid(np.arange(0, 100), np.arange(0, 100))

cmaps: list = wvtables.values()  # TODO list of colormaps. Please set!
names: list = list(wvtables.keys())
fig, axs = plt.subplots(len(cmaps), 1, figsize=(10, len(cmaps)), dpi=500, constrained_layout=True)
for i, cmap in enumerate(cmaps):
    axs[i].pcolormesh(X, Y, X, cmap=cmap[0])
    axs[i].set_xticks([])
    axs[i].set_yticks([])
    axs[i].set_ylabel(names[i])

plt.savefig(r"C:\Users\deela\Downloads\wvcolormaps.png", dpi=500)

cmaps: list = irtables.values()  # TODO list of colormaps. Please set!
names: list = list(irtables.keys())
fig, axs = plt.subplots(len(cmaps), 1, figsize=(10, len(cmaps)), dpi=500, constrained_layout=True)
for i, cmap in enumerate(cmaps):
    axs[i].pcolormesh(X, Y, X, cmap=cmap[0])
    axs[i].set_xticks([])
    axs[i].set_yticks([])
    axs[i].set_ylabel(names[i])

plt.savefig(r"C:\Users\deela\Downloads\ircolormaps.png", dpi=500)
plt.show()
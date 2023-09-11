#!/usr/bin/env python3

# MIT license
#
# Copyright © 2022-2023 Timo Koch
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the “Software”), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

import numpy as np
import scipy.special as sp
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.animation as animation
import sys

WOMERSLEY_NUMBER = 1.0
if len(sys.argv) == 2:
    WOMERSLEY_NUMBER = float(sys.argv[1])


def velocity(r, t, wo):
    """The analytical solution"""
    r = np.abs(r)  # symmetric profile
    arg = wo * 1j ** (3.0 / 2.0)
    j0 = lambda s: sp.jv(0, s)
    j0Arg = j0(arg)
    complex_value = -(1.0 - j0(r * arg) / j0Arg) * np.exp(t * 1j) / 1j
    return np.real(complex_value)


def velocity_plot(r, t, wo, scaling=1.0):
    """Scaled and translated analytical solution for plotting"""
    return scaling * velocity(r, t, wo) / np.pi * 180


fig, ax = plt.subplots(2, 1, gridspec_kw={"height_ratios": [1, 3]}, figsize=(6,4))
xdata, ydata = np.linspace(-1.0, 1.0, 100, endpoint=True), []
dpt = np.linspace(0, 2 * np.pi, 360, endpoint=True)
time = np.linspace(0, 2 * np.pi, 13, endpoint=True)
timelabels = [f"{int(np.round(t))}°" for t in time / np.pi * 180]

(l3n,) = ax[0].plot([], [], "k")
(l2n,) = ax[0].plot([], [], "o")

ax[0].set_xlim(0, 360)
ax[0].set_ylim(-1, 1)
ax[0].set_xticks(time / np.pi * 180)
ax[0].set_xticklabels(timelabels)

(ln,) = ax[1].plot([], [], "k")

ax[1].set_xlim(-1, 1)
ax[1].set_ylim(-100, 100)
ax[1].set_xlabel(r"$r^* = r/R$")
ax[1].set_xticks(np.linspace(-1, 1, 5))
ax[1].set_xticklabels(np.abs(np.linspace(-1, 1, 5)))
ax[1].get_yaxis().set_visible(False)

def init():
    return ln, l2n, l3n

# to make it fit into the same window
SCALING = 3.0/WOMERSLEY_NUMBER**0.5
def update(t):
    ln.set_data(xdata, velocity_plot(r=xdata, t=t, wo=WOMERSLEY_NUMBER, scaling=SCALING))
    l2n.set_data([t / np.pi * 180,], np.cos([t,]))
    l3n.set_data(dpt / np.pi * 180, np.cos(dpt))
    return ln, l2n, l3n


ani = FuncAnimation(
    fig,
    update,
    frames=np.linspace(0, 2 * np.pi, 128, endpoint=False),
    init_func=init,
    blit=True,
    repeat=True,
    interval=10,
)

# writer = animation.PillowWriter(fps=30)
# ani.save(f"womersley-{WOMERSLEY_NUMBER}.gif", writer=writer)

plt.show()

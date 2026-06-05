import random
import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def drift():
    dt = 0.1
    drift_val = 1
    sigma = random.gauss(0, 1)  
    val = (drift_val * dt) + (sigma * math.sqrt(dt)) 
    return val

def weiner():
    particlepos = {0: 0}
    for i in range(50):
        newpos = particlepos[i] + drift()
        particlepos[i+1] = newpos
    return list(particlepos.keys()), list(particlepos.values())

x, y = weiner()

fig, ax = plt.subplots()
line, = ax.plot([], [], lw=2, color='blue')

ax.set_xlim(min(x), max(x))
ax.set_ylim(min(y) - 2, max(y) + 2)
ax.axhline(y=2.5, color='r', linestyle='--', linewidth=2,label='Boundary')
ax.set_xlabel('Time')
ax.legend()
cond_text = ax.text(.5, 1.05, "", fontsize = 12, color = 'Red', transform=ax.transAxes, 
        verticalalignment='center')

def update(i):
    current_x = x[:i+1]
    current_y = y[:i+1]
    if current_y and current_y[-1] >= 2.5:
        cond_text.set_text("Decision Made")
    line.set_data(current_x, current_y)
    return line,

ani = FuncAnimation(
    fig, update, frames=len(x), interval=150, blit=True, repeat=False
)

ani.save("wave_animation.gif", writer='pillow')

plt.show()

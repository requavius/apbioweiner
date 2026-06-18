import math
import random
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
 

V_REST = -70.0     
V_THRESH = -55.0    # mV  firing threshold
V_RESET = -70.0     # mV  post-spike reset
V_SPIKE = 40.0      # mV  drawn action-potential peak
TAU = 12.0          # ms  membrane time constant (the RC leak)
DT = 0.5            # ms  integration step
N_STEPS = 200       # 100 ms window
NOISE_SD = 2.3      # mV * ms^-0.5  synaptic input noise
 
DRIVE_LOW = 12.0   
DRIVE_HIGH = 20.0   
 
 
def simulate(drive):
    t = [0.0]
    v = [V_REST]
    spike_at = None
    for i in range(N_STEPS):
        if spike_at is not None:
            v.append(V_RESET)           
            t.append(t[-1] + DT)
            continue
        dv = ((-(v[-1] - V_REST) + drive) / TAU) * DT \
            + NOISE_SD * math.sqrt(DT) * random.gauss(0, 1)
        new_v = v[-1] + dv
        if new_v >= V_THRESH:
            spike_at = i + 1
            new_v = V_SPIKE             
        v.append(new_v)
        t.append(t[-1] + DT)
    return t, v, spike_at
 
 
random.seed(7)
for _ in range(5000):
    tL, vL, sL = simulate(DRIVE_LOW)
    tH, vH, sH = simulate(DRIVE_HIGH)
    if sL is None and sH is not None and sH < int(N_STEPS * 0.75):
        break
 
fig, (axL, axH) = plt.subplots(1, 2, figsize=(11, 4.6), sharey=True)
fig.suptitle("Dopamine biases how fast a neuron's voltage drifts to threshold",
             fontsize=13, fontweight="bold")
 
for ax, title in ((axL, "Low dopamine"), (axH, "High dopamine")):
    ax.set_xlim(0, N_STEPS * DT)
    ax.set_ylim(-82, 50)
    ax.axhline(V_THRESH, color="#c0392b", ls="--", lw=1.5)
    ax.axhline(V_REST, color="#7f8c8d", ls=":", lw=1.2)
    ax.set_xlabel("Time (ms)")
    ax.set_title(title, fontsize=11)
    ax.text(2, V_THRESH + 2, "Threshold (-55 mV)", color="#c0392b", fontsize=9)
    ax.text(2, V_REST - 6, "Rest (-70 mV)", color="#7f8c8d", fontsize=9)
 
axL.set_ylabel("Membrane potential (mV)")
 
lineL, = axL.plot([], [], lw=2, color="#2c3e50")
lineH, = axH.plot([], [], lw=2, color="#2c3e50")
txtL = axL.text(0.5, 0.92, "", transform=axL.transAxes, ha="center",
                fontsize=12, color="#27ae60", fontweight="bold")
txtH = axH.text(0.5, 0.92, "", transform=axH.transAxes, ha="center",
                fontsize=12, color="#27ae60", fontweight="bold")
 
 
def update(i):
    lineL.set_data(tL[:i + 1], vL[:i + 1])
    lineH.set_data(tH[:i + 1], vH[:i + 1])
    if sL is not None and i >= sL:
        txtL.set_text("Action potential fired")
    if sH is not None and i >= sH:
        txtH.set_text("Action potential fired")
    elif sH is None or i < sH:
        txtH.set_text("")
    return lineL, lineH, txtL, txtH
 
 
ani = FuncAnimation(fig, update, frames=len(tL), interval=60,
                    blit=False, repeat=False)
ani.save("neuron_decision.gif", writer="pillow", fps=20)
plt.close(fig)
print("saved neuron_decision.gif | low spike:", sL, "| high spike:", sH)

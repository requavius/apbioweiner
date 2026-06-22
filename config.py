from dataclasses import dataclass

@dataclass
class Params:
    V_REST: float = -70.0
    V_THRESH: float = -55.0    # mV  firing threshold
    V_RESET: float = -70.0     # mV  post-spike reset
    V_SPIKE: float = 40.0      # mV  drawn action-potential peak
    TAU: float = 12.0          # ms  membrane time constant (the RC leak)
    DT: float = 0.5            # ms  integration step
    N_STEPS: float = 200       # 100 ms window
    NOISE_SD: float = 2.3      # mV * ms^-0.5  synaptic input noise
    drive: float = 15          #

class Option:
    def __init__(self, name,  present, value = 0, effort = 0.0, driver_slider = None, artist = None):
        self.name = name
        self.value = value         # learned worth of checking it; static at first, learned later
        self.effort = effort     # cost to obtain: on_desk 0.0, in_bag 0.4, other_room 1.0
        self.present = present       # bool; everything that loops the list respects this
        self.driver_slider = driver_slider  # the matplotlib Slider that sets its dopamine-driven input
        self.artist = artist     # the drawn marker in the scene, so you can hide it on removal
import random
import math
from collections import deque

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import CheckButtons, RadioButtons, Button, Slider
from matplotlib.patches import Rectangle, Circle

from config import Params, Option

EFFORT_COST = 11.5    # mV of drive removed at effort 1.0 (object far away)
WINDOW = 200          # timesteps shown in the scrolling traces
GRID_THRESHOLD = 6    # more neurons than this -> light-up grid instead of traces
FRAME_MS = 30         # wall-clock ms between simulation steps
N_NEURONS = 18       # neurons per present object
RACE_THRESHOLD = 50   # accumulated spikes needed to win attention
NOTIF_BOOST = 16.0    # extra drive a phone notification injects (decays away)
LEARN_RATE = 0.25     # alpha: how fast learned value tracks the reward
REWARD_MEAN = 8.0     # default payoff (live-tweakable via the Reward slider)
REWARD_MAX = 25.0     # top of the Reward slider's range
REWARD_SD = 5.0       # noise around the set reward, so the error never settles

# Discrete effort settings offered by each object's dropdown.
EFFORT_CHOICES = [
    ("On desk", 0.0),
    ("In bag", 0.4),
    ("Across room", 1.0),
    ("Far away", 1.4),
]


class population:

    _instances = 0                     # used to give each pool a distinct seed

    def __init__(self, option, n=3, params=None, seed=7):
        self.option = option
        self.n = n
        self.params = params or Params()
        self.rng = random.Random(seed + population._instances)
        population._instances += 1
        self.v = [self.params.V_REST for _ in range(n)]
        self.spiked = [False] * n
        self.history = [deque([self.params.V_REST] * WINDOW, maxlen=WINDOW)
                        for _ in range(n)]
        self.brightness = np.zeros(n)
        self.boost = 0.0               # transient drive from a notification
        self.phasic = 0.0              # last outcome's prediction error, decays
        # grid geometry, used when n > GRID_THRESHOLD
        self.cols = math.ceil(math.sqrt(n))
        self.rows = math.ceil(n / self.cols)

    def effective_drive(self):
        return self.option.value - self.option.effort * EFFORT_COST + self.boost

    def step(self):

        params = self.params
        drive = self.effective_drive()
        self.boost *= 0.92                 # the notification jolt fades away
        self.phasic *= 0.85                # the phasic RPE burst fades away
        self.brightness *= 0.75            # fade previous flashes
        fired = 0
        for k in range(self.n):
            if self.spiked[k]:
                self.v[k] = params.V_RESET   # step after a spike: back to rest
                self.spiked[k] = False
            else:
                dv = ((-(self.v[k] - params.V_REST) + drive) / params.TAU) * params.DT \
                    + params.NOISE_SD * math.sqrt(params.DT) * self.rng.gauss(0, 1)
                nv = self.v[k] + dv
                if nv >= params.V_THRESH:
                    nv = params.V_SPIKE
                    self.spiked[k] = True
                    self.brightness[k] = 1.0
                    fired += 1
                self.v[k] = nv
            self.history[k].append(self.v[k])
        return fired


class VizPanel:

    def __init__(self, ax, pop):
        self.ax = ax
        self.pop = pop
        self.mode = "grid" if pop.n > GRID_THRESHOLD else "trace"
        ax.clear()
        ax.set_facecolor("#111111")
        if self.mode == "grid":
            self.mask = np.zeros(pop.rows * pop.cols, dtype=bool)
            self.mask[pop.n:] = True        # cells past n aren't neurons
            self.img = ax.imshow(self._grid(), cmap="inferno", vmin=0, vmax=1)
            self.img.cmap.set_bad("#202020")
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(f"{pop.n} neurons — click one for cascade + DDM", fontsize=9)
        else:
            ax.set_xlim(0, WINDOW)
            ax.set_ylim(-82, 50)
            ax.axhline(pop.params.V_THRESH, color="#c0392b", ls="--", lw=1)
            ax.axhline(pop.params.V_REST, color="#7f8c8d", ls=":", lw=1)
            ax.set_xticks([])
            colors = plt.cm.viridis(np.linspace(0.1, 0.85, pop.n))
            self.lines = [ax.plot(range(WINDOW), list(pop.history[k]),
                                  lw=1.5, color=colors[k])[0]
                          for k in range(pop.n)]
            ax.set_title(f"{pop.n} neurons — click one for cascade + DDM",
                         fontsize=9)

    def _grid(self):
        padded = np.zeros(self.pop.rows * self.pop.cols)
        padded[:self.pop.n] = self.pop.brightness
        return np.ma.masked_array(padded.reshape(self.pop.rows, self.pop.cols),
                                  mask=self.mask.reshape(self.pop.rows, self.pop.cols))

    def update(self):
        if self.mode == "grid":
            self.img.set_array(self._grid())
        else:
            for k, ln in enumerate(self.lines):
                ln.set_ydata(self.pop.history[k])


class CascadePanel:
    
    VAL_MAX = 25.0        # value scale that fills the tonic gauges
    PHASIC_MAX = 12.0     # |delta| that saturates the phasic flash

    def __init__(self, ax):
        self.ax = ax
        self.obj = None
        self.gauges = {}      # label -> (fill Rectangle, x0, full width)
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 12)
        ax.axis("off")
        ax.set_facecolor("#0e0e12")
        self._build()
        ax.set_visible(False)   # hidden until a neuron is clicked

    def _node(self, x, y, text, fc, fontsize=8.5):
        return self.ax.text(x, y, text, ha="center", va="center",
                            fontsize=fontsize, color="white", zorder=5,
                            bbox=dict(boxstyle="round,pad=0.3", fc=fc,
                                      ec="white", lw=1))

    def _arrow(self, x0, y0, x1, y1, color="#9aa0a6"):
        self.ax.annotate("", xy=(x1, y1), xytext=(x0, y0), zorder=3,
                         arrowprops=dict(arrowstyle="-|>", color=color, lw=1.6))

    def _gauge(self, y, label):
        """One labelled horizontal bar with a live %% readout on the right."""
        x0, w, h = 3.4, 4.3, 0.55
        self.ax.add_patch(Rectangle((x0, y), w, h, fill=True, fc="#2a2a33",
                                    ec="#888", lw=1.0, zorder=3))
        fill = Rectangle((x0, y), 0.0, h, fc="#3498db", ec="none", zorder=4)
        self.ax.add_patch(fill)
        self.ax.text(x0 - 0.25, y + h / 2, label, ha="right", va="center",
                     fontsize=8.5, color="#ecf0f1")
        pct = self.ax.text(x0 + w + 0.2, y + h / 2, "", ha="left", va="center",
                           fontsize=8, color="#ced6e0")
        self.gauges[label] = (fill, x0, w, pct)

    def _build(self):
        ax = self.ax
        # opaque dark background (axis('off') hides the axes patch, so draw one)
        ax.add_patch(Rectangle((0, 0), 10, 12, fc="#15151c", ec="none", zorder=0))
        self.title = ax.text(5, 11.4, "", ha="center", fontsize=11,
                             fontweight="bold", color="#ecf0f1")

        # === TOP: the phasic signal — a burst of dopamine sized by surprise ===
        ax.text(5, 10.55, "PHASIC: surprise burst", ha="center", fontsize=7.5,
                color="#7f8c9b", style="italic")
        self.glow = Circle((5, 9.7), 0.45, fc="#2ecc71", alpha=0.0, zorder=4)
        ax.add_patch(self.glow)
        self._node(5, 9.7, "Dopamine\nrelease", "#9b59b6")
        self.delta_txt = ax.text(7.6, 9.7, "", ha="left", va="center",
                                 fontsize=11, color="#2ecc71", fontweight="bold")

        # receptors: the surprise routes to D1 (raise cAMP) or D2 (lower cAMP)
        self._arrow(4.3, 9.3, 3.3, 8.7)
        self._arrow(5.7, 9.3, 6.7, 8.7)
        self.d1_glow = Circle((2.8, 8.2), 1.05, fc="#2ecc71", alpha=0.0, zorder=2)
        self.d2_glow = Circle((7.2, 8.2), 1.05, fc="#e74c3c", alpha=0.0, zorder=2)
        ax.add_patch(self.d1_glow)
        ax.add_patch(self.d2_glow)
        self._node(2.8, 8.2, "D1\n(Gs)", "#27ae60", fontsize=8)
        self._node(7.2, 8.2, "D2\n(Gi)", "#c0392b", fontsize=8)
        ax.text(2.8, 7.25, "+ cAMP", ha="center", fontsize=7.5, color="#2ecc71")
        ax.text(7.2, 7.25, "− cAMP", ha="center", fontsize=7.5, color="#e74c3c")
        self._arrow(3.1, 7.0, 4.4, 6.75, "#2ecc71")
        self._arrow(6.9, 7.0, 5.6, 6.75, "#e74c3c")
        self._node(5.0, 6.45, "Adenylyl cyclase", "#34495e", fontsize=8)

        # === BOTTOM: the steady signal cascading down to drive ===
        ax.text(0.2, 5.62, "STEADY: from value ↓", ha="left",
                fontsize=7.5, color="#7f8c9b", style="italic")
        self._gauge(5.0, "cAMP")
        self._gauge(3.95, "PKA")
        self._gauge(2.9, "Ion channel")
        self._gauge(1.85, "Drive")
        self.value_txt = ax.text(5.0, 1.0, "", ha="center", fontsize=9.5,
                                 color="#3498db", fontweight="bold")

    # ---- bind to a clicked object's dopamine state and show ----
    def bind(self, obj):
        self.obj = obj
        for fill, _x0, _w, _pct in self.gauges.values():
            fill.set_facecolor(obj.color)
        self.value_txt.set_color(obj.color)
        self.title.set_text(f"{obj.option.name}: dopamine cascade")
        self.ax.set_visible(True)
        self._refresh()

    def update(self):
        if self.obj is not None and self.ax.get_visible():
            self._refresh()

    def _refresh(self):
        obj = self.obj
        value = obj.option.value
        phasic = obj.pop.phasic if obj.pop is not None else 0.0
        drive = obj.pop.effective_drive() if obj.pop is not None else value

        # phasic flash at dopamine release: size = |delta|, green up / red down
        mag = min(abs(phasic) / self.PHASIC_MAX, 1.0)
        up = phasic >= 0
        self.glow.set_radius(0.45 + 1.1 * mag)
        self.glow.set_alpha(0.15 + 0.7 * mag)
        self.glow.set_facecolor("#2ecc71" if up else "#e74c3c")
        self.delta_txt.set_text(f"δ = {phasic:+.1f}")
        self.delta_txt.set_color("#2ecc71" if up else "#e74c3c")

        # the surprise lights up the matching receptor arm (D1 up / D2 down)
        self.d1_glow.set_alpha(0.75 * min(max(0.0, phasic) / self.PHASIC_MAX, 1.0))
        self.d2_glow.set_alpha(0.75 * min(max(0.0, -phasic) / self.PHASIC_MAX, 1.0))

        # steady tonic level propagates down each downstream gauge
        tonic = max(0.0, min(value / self.VAL_MAX, 1.0))
        levels = {"cAMP": tonic, "PKA": tonic ** 1.1, "Ion channel": tonic,
                  "Drive": max(0.0, min(drive / self.VAL_MAX, 1.0))}
        for name, (fill, _x0, w, pct) in self.gauges.items():
            lv = levels[name]
            fill.set_width(w * lv)
            pct.set_text(f"{lv * 100:.0f}%")
        self.value_txt.set_text(f"value = {value:.1f}   drive = {drive:.1f}")


class DDMPanel:
    """A simplified drift-diffusion view of ONE selected neuron.

    The neuron's membrane potential is the decision variable: it drifts (with
    noise) up from rest toward the firing threshold, and crossing the threshold
    is the 'decision' — a spike. This panel plots that trajectory, marks the
    threshold bound, and flags the moment the neuron fires. Like the cascade,
    it is a single reusable view re-pointed at whichever neuron you click.
    """

    def __init__(self, ax):
        self.ax = ax
        self.obj = None
        self.k = 0
        ax.set_facecolor("#0e0e12")
        self._build()
        ax.set_visible(False)   # hidden until a neuron is clicked

    def _build(self):
        ax = self.ax
        p = Params()
        ax.set_xlim(0, WINDOW)
        ax.set_ylim(p.V_REST - 12, p.V_SPIKE + 8)
        ax.set_xticks([])
        ax.tick_params(colors="#9aa0a6", labelsize=7)
        for spine in ax.spines.values():
            spine.set_color("#444")
        self.title = ax.set_title("", fontsize=9, color="#2c3e50",
                                  fontweight="bold")

        # the threshold (decision bound) and the resting start level
        self.thr_line = ax.axhline(p.V_THRESH, color="#e74c3c", ls="--", lw=1.5)
        ax.axhline(p.V_REST, color="#7f8c8d", ls=":", lw=1.0)
        ax.text(WINDOW * 0.97, p.V_THRESH + 2, "threshold → fire", ha="right",
                va="bottom", fontsize=7, color="#e74c3c")
        ax.text(WINDOW * 0.03, p.V_REST - 9, "start (rest)", ha="left",
                va="top", fontsize=7, color="#95a5a6")

        # the decision variable (membrane potential) over time
        (self.trace,) = ax.plot([], [], lw=1.6, color="#f1c40f")
        # flag shown the instant the neuron crosses threshold and fires
        self.fire_txt = ax.text(WINDOW * 0.5, p.V_SPIKE - 6, "", ha="center",
                                va="center", fontsize=11, color="#2ecc71",
                                fontweight="bold")

    def bind(self, obj, k):
        self.obj = obj
        self.k = k
        self.trace.set_color(obj.color)
        self.ax.set_visible(True)
        self._refresh()

    def update(self):
        if self.obj is not None and self.ax.get_visible():
            self._refresh()

    def _refresh(self):
        pop = self.obj.pop
        if pop is None or self.k >= pop.n:
            return
        hist = pop.history[self.k]
        self.trace.set_data(range(len(hist)), list(hist))
        self.title.set_text(f"{self.obj.option.name} neuron #{self.k} — DDM")
        # brightness spikes to 1.0 on firing, then fades: use it as the flag
        if pop.brightness[self.k] > 0.4:
            self.fire_txt.set_text("● fired!")
            self.thr_line.set(linewidth=2.6, color="#ff7675")
        else:
            self.fire_txt.set_text("")
            self.thr_line.set(linewidth=1.5, color="#e74c3c")


class DeskObject:
    """One thing on the desk: its option, scene artist, color, its own neuron
    population, an attention accumulator, and a clickable neuron panel."""

    def __init__(self, option, artist, color):
        self.option = option
        self.artist = artist
        self.color = color
        self.default_edge = (artist.get_edgecolor(), artist.get_linewidth())
        self.pop = None
        self.acc = 0.0               # accumulated spikes this race
        self.panel_ax = None         # neuron-viz axes (assigned by the app)
        self.panel = None            # VizPanel when shown
        self.shown = False           # is the neuron panel open?
        if option.present:
            self.spawn()

    def spawn(self, n=N_NEURONS):
        self.option.present = True
        self.pop = population(self.option, n=n)
        self.artist.set_visible(True)

    def clear(self):
        self.option.present = False
        self.pop = None
        self.acc = 0.0
        self.artist.set_visible(False)

    def highlight(self, on):
        if on:
            self.artist.set_edgecolor("#f1c40f")
            self.artist.set_linewidth(4)
        else:
            edge, lw = self.default_edge
            self.artist.set_edgecolor(edge)
            self.artist.set_linewidth(lw)


class DeskApp:
    """A single-window desk where every object races for attention. Each
    object's neurons spike into its own accumulator; the first to RACE_THRESHOLD
    wins and stays highlighted until someone else wins. The race, the per-object
    effort/present controls, and a 'send notification' button all live in this
    one window; click an object on the desk to open/close a live view of its
    neurons (several can be open at once)."""

    def __init__(self):
        self.winner = None           # the object highlighted as last winner
        self._build()                # creates the one reusable cascade panel

        self.fig.canvas.mpl_connect("button_press_event", self._on_click)
        self.ani = FuncAnimation(self.fig, self._update, interval=FRAME_MS,
                                 blit=False, cache_frame_data=False)

    # ---- build the whole single-window layout ----
    def _build(self):
        self.fig = plt.figure(figsize=(14, 8))
        self.fig.canvas.manager.set_window_title("Desk — attention race")

        # --- desk scene (left) ---
        self.ax = self.fig.add_axes([0.03, 0.34, 0.40, 0.60])
        self.ax.set_xlim(0, 10)
        self.ax.set_ylim(0, 7)
        self.ax.axis("off")
        self.ax.add_patch(Rectangle((0.5, 0.4), 9.0, 3.2, facecolor="#a9744f",
                                     edgecolor="#7a5436", lw=2, zorder=0))
        self.ax.text(5, 0.15, "Desk", ha="center", color="#7a5436", fontsize=11)
        self.hint = self.ax.text(5, 6.6, "Click an object to watch its neurons",
                                 ha="center", color="#34495e", fontsize=10)

        # homework sheet, centered on the desk
        hw_artist = Rectangle((4.0, 1.0), 2.0, 2.4, facecolor="#fdfdf6",
                              edgecolor="#cfcfc0", lw=1.5, zorder=2)
        self.ax.add_patch(hw_artist)
        for ly in (1.4, 1.8, 2.2, 2.6, 3.0):
            self.ax.plot([4.2, 5.8], [ly, ly], color="#bcc6d6", lw=0.8, zorder=2)
        self.homework = DeskObject(Option(name="Homework", present=True,
                                          value=13.0, effort=0.0), hw_artist, "#2e86de")

        # phone, off to the side
        ph_artist = Rectangle((0, 0), 1, 1, facecolor="#2c3e50",
                              edgecolor="#1a252f", lw=1.5, zorder=3)
        self.ax.add_patch(ph_artist)
        self.phone = DeskObject(Option(name="Phone", present=True, value=18.0,
                                       effort=0.4), ph_artist, "#e67e22")
        # buzzing badge shown while a notification is active
        self.buzz = self.ax.text(0, 0, "buzz!", ha="center", color="#e74c3c",
                                 fontsize=11, fontweight="bold", zorder=4)
        self.buzz.set_visible(False)

        self.objects = [self.phone, self.homework]

        # --- attention race (middle-top) ---
        n = len(self.objects)
        self.race_ax = self.fig.add_axes([0.50, 0.55, 0.18, 0.36])
        self.race_ax.set_xlim(0, RACE_THRESHOLD)
        self.race_ax.set_ylim(-0.5, n - 0.5)
        self.race_ax.set_yticks(range(n))
        self.race_ax.set_yticklabels([o.option.name for o in self.objects])
        self.race_ax.set_xticks([])
        self.race_ax.axvline(RACE_THRESHOLD, color="#c0392b", lw=2)
        self.race_ax.set_title("Attention race", fontsize=11)
        self.bars = self.race_ax.barh(range(n), [0] * n,
                                      color=[o.color for o in self.objects])

        # --- tweakable reward (between the race and the button) ---
        self.reward_ax = self.fig.add_axes([0.52, 0.50, 0.14, 0.03])
        self.reward_slider = Slider(self.reward_ax, "Reward", 0.0, REWARD_MAX,
                                    valinit=REWARD_MEAN, valstep=0.5,
                                    color="#e67e22")

        # --- send-notification button (under the race) ---
        self.notif_ax = self.fig.add_axes([0.50, 0.40, 0.18, 0.08])
        self.notif_btn = Button(self.notif_ax, "Send notification\nto phone",
                                color="#dfe6e9", hovercolor="#b2bec3")
        self.notif_btn.on_clicked(self._on_notify)

        # --- neuron panels (right), hidden until clicked ---
        top, bottom, gap = 0.94, 0.34, 0.05
        slot = (top - bottom - gap * (n - 1)) / n
        for i, obj in enumerate(self.objects):
            y = top - (i + 1) * slot - i * gap
            obj.panel_ax = self.fig.add_axes([0.72, y, 0.26, slot])
            obj.panel_ax.set_visible(False)

        # --- reusable cascade panel (bottom-middle), hidden until a neuron click ---
        self.cascade_ax = self.fig.add_axes([0.44, 0.04, 0.27, 0.35])
        self.cascade = CascadePanel(self.cascade_ax)

        # --- reusable DDM panel for the selected neuron (left of the cascade) ---
        self.ddm_ax = self.fig.add_axes([0.285, 0.06, 0.145, 0.25])
        self.ddm = DDMPanel(self.ddm_ax)

        # --- global view controls (top strip) ---
        # neuron count: few -> per-neuron trace/DDM view, many -> light-up grid
        self.n_neurons = N_NEURONS
        self.neuron_ax = self.fig.add_axes([0.07, 0.955, 0.17, 0.022])
        self.neuron_slider = Slider(self.neuron_ax, "Neurons", 1, 36,
                                    valinit=N_NEURONS, valstep=1, color="#16a085")
        self.neuron_slider.on_changed(self._on_neuron_count)

        # toggles to hide the cell-signaling cascade and the per-cell DDM
        self.show_cascade = True
        self.show_ddm = True
        self.fig.text(0.305, 0.965, "Show:", fontsize=9, va="center",
                      ha="right", fontweight="bold", color="#34495e")
        self.toggle_ax = self.fig.add_axes([0.31, 0.945, 0.13, 0.045])
        self.toggle_ax.set_facecolor("none")
        self.toggle_check = CheckButtons(self.toggle_ax, ["Cascade", "DDM"],
                                         [True, True])
        self.toggle_check.on_clicked(self._on_panel_toggle)

        # --- per-object controls: left and right bands, middle kept for cascade ---
        self.widgets = []                # keep refs alive
        labels = [c[0] for c in EFFORT_CHOICES]
        bands = [0.03, 0.70]             # phone on the left, homework on the right
        slot_w = 0.27
        for i, obj in enumerate(self.objects):
            sx = bands[i] if i < len(bands) else 0.03 + i * slot_w
            self.fig.text(sx + 0.02, 0.28, obj.option.name,
                          fontsize=11, fontweight="bold")

            radio_ax = self.fig.add_axes([sx + 0.01, 0.04, slot_w * 0.45, 0.22])
            radio_ax.set_facecolor("#f5f5f5")
            radio = RadioButtons(radio_ax, labels,
                                 active=self._effort_index(obj.option.effort))
            radio.on_clicked(lambda lbl, o=obj: self._on_effort(o, lbl))

            check_ax = self.fig.add_axes([sx + slot_w * 0.52, 0.10,
                                          slot_w * 0.4, 0.10])
            check_ax.set_facecolor("none")
            check = CheckButtons(check_ax, ["Present"], [obj.option.present])
            check.on_clicked(lambda lbl, o=obj: self._on_present(o))

            self.widgets += [radio, check]

        self._draw_phone()

    # ---- helpers ----
    @staticmethod
    def _effort_index(effort):
        diffs = [abs(effort - val) for _, val in EFFORT_CHOICES]
        return diffs.index(min(diffs))

    def _draw_phone(self):
        """Closer effort = big and near the front; farther = small and back."""
        frac = min(self.phone.option.effort / 1.5, 1.0)
        w = 0.9 - 0.4 * frac
        h = 1.7 - 0.7 * frac
        cx = 8.3
        cy = 1.6 + 1.6 * frac
        self.phone.artist.set_bounds(cx - w / 2, cy - h / 2, w, h)
        self.buzz.set_position((cx, cy + h / 2 + 0.4))

    # ---- control callbacks ----
    def _on_effort(self, obj, label):
        obj.option.effort = dict(EFFORT_CHOICES)[label]
        if obj is self.phone:
            self._draw_phone()

    def _on_present(self, obj):
        if obj.option.present:
            if self.winner is obj:       # drop the highlight if the winner leaves
                self.winner = None
            obj.clear()
            self._set_panel(obj, False)
        else:
            obj.spawn(self.n_neurons)
            if obj is self.phone:
                self._draw_phone()

    def _on_neuron_count(self, val):
        """Resize every pool; the panel auto-switches grid <-> trace/DDM view."""
        n = int(round(val))
        if n == self.n_neurons:
            return
        self.n_neurons = n
        for obj in self.objects:
            if obj.option.present:
                obj.spawn(n)                   # rebuild the pool with n neurons
                if obj.shown:
                    self._set_panel(obj, True)  # rebuild panel: grid <-> trace
        # keep the selected-neuron views pointing at a neuron that still exists
        if self.ddm.obj is not None and self.ddm.k >= n:
            self.ddm.k = 0

    def _on_panel_toggle(self, _label):
        """Show/hide the cell-signaling cascade and the per-cell DDM panels."""
        self.show_cascade, self.show_ddm = self.toggle_check.get_status()
        self.cascade_ax.set_visible(self.show_cascade
                                    and self.cascade.obj is not None)
        self.ddm_ax.set_visible(self.show_ddm and self.ddm.obj is not None)

    def _on_notify(self, _event):
        """A notification jolts the phone's dopamine drive, then resolves into a
        reward. The reward-prediction error nudges the phone's learned value."""
        if self.phone.pop is None:
            return
        self.phone.pop.boost += NOTIF_BOOST
        # The notification resolves: draw what checking it actually paid off.
        # Notifications are usually boring (low reward) and vary click-to-click,
        # so delta tends negative and the learned value drifts down without ever
        # settling.
        reward = max(0.0, random.gauss(self.reward_slider.val, REWARD_SD))
        option = self.phone.option
        delta = reward - option.value
        self.phone.pop.phasic = delta      # phasic burst the cascade flashes
        option.value += LEARN_RATE * delta

    # ---- click handling: a neuron opens the cascade, a desk object toggles it ----
    def _on_click(self, event):
        # click a neuron in an open panel -> bind the cascade + DDM to that neuron
        for obj in self.objects:
            if (obj.shown and obj.pop is not None and obj.panel is not None
                    and event.inaxes is obj.panel_ax):
                k = self._neuron_index(obj, event)
                if k is not None:
                    if self.show_cascade:
                        self.cascade.bind(obj)
                    if self.show_ddm:
                        self.ddm.bind(obj, k)
                return
        # otherwise, click a desk object -> toggle its neuron panel
        if event.inaxes is not self.ax:
            return
        for obj in self.objects:
            if obj.option.present and obj.artist.contains(event)[0]:
                self._set_panel(obj, not obj.shown)
                break

    @staticmethod
    def _neuron_index(obj, event):
        """Map a click in a neuron panel to that neuron's index (or None)."""
        if event.xdata is None or event.ydata is None:
            return None
        pop = obj.pop
        if obj.panel.mode == "grid":
            col = int(round(event.xdata))
            row = int(round(event.ydata))
            if 0 <= col < pop.cols and 0 <= row < pop.rows:
                k = row * pop.cols + col   # matches VizPanel._grid layout
                return k if 0 <= k < pop.n else None
            return None
        # trace mode: pick the neuron whose trace is nearest the click point
        xi = max(0, min(WINDOW - 1, int(round(event.xdata))))
        return min(range(pop.n),
                   key=lambda i: abs(pop.history[i][xi] - event.ydata))

    def _set_panel(self, obj, show):
        if show and obj.pop is not None:
            if obj.panel is None or obj.panel.pop is not obj.pop:
                obj.panel = VizPanel(obj.panel_ax, obj.pop)
                obj.panel_ax.set_title(f"{obj.option.name} neurons", fontsize=9)
            obj.panel_ax.set_visible(True)
            obj.shown = True
        else:
            obj.panel_ax.set_visible(False)
            obj.panel = None
            obj.shown = False

    # ---- the always-on loop ----
    def _update(self, _):
        # advance every present population and accumulate its spikes
        for obj in self.objects:
            if obj.pop is not None:
                obj.acc += obj.pop.step()

        # first across the line wins; it stays highlighted until someone beats it
        ready = [o for o in self.objects
                 if o.option.present and o.acc >= RACE_THRESHOLD]
        if ready:
            won = max(ready, key=lambda o: o.acc)
            if won is not self.winner:
                if self.winner is not None:
                    self.winner.highlight(False)
                won.highlight(True)
                self.winner = won
            for o in self.objects:
                o.acc = 0.0

        # race bars
        for bar, obj in zip(self.bars, self.objects):
            bar.set_width(obj.acc if obj.option.present else 0)

        # buzzing badge while the notification jolt is still meaningful
        self.buzz.set_visible(self.phone.option.present
                              and self.phone.pop is not None
                              and self.phone.pop.boost > 1.0)

        # refresh any open neuron panels
        for obj in self.objects:
            if obj.shown and obj.panel is not None:
                obj.panel.update()

        # refresh the reusable cascade + DDM views if a neuron is selected
        self.cascade.update()
        self.ddm.update()
        return []

    def run(self):
        plt.show()


if __name__ == "__main__":
    DeskApp().run()

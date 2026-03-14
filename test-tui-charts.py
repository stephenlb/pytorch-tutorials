from uniplot import plot
import numpy as np

# 1. Prepare data
x = np.linspace(0, 10, 50)
y_series = [np.sin(x), np.cos(x), np.sin(x) + 2]

# 2. Plot with corrected parameters
plot(
    y_series,
    title="Exhaustive Styling Example",
    x_labels="Time (s)",
    y_labels="Amplitude",
    x_unit="sec",
    y_unit="v",
    legend_labels=["Sine", "Cosine", "Offset Sine"],
    width=70,
    height=24,
    #rounded_corners=True,
    color=["cyan", "#ff00ff", (0, 255, 0)],
    #x_gridlines_color=["white"],
    #y_gridlines_color=["magenta"],
    #x_min=0, x_max=10,
    #y_min=-1.5, y_max=3.5,
    #y_gridlines=[-1, 0, 1, 2, 3],
    # Fixed: Use character_set instead of force_ascii
    #character_set="ascii" 
)

# -*- coding: utf-8 -*-
"""
=============================================================================
Experiment 1: Real-Time ECG Artifact Cancellation (ANC) using q-BLMS
Associated Paper: "A q-Jackson Block LMS for Robust 
                   Adaptive Filtering under Correlated Inputs"
=============================================================================
"""

import wfdb 
import numpy as np
import matplotlib.pyplot as plt
import time

# Import the core algorithm from the main module
from q_japa_algorithm import q_japa_stream

print("Downloading PhysioNet ECG record...")
# Load sample ECG data
record = wfdb.rdrecord('100', pn_dir='mitdb', sampto=3000)
s = record.p_signal[:, 0]
s = (s - np.mean(s)) / np.std(s) 
N = len(s)
fs = record.fs # 360 Hz
t_total = np.arange(N) / fs

# Generate correlated noise
np.random.seed(42)
v = np.random.randn(N)
u = np.zeros(N)
for n in range(1, N):
    u[n] = 0.9 * u[n-1] + v[n]

# Simulate the physiological path of the artifact
body_path = np.array([0.4, 0.2, -0.1, 0.05, -0.02])
electrode_noise = np.zeros(N)
for n in range(len(body_path), N):
    electrode_noise[n] = np.dot(body_path, u[n : n - len(body_path) : -1])

# Contaminated primary signal
d = s + electrode_noise


# ==========================================
# INTERACTIVE PLOT CONFIGURATION (ECG Monitor)
# ==========================================

M = 20
K = 3
eta = 0.00008
q_vec = np.full(M, 1.1)

plt.ion()
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10))
fig.suptitle('Real-Time Simulation: ECG Artifact Cancellation with q-BLMS', fontsize=14)

# Display window (e.g., last 3 seconds = 1080 samples)
window_seconds = 3
window_samples = int(window_seconds * fs)

# Subplots visual configuration
axes = [ax1, ax2, ax3]
titles = ['Signal d(n): Contaminated ECG (Input)', 
          'Signal e(n): Recovered ECG (q-BLMS Output)', 
          'Signal s(n): Clean Original ECG (Ground Truth)']
colors = ['#d62728', '#1f77b4', '#2ca02c']

lines = []
for ax, title, color in zip(axes, titles, colors):
    ax.set_title(title)
    ax.set_ylabel('Amplitude')
    ax.grid(True, alpha=0.3)
    ax.set_ylim([-8, 8]) # Adjust based on normalized signal variance
    line, = ax.plot([], [], color=color, linewidth=1.2)
    lines.append(line)

ax3.set_xlabel('Time (seconds)')
plt.tight_layout()

# Variables to store plotting history
t_history = []
d_history = []
e_history = []
s_history = []

print("Starting real-time monitor...")

# ==========================================
# REAL-TIME STREAMING PROCESSING LOOP
# ==========================================

# Refresh control: update plot every 10 iterations to prevent Matplotlib lag
refresh_rate = 10 

for n, error, weights in q_japa_stream(u, d, M, K, eta, q_vec):
    
    current_t = t_total[n]
    
    # Store current instance data
    t_history.append(current_t)
    d_history.append(d[n])
    e_history.append(error)
    s_history.append(s[n])
    
    # Keep only the sliding window data to avoid memory saturation
    if len(t_history) > window_samples:
        t_history.pop(0)
        d_history.pop(0)
        e_history.pop(0)
        s_history.pop(0)
        
    # Update graphical interface only on refresh_rate multiples
    if n % refresh_rate == 0:
        lines[0].set_data(t_history, d_history)
        lines[1].set_data(t_history, e_history)
        lines[2].set_data(t_history, s_history)
        
        # Dynamically shift X-axis
        x_min = max(0, current_t - window_seconds)
        x_max = max(window_seconds, current_t)
        
        for ax in axes:
            ax.set_xlim(x_min, x_max)
            
        fig.canvas.draw()
        fig.canvas.flush_events()

plt.ioff()
plt.show()

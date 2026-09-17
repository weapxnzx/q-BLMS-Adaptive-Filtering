# -*- coding: utf-8 -*-
"""
=============================================================================
Experiment 2: Tracking Capability and Learning Curve (MSE)
Associated Paper: "A q-Jackson Affine Projection Algorithm for Robust 
                   Adaptive Filtering under Correlated Inputs"
=============================================================================
"""

import wfdb 
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import PillowWriter
from collections import deque

# Import the core algorithm from the main module
from q_japa_algorithm import q_japa_stream

# ==========================================
# 1. SIGNAL GENERATION & DYNAMIC SYSTEM CHANGE
# ==========================================
print("Downloading PhysioNet ECG record...")
record = wfdb.rdrecord('100', pn_dir='mitdb', sampto=3000)
s = record.p_signal[:, 0]
s = (s - np.mean(s)) / np.std(s) 
N = len(s)
fs = record.fs # 360 Hz
t_total = np.arange(N) / fs

np.random.seed(42)
v = np.random.randn(N)
u = np.zeros(N)
for n in range(1, N):
    u[n] = 0.9 * u[n-1] + v[n]

# Define an abrupt change in the physiological path
body_path_initial = np.array([0.4, 0.2, -0.1, 0.05, -0.02])
body_path_final = np.array([-0.5, 0.3, 0.2, -0.1, 0.0]) 

electrode_noise = np.zeros(N)
movement_instant = int(4.0 * fs) # Abrupt change at t = 4.0s

for n in range(len(body_path_initial), N):
    if n < movement_instant:
        electrode_noise[n] = np.dot(body_path_initial, u[n : n - len(body_path_initial) : -1])
    else:
        electrode_noise[n] = np.dot(body_path_final, u[n : n - len(body_path_final) : -1])

# Contaminated primary signal
d = s + electrode_noise

# ==========================================
# 2. PLOT & ALGORITHM CONFIGURATION
# ==========================================
M = 20
K = 3
eta = 0.0002
q_vec = np.full(M, 1.5)
q_class = np.full(M, 1.0) 

plt.ion()
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
fig.suptitle('Tracking Capability: q-JAPA vs Classical JAPA', fontsize=14)

window_seconds = 3
window_samples = int(window_seconds * fs)

# Plot 1: Input Signal
ax1.set_title('Signal d(n): Contaminated ECG (Impact at t=4.0s)')
ax1.set_ylabel('Amplitude')
ax1.grid(True, alpha=0.3)
ax1.set_ylim([-8, 8])
line_d, = ax1.plot([], [], color='#d62728', linewidth=1.2)

# Plot 2: MSE Learning Curve (dB)
ax2.set_title('Smoothed Learning Curve: MSE in Logarithmic Scale (dB)')
ax2.set_ylabel('MSE (dB)')
ax2.set_xlabel('Time (seconds)')
ax2.grid(True, alpha=0.3)
ax2.set_ylim([-30, 20]) # Typical decibel range for this noise level
line_mse_q, = ax2.plot([], [], color='#1f77b4', linewidth=1.5, label='q-JAPA (q=1.5)')
line_mse_class, = ax2.plot([], [], color='#ff7f0e', linewidth=1.5, label='Classical JAPA (q=1.0)')
ax2.legend(loc='upper right')

plt.tight_layout()

# ==========================================
# 3. STREAMING LOOP WITH MSE CALCULATION
# ==========================================
t_history = []
d_history = []
mse_q_history = []
mse_class_history = []

# Buffers to smooth the squared error (Moving average of 100 samples)
smooth_window = 100
sq_err_q_buffer = deque(maxlen=smooth_window)
sq_err_class_buffer = deque(maxlen=smooth_window)

print("Starting monitor and recording GIF...")
writer = PillowWriter(fps=15)
writer.setup(fig, 'learning_curve_MSE.gif', dpi=100)

refresh_rate = 10 
stream_q = q_japa_stream(u, d, M, K, eta, q_vec)
stream_class = q_japa_stream(u, d, M, K, eta, q_class)

for (n, error_q, _), (_, error_class, _) in zip(stream_q, stream_class):
    current_t = t_total[n]
    
    # 1. Obtain pure instantaneous error (difference against Ground Truth, not 'd')
    e_real_q = error_q - s[n] 
    e_real_class = error_class - s[n]
    
    # 2. Feed squared error buffers
    sq_err_q_buffer.append(e_real_q**2)
    sq_err_class_buffer.append(e_real_class**2)
    
    # 3. Calculate moving average and convert to dB (using 1e-10 to avoid log(0))
    mse_q_db = 10 * np.log10(np.mean(sq_err_q_buffer) + 1e-10)
    mse_class_db = 10 * np.log10(np.mean(sq_err_class_buffer) + 1e-10)
    
    t_history.append(current_t)
    d_history.append(d[n])
    mse_q_history.append(mse_q_db)
    mse_class_history.append(mse_class_db)
    
    if len(t_history) > window_samples:
        t_history.pop(0)
        d_history.pop(0)
        mse_q_history.pop(0)
        mse_class_history.pop(0)
        
    if n % refresh_rate == 0:
        line_d.set_data(t_history, d_history)
        line_mse_q.set_data(t_history, mse_q_history)
        line_mse_class.set_data(t_history, mse_class_history)
        
        x_min = max(0, current_t - window_seconds)
        x_max = max(window_seconds, current_t)
        ax1.set_xlim(x_min, x_max)
        ax2.set_xlim(x_min, x_max)
            
        fig.canvas.draw()
        fig.canvas.flush_events()
        writer.grab_frame()

plt.ioff()
writer.finish()
print("Simulation finished! GIF saved as 'learning_curve_MSE.gif'.")
plt.show()

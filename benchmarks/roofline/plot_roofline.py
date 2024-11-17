import matplotlib.pyplot as plt
import numpy as np

# Data from the table
ai_tile = np.array([21.3333, 17.4545, 14.2222, 10.1053, 7.75758, 5.26027, 3.96899])
performance = np.array([452.372, 431.352, 434.016, 390.701, 350.647, 239.046, 185.079])

# Data from the new table
ai_tile_unopt = np.array([21.3333, 17.4545, 14.2222, 10.1053, 7.75758, 5.26027, 3.96899])
performance_unopt = np.array([334.003, 312.599, 279.955, 241.288, 201.802, 160.171, 127.972])

# Peak performance and bandwidth
peak_performance = 512  # OPS/cycle
peak_bandwidth = 64     # bytes/cycle

# Compute the ridge point (Operational Intensity where performance transitions)
oi_ridge = peak_performance / peak_bandwidth  # OI at the ridge point

# Create Operational Intensity values for the bandwidth-limited region
oi_bw = np.logspace(0, np.log10(oi_ridge), 100)
perf_bw = peak_bandwidth * oi_bw

# Create Operational Intensity values for the compute-limited region
oi_compute = np.logspace(np.log10(oi_ridge), 1.5, 100)
perf_compute = np.full_like(oi_compute, peak_performance)

# Plotting
plt.figure(figsize=(10, 6))

# Plot the peak bandwidth line (memory-bound region)
plt.loglog(oi_bw, perf_bw, label='Peak Bandwidth (64 bytes/cycle)', linestyle='--', color='blue')

# Plot the peak performance line (compute-bound region)
plt.loglog(oi_compute, perf_compute, label='Peak Performance (512 OPS/cycle)', linestyle='--', color='red')

# Plot the measured data points
plt.loglog(ai_tile, performance, 'o', label='Measured Data', markersize=8,
           markerfacecolor='green', markeredgecolor='black')


plt.loglog(ai_tile_unopt, performance_unopt, 'o', label='Measured Data', markersize=8,
           markerfacecolor='red', markeredgecolor='black')

# Labels and title
plt.xlabel('Operational Intensity [OPS/byte]', fontsize=12)
plt.ylabel('Performance [OPS/cycle]', fontsize=12)
plt.title('Roofline Model', fontsize=14)

# Annotate the ridge point
plt.axvline(x=oi_ridge, color='gray', linestyle=':', label=f'Ridge Point OI = {oi_ridge}')

# Set grid, legend, and layout
plt.grid(True, which="both", linestyle='--', linewidth=0.5)
plt.legend(fontsize=10)
plt.tight_layout()

# Display the plot
plt.savefig('roofline.png')

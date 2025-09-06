import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas_datareader import data as pdr
import pywt

# Parameters
symbol = 'DGS10'
start_date = '1963-01-01'
end_date = '2025-05-31'

# Load data
df = pdr.DataReader(symbol, 'fred', start_date, end_date).dropna()
series = np.diff(df[symbol].values)
dates = df.index[1:]

# Known structural break index in the differenced series
known_break_index = 14276

# Structural break detection function
def is_structural_break(coeffs, break_point, orig_length, threshold=3):
    for detail in coeffs[1:]:  # Skip approx coefficients
        idx = int(break_point * len(detail) / orig_length)
        window = detail[max(idx-2,0):min(idx+3,len(detail))]
        if np.any(np.abs(window) > threshold * np.std(detail)):
            return True
    return False

# Compute DWT once for efficiency
coeffs = pywt.wavedec(series, 'db4', level=3)

# Detect structural breaks over full series
break_points = [bp for bp in range(len(series)) if is_structural_break(coeffs, bp, len(series))]

# Plot differenced series with breaks
plt.figure(figsize=(14,6))
plt.plot(dates, series, label='DGS10 1st Difference')
for bp in break_points:
    color = 'red' if bp == known_break_index else 'orange'
    size = 60 if bp == known_break_index else 30
    plt.scatter(dates[bp], series[bp], color=color, s=size, label='Known 2020 Break' if bp == known_break_index else 'Detected Break (Other)', alpha=0.7, zorder=5 if bp == known_break_index else 4)
    if bp == known_break_index:
        plt.axvline(dates[bp], color='red', linestyle='--', alpha=0.7)
handles, labels = plt.gca().get_legend_handles_labels()
unique = dict(zip(labels, handles))
plt.legend(unique.values(), unique.keys())
plt.title('DGS10 Differenced Series with Structural Break Points')
plt.xlabel('Date')
plt.ylabel('First Difference')
plt.tight_layout()
plt.show()

# Equal length before and after samples for comparison
post_len = len(series) - known_break_index
before_sample = series[known_break_index - post_len:known_break_index]
after_sample = series[known_break_index:]

# Define cutoff slightly above mean to exclude small irrelevant area near mean
cutoff_factor = 0.05
cutoff_before = before_sample.mean() + cutoff_factor * before_sample.std()
cutoff_after = after_sample.mean() + cutoff_factor * after_sample.std()

# Calculate upper partial moments function
def upper_partial_moment(data, cutoff):
    # Only consider values above cutoff
    pos_dev = data[data > cutoff] - cutoff
    # Sort for step plot
    sorted_dev = np.sort(pos_dev)
    if len(sorted_dev) == 0:
        return np.array([]), np.array([])
    cumsum = np.cumsum(sorted_dev) / np.sum(sorted_dev)  # Normalize cumulative sum
    return sorted_dev, cumsum

b_up_x, b_up_cum = upper_partial_moment(before_sample, cutoff_before)
a_up_x, a_up_cum = upper_partial_moment(after_sample, cutoff_after)

# Plot upper partial moment comparison
plt.figure(figsize=(10,6))
if len(b_up_x) > 0:
    plt.step(b_up_x, b_up_cum, label='Upper Partial Moment Before 2020 Break', where='post')
if len(a_up_x) > 0:
    plt.step(a_up_x, a_up_cum, label='Upper Partial Moment After 2020 Break', where='post')
plt.title('Upper Partial Moments Comparison Before and After 2020 Break')
plt.xlabel('Excess Value Above Cutoff')
plt.ylabel('Normalized Cumulative Sum (Partial Moment)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

print("Number of detected break points:", len(break_points))
print("Known 2020 break detected at index:", known_break_index)
print("Cutoff before break:", cutoff_before)
print("Cutoff after break:", cutoff_after)

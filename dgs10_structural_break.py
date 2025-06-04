import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas_datareader import data as pdr
import pywt

# 2020 
# Random break at index: 14276
# Structural break detected at this point? True
#
# Download DGS10 data
symbol = 'DGS10'
start_date = '1963-01-01'
end_date = '2025-05-31'
df = pdr.DataReader(symbol, 'fred', start_date, end_date)
df = df.dropna()
series = np.diff(df[symbol].values)
dates = df.index[1:]

for break_point in range(14276, 14280):
        
    # Randomize a break point (not too close to edges)
    #break_point = np.random.randint(50, len(series)-50)

    # Compute DWT (3 levels)
    coeffs = pywt.wavedec(series, 'db4', level=3)

    # Function to check for structural break at break_point
    def is_structural_break(coeffs, break_point, threshold=3):

        for detail in coeffs[1:]:  # skip approximation

            # ratio break_point / len(series) gives the relative position of the break in the original
            idx = int(break_point * len(detail) / len(series))

            # selects a window of 5 coefficients centered at idx (from idx-2 to idx+2), handling boundaries.
            window = detail[max(idx-2,0):min(idx+3,len(detail))]

            if np.any(np.abs(window) > threshold * np.std(detail)):
                return True

        return False

    result = is_structural_break(coeffs, break_point)

    # Plot the series and break point
    plt.figure(figsize=(14,6))
    plt.plot(dates, series, label='DGS10 1st Difference')
    plt.axvline(dates[break_point], color='red', linestyle='--', label='Break Point')
    plt.scatter(dates[break_point], series[break_point], color='red', zorder=5)
    plt.title(f'DGS10 Differenced Series with Random Break Point\nStructural Break Detected: {result}')
    plt.xlabel('Date')
    plt.ylabel('First Difference')
    plt.legend()
    plt.tight_layout()
    plt.show()

    print("Random break at index:", break_point)
    print("Structural break detected at this point?", result)



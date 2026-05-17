import pandas as pd
import numpy as np
import os

# Generate EV Motor Sample (length 12000)
ev_signal = np.random.randn(12000)
pd.DataFrame(ev_signal, columns=['signal']).to_csv('sample_ev_motor_12000.csv', index=False)

# Generate Signal Health Sample (length 5000)
health_signal = np.random.randn(5000)
pd.DataFrame(health_signal, columns=['signal']).to_csv('sample_signal_health_5000.csv', index=False)

print("Generated sample_ev_motor_12000.csv and sample_signal_health_5000.csv")

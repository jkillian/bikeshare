import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('data/train.csv',parse_dates='datetime',index_col='datetime')
# Holiday
plt.scatter(df['holiday'], df['count'])
plt.show()

# Workingday
plt.scatter(df['workingday'], df['count'])
plt.show()

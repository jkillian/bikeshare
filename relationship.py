import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('data/train.csv', parse_dates='datetime', index_col='datetime')

# datetime
plt.scatter(list(df.index.values), list(df['count'].values))
plt.title("Datetime vs Count")

# Season
df.plot(kind='scatter', x='season', y='count')
plt.title("Season vs Count")

# Holiday
df.plot(kind='scatter', x='holiday', y='count')
plt.title("Holiday vs Count")

# Workingday
df.plot(kind='scatter', x='workingday', y='count')
plt.title("Workingday vs Count")

# Weather
df.plot(kind='scatter', x='weather', y='count')
plt.title("Weather vs Count")

# Humidity
df.plot(kind='scatter', x='humidity', y='count')
plt.title("Humidity vs Count")

# temp
df.plot(kind='scatter', x='temp', y='count', alpha=0.1)
plt.title("Temp vs Count")

# atemp
df.plot(kind='scatter', x='atemp', y='count')
plt.title("Atemp vs Count")

# Windspeed
df.plot(kind='scatter', x='windspeed', y='count')
plt.title("Windspeed vs Count")

# Show all plots
plt.show()

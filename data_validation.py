
# coding: utf-8

# ### Setup ###
# #### Import Libraries ####

# In[78]:

get_ipython().magic('pylab inline')
import pandas as pd
import datetime
import matplotlib.pyplot as plt


# #### Read Data ####

# In[79]:

train_data = pd.read_csv('data/train.csv',parse_dates=[0])
test_data = pd.read_csv('data/test.csv',parse_dates=[0])
# Show a few rows so we know what we're working with
train_data[100:105]


# ### Validate Data ###
# #### Basic bounds checks on data ####

# In[95]:

def err(msg, i):
    print("Error in Row {}: {}".format(i, msg));

for i, row in train_data.iterrows():
    time, season, holiday, workingday = row['datetime'], row['season'], row['holiday'], row['workingday']
    weather, temp, atemp, humidity, windspeed = row['weather'], row['temp'], row['atemp'], row['humidity'], row['windspeed']
    casual, reg, cnt = row['casual'], row['registered'], row['count']
    
    if time > datetime.datetime.now():
        err('timestamp from the future', i)
    if season not in [1,2,3,4]:
        err('invalid season', i)
    if holiday not in [False, True]:
        err('invalid holiday', i)
    if workingday not in [False, True]:
        err('invalid workingday', i)
    if holiday and workingday:
        err("can't be holiday and workingday", i)
    if weather not in [1,2,3,4]:
        err('invalid weather conditions', i)
    if temp < -35 or temp > 50: # record temps -26, 41
        err('invalid temperature', i)
    if atemp < -50 or atemp > 60:
        err('invalid feels like temp (atemp)', i)
    if humidity < 0 or humidity > 100:
        err('invalid humidity', i)
    if windspeed < 0 or windspeed > 80:
        err('invalid windspeed', i)
    if casual < 0 or reg < 0 or cnt < 0:
        err('Negative customers', i)
    if casual + reg != cnt:
        err('count != casual + registered', i)
        
print("Error Check Complete")


# #### Histograms of Data ####
# ##### Setup plotting #####

# In[81]:

plt.style.use('ggplot')


# ##### Plot all fields (except datetime for now) #####

# In[89]:

four_bins = [0.5,1.5,2.5,3.5,4.5]
two_bins = [-0.5,0.5,1.5]
plt.figure()
train_data['season'].plot(kind='hist', bins=four_bins, title='Season')
plt.figure()
train_data['holiday'].plot(kind='hist', bins=two_bins, title="Holiday")
plt.figure()
train_data['workingday'].plot(kind='hist', bins=two_bins, title="Working Days")
plt.figure()
train_data['weather'].plot(kind='hist', bins=four_bins, title='Weather')
plt.figure()
train_data['temp'].plot(kind='hist', title='Temp (C)')
plt.figure()
train_data['atemp'].plot(kind='hist', title='Feels Like Temp (C)')
plt.figure()
train_data['humidity'].plot(kind='hist', title='Humidity')
plt.figure()
train_data['windspeed'].plot(kind='hist', title='Windspeed (? units)')
plt.figure()
train_data['casual'].plot(kind='hist', title='Casual Customers')
plt.figure()
train_data['registered'].plot(kind='hist', title='Registered Customers')
plt.figure()
train_data['count'].plot(kind='hist', title='Total Customers')

plt.show()


# ##### Some random checks after looking at histograms #####
# The data that's following doesn't quite make sense:
# * Shouldn't we have temperatures below 0C?
# * Why only one hour ever with severe weather?

# In[94]:

print('Min Temp: {}'.format(min(train_data['temp'])))
print('Max Temp: {}'.format(max(train_data['temp'])))
print('Min aTemp: {}'.format(min(train_data['atemp'])))
print('Max aTemp: {}'.format(max(train_data['atemp'])))
print('Num Days with Weather of 4: {}'.format(len(train_data[train_data['weather'] == 4])))


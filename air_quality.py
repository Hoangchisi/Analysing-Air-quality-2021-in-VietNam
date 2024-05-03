#----------------------------------import libraries---------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

#--------------------------------clean data-----------------------------------
#open data file
df = pd.read_csv("historical_air_quality_2021_en.csv")

def convert(x,k):
  return x*1000000*k
def convert_aqi(aqi_table,k):
  new_aqi_table = []
  for i in aqi_table:
    new_aqi_table.append(convert(i,k))
  return new_aqi_table

#delete unnecessary datas
del df['AQI index']
del df["Url"]
del df["Status"]
del df["Station name"]
del df["Data Time Tz"]
del df["Alert level"]
del df["Station ID"]

keys = ['Location','Dominent pollutant','CO','Dew','Humidity','NO2','O3','Pressure','PM10','PM2.5','SO2','Temperature','Wind','Data Time S']

#remove duplicates rows
df.drop_duplicates(inplace=True)

#remove rows having '-'
for x in df.index:
    for k in keys:
        if df.loc[x, k] == "-" or str(df.loc[x,'Location']) == 'nan':
            df.drop(x,inplace=True)
            break

#convert to date time
df['Data Time S'].to_string
df['Data Time S'] = pd.to_datetime(df['Data Time S'])
df['Month'] = df['Data Time S'].dt.month
df['Day'] = df['Data Time S'].dt.day
df['Hour'] = df['Data Time S'].dt.hour

#handle wrong format data
for x in df.index:
    df.loc[x,'Pressure'] = str(df.loc[x,'Pressure'])
    df.loc[x,'Pressure'] = df.loc[x,'Pressure'].replace(",", "")

for x in df.index:
    if df.loc[x,'Pressure'] == "nan":
        df.loc[x,'Pressure'] = ""

df['Pressure'] = pd.to_numeric(df['Pressure'])

#fill blanks with mean value of each column
for k in keys[2:12]:
    df[k] = pd.to_numeric(df[k])
    df[k] = df[k].fillna(value = df[k].mean())

# add latitude and longitude columns
df['Location'].to_string
for i in df.index:
    x = df.loc[i,'Location'].split(',')

    df.loc[i,'Latitude'] = float(x[0])
    df.loc[i,'Longitude'] = float(x[1])

#------------------------------calculating data------------------------------

#initial AQI table
aqiTable = {
    'PM10':{
        'Cmin':[0, 55, 155, 255, 355, 425, 505],
        'Cmax':[54, 154, 254, 354, 424, 504, 604],
        'Imin':[0, 51, 101, 151, 201, 301, 401],
        'Imax':[50, 100, 150, 200, 300, 400, 500],
    },
    'PM2.5':{
        'Cmin':[0, 12.1, 35.5, 55.5, 150.5, 250.5, 350.5],
        'Cmax':[12, 35.4, 55.4, 150.4, 250.4, 350.4, 500],
        'Imin':[0, 51, 101, 151, 201, 301, 401],
        'Imax':[50, 100, 150, 200, 300, 400, 500],
    },
    'CO':{
        'Cmin':[0.0,4.5,9.5,12.5,15.5,30.5,40.5],
        'Cmax':[4.4,9.4,12.4,15.4,30.4,40.4,50.4],
        'Imin':[0, 51, 101, 151, 201, 301, 401],
        'Imax':[50, 100, 150, 200, 300, 400, 500],
    },
    'SO2':{
        'Cmin':[0,36,76,186,305,605,805],
        'Cmax':[35,75,185,304,604,804,1004],
        'Imin':[0, 51, 101, 151, 201, 301, 401],
        'Imax':[50, 100, 150, 200, 300, 400, 500],
    },
    'O3':{
        'Cmin':[0.000,0.055,0.125,0.165,0.205,0.405,0.505],
        'Cmax':[0.054,0.070,0.164,0.204,0.404,0.504,0.604],
        'Imin':[0, 51, 101, 151, 201, 301, 401],
        'Imax':[50, 100, 150, 200, 300, 400, 500],
    },
    'NO2':{
        'Cmin':[0,54,101,361,650,1250,1650],
        'Cmax':[53,100,360,649,1249,1649,2049],
        'Imin':[0, 51, 101, 151, 201, 301, 401],
        'Imax':[50, 100, 150, 200, 300, 400, 500],
    }
}

for i in aqiTable.keys():
  if i != 'PM10' and i != 'PM2.5':
    if i!= 'NO2' and i != 'SO2': 
        aqiTable[i]['Cmin'] = convert_aqi(aqiTable[i]['Cmin'],1)
        aqiTable[i]['Cmax'] = convert_aqi(aqiTable[i]['Cmax'],1)
    else:
        aqiTable[i]['Cmin'] = convert_aqi(aqiTable[i]['Cmin'],10**(-3))
        aqiTable[i]['Cmax'] = convert_aqi(aqiTable[i]['Cmax'],10**(-3))

#calculate AQI of each rows
for x in df.index:
    aqi = 0
    for k in aqiTable.keys():
        pos = 0
        for i in range(0,7):
            if df.loc[x,k] >= aqiTable[k]['Cmin'][i]:
                pos = i
        aqi = max(aqi, (df.loc[x,k] - aqiTable[k]['Cmin'][pos])*(aqiTable[k]['Imax'][pos] - aqiTable[k]['Imin'][pos])/(aqiTable[k]['Cmax'][pos] - aqiTable[k]['Cmin'][pos]) + aqiTable[k]['Imin'][pos])
    df.loc[x,'AQI'] = aqi

#---------------------------visually represent data---------------------------------
aqiLevels = [0,50,100,150,200,300,400,500]

#AQI's distribution of VietNam in 2021
xlist = np.array(df['AQI'])
plt.figure(figsize = (12,6))
sns.histplot(xlist, bins = aqiLevels, edgecolor = 'black',kde = True)

plt.xlabel('AQI')
plt.ylabel('Frequency')
plt.title('Air Quality Index of VietNam in 2021.')

plt.savefig("AQI's distribution of VietNam in 2021.png")

#AQI's distribution follow by coordinates
plt.figure(figsize=(12,6))

aqi_north = np.array(df[df['Latitude'] >= 16]['AQI'])
plt.subplot(1,2,1)
sns.histplot(aqi_north, bins = aqiLevels,edgecolor = 'black', kde = True)
plt.xlabel('AQI')
plt.ylabel('Frequency')
plt.title('AQI in Northern of VietNam')

aqi_south = np.array(df[df['Latitude'] < 16]['AQI'])
plt.subplot(1,2,2)
sns.histplot(aqi_south, bins = aqiLevels,edgecolor = 'black', kde = True)
plt.xlabel('AQI')
plt.ylabel('Frequency')
plt.title('AQI in the Southern of VietNam')

plt.savefig("AQI's distribution follow by coordinates.png")

#AQI follow by month
plt.figure(figsize=(12,6))

aqi_month = [0]*13
for m in range(0,13):
   aqi_month[m] = df[df['Month'] == m]['AQI'].mean()

plt.bar(np.array(range(1,12)), aqi_month[1:12], edgecolor = 'black')
plt.xlabel('Month')
plt.ylabel('AQI')
plt.title("AQI's distribution follow by month")

for index in range(1,12):
   plt.annotate(f'{round(aqi_month[index],0)}\n', xy=(index, aqi_month[index]), ha='center', va='center')
plt.savefig("AQI follow by month.png")
#AQI follow by day
plt.figure(figsize=(12,6))

aqi_day = [0]*32
for d in range(0,32):
   aqi_day[d] = df[df['Day'] == d]['AQI'].mean()

plt.bar(np.array(range(1,32)), aqi_day[1:32], edgecolor = 'black')
plt.xlabel('Day')
plt.ylabel('AQI')
plt.title("AQI's distribution follow by day")

for index in range(1,32):
   plt.annotate(f'{round(aqi_day[index],0)}\n', xy=(index, aqi_day[index]), ha='center', va='center')
plt.savefig("AQI follow by day.png")
#AQI follow by hour
plt.figure(figsize=(12,6))

aqi_hour = [0]*24
for t in range(0,24):
   aqi_hour[t] = df[df['Hour'] == t]['AQI'].mean()

plt.bar(np.array(range(0,24)), aqi_hour[0:24], edgecolor = 'black')
plt.xlabel('Hour')
plt.ylabel('AQI')
plt.title("AQI's distribution follow by hour")

for index in range(0,24):
   plt.annotate(f'{round(aqi_hour[index],0)}\n', xy=(index, aqi_hour[index]), ha='center', va='center')
plt.savefig("AQI follow by hour.png")

plt.show()

#---------------------save clean data----------------------

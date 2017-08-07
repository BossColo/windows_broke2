import numpy as np
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
import matplotlib.dates as date
import seaborn as sns
import urllib

sns.set_context('talk')

data_crime_raw = pd.read_csv('.\\NYPD_Complaint_Data_Historic.csv',
                             usecols=['CMPLNT_FR_DT', 'OFNS_DESC', 'LAW_CAT_CD', 'Latitude', 'Longitude'],
                             dtype={'OFNS_DESC':'category', 'LAW_CAT_CD':'category', 'Latitude':float, 'Longitude':float})

data_crime_raw['CMPLNT_FR_DT'] = pd.to_datetime(data_crime_raw['CMPLNT_FR_DT'], format='%m/%d/%Y', errors='coerce')

data_311_raw = pd.read_csv('.\\311_Service_Requests_from_2010_to_Present.csv',
                           usecols=['Created Date', 'Complaint Type', 'Descriptor', 'Latitude', 'Longitude'],
                           dtype={'Complaint Type':'category', 'Descriptor':'category', 'Latitude':float, 'Longitude':float})

data_311_raw['created_date'] = pd.to_datetime(data_311_raw['Created Date'], format='%m/%d/%Y %I:%M:%S %p', errors='coerce')

data_crime = data_crime_raw[data_crime_raw.CMPLNT_FR_DT > pd.to_datetime(dt.date(2010,1,1))].dropna()
data_311 = data_311_raw[data_311_raw.created_date < pd.to_datetime(dt.date(2016,1,1))].dropna()

minlat = data_crime.Latitude.min()
maxlat = data_crime.Latitude.max()
minlon = data_crime.Longitude.min()
maxlon = data_crime.Longitude.max()

latrange = np.arange(minlat, maxlat+0.02, 0.02)
lonrange = np.arange(minlon, maxlon+0.02, 0.02)

data_crime = data_crime[data_crime.LAW_CAT_CD != 'VIOLATION']

d_c_grouped = data_crime.groupby(
    by=[pd.cut(data_crime['Latitude'], latrange),
        pd.cut(data_crime['Longitude'], lonrange),
        pd.TimeGrouper(key='CMPLNT_FR_DT',freq='M')])

d_311_grouped = data_311.groupby(
    by=[pd.cut(data_311['Latitude'], latrange),
        pd.cut(data_311['Longitude'], lonrange),
        pd.TimeGrouper(key='created_date',freq='M')])

max_var_loc = d_c_grouped.size().unstack().var(axis=1).argmax()

data_crime_window = data_crime_raw[data_crime_raw.CMPLNT_FR_DT.between(
    pd.to_datetime(dt.date(2010,1,1)), pd.to_datetime(dt.date(2015,2,1)))].dropna()
data_311_window = data_311_raw[data_311_raw.created_date.between(
    pd.to_datetime(dt.date(2010,1,1)), pd.to_datetime(dt.date(2015,2,1)))].dropna()

d_c_win_grouped = data_crime_window.groupby(
    by=[pd.cut(data_crime_window['Latitude'], latrange),
        pd.cut(data_crime_window['Longitude'], lonrange),
        pd.TimeGrouper(key='CMPLNT_FR_DT',freq='5D')])

d_311_win_grouped = data_311_window.groupby(
    by=[pd.cut(data_311_window['Latitude'], latrange),
        pd.cut(data_311_window['Longitude'], lonrange),
        pd.TimeGrouper(key='created_date',freq='5D')])

plt.close('all')
fig, ax = plt.subplots(figsize=(20,10))

d_c_plot = d_c_win_grouped.size().unstack().loc[max_var_loc][:-1]
crime_regression = np.polyfit(d_c_plot.index.astype(np.int64), d_c_plot.data, 1)
plot_crime = ax.plot_date(d_c_plot.index, d_c_plot.data, 'b', label='NYPD complaints')
reg_crime = ax.plot_date(d_c_plot.index, crime_regression[0]*d_c_plot.index.astype(int)+crime_regression[1], 'r',
                        label='NYPD complaints linear regression')

d_311_plot = d_311_win_grouped.size().unstack().loc[max_var_loc][:-1]
regression = np.polyfit(d_311_plot.index.astype(np.int64), d_311_plot.data, 1)
plot_311 = ax.plot_date(d_311_plot.index, d_311_plot.data, 'g', label='311 complaints')
reg_311 = ax.plot_date(d_311_plot.index, regression[0]*d_311_plot.index.astype(int)+regression[1], 'm',
                      label='311 complaints linear regression')

ax.legend()

plt.savefig('NYPDand311trend.png', format='png')

plt.close('all')
fig, ax = plt.subplots(figsize=(20, 10))

d_311_largest = d_311_grouped['Complaint Type'].value_counts().unstack().loc[max_var_loc].sum().nlargest(30)
d_311_select = data_311_window[data_311_window['Complaint Type'].isin(d_311_largest.index)]

d_311_select_grouped = d_311_select.groupby(
    by=[pd.cut(d_311_select['Latitude'], latrange),
        pd.cut(d_311_select['Longitude'], lonrange),
        pd.TimeGrouper(key='created_date',freq='M')])

d_311_select_plot = d_311_select_grouped['Complaint Type'].value_counts().unstack().loc[max_var_loc][:-1]
d_311_select_plot.plot(kind='bar', stacked=True, ax=ax)
ax.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
xtl=[dt.datetime.strptime(item.get_text(), '%Y-%m-%d %H:%M:%S').strftime('%b %Y') for item in ax.get_xticklabels()]
_=ax.set_xticklabels(xtl)
fig.subplots_adjust(right=0.8)
plt.savefig('complaints311categories.png', format='png')
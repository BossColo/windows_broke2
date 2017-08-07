import numpy as np
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
import matplotlib.dates as date
import seaborn as sns
from scipy import stats

sns.set_context('talk')


data_crime_raw = pd.read_csv('.\\NYPD_Complaint_Data_Historic.csv',
                             usecols=['CMPLNT_FR_DT', 'OFNS_DESC', 'LAW_CAT_CD', 'Latitude', 'Longitude', 'BORO_NM'],
                             dtype={'OFNS_DESC':'category', 'LAW_CAT_CD':'category', 'BORO_NM':'category',
                                    'Latitude':float, 'Longitude':float})

data_crime_raw['CMPLNT_FR_DT'] = pd.to_datetime(data_crime_raw['CMPLNT_FR_DT'], format='%m/%d/%Y', errors='coerce')


data_311_raw = pd.read_csv('.\\311_Service_Requests_from_2010_to_Present.csv',
                           usecols=['Created Date', 'Complaint Type', 'Descriptor', 'Latitude', 'Longitude', 'Borough'],
                           dtype={'Complaint Type':'category', 'Descriptor':'category', 'Borough':'category',
                                  'Latitude':float, 'Longitude':float})

data_311_raw['created_date'] = pd.to_datetime(data_311_raw['Created Date'], format='%m/%d/%Y %I:%M:%S %p', errors='coerce')


data_crime = data_crime_raw[data_crime_raw.CMPLNT_FR_DT > pd.to_datetime(dt.date(2010,1,1))].dropna()
data_311 = data_311_raw[data_311_raw.created_date < pd.to_datetime(dt.date(2016,12,31))].dropna()


minlat = data_crime.Latitude.min()
maxlat = data_crime.Latitude.max()
minlon = data_crime.Longitude.min()
maxlon = data_crime.Longitude.max()

latrange = np.arange(minlat, maxlat+0.02, 0.02)
lonrange = np.arange(minlon, maxlon+0.02, 0.02)


data_crime = data_crime[data_crime.LAW_CAT_CD != 'VIOLATION']


d_311_grouped = data_311.groupby(
    by=[pd.cut(data_311['Latitude'], latrange),
        pd.cut(data_311['Longitude'], lonrange)])

d_c_grouped = data_crime.groupby(
    by=[pd.cut(data_crime['Latitude'], latrange),
    pd.cut(data_crime['Longitude'], lonrange)])

data = d_311_grouped.size().to_frame('311').merge(d_c_grouped.size().to_frame('crime'), left_index=True, right_index=True)


plt_311 = np.array(data['311'].apply(int))
plt_crime = np.array(data['crime'].apply(int))
plt.close('all')
sns.regplot(x=plt_311, y=plt_crime)
plt.suptitle('311 v total crime 2010-2016\ngrouped by location')
plt.xlabel('Total 311 complaints')
plt.ylabel('Total reported crime')
plt.savefig('311vcrime.png', format='png')

d_311_grouped = data_311[data_311.Borough == 'MANHATTAN'].groupby(
    by=[pd.TimeGrouper(key='created_date',freq='M'), 'Complaint Type']).size().to_frame('total')

d_c_grouped = data_crime[data_crime.BORO_NM == 'MANHATTAN'].groupby(
    by=pd.TimeGrouper(key='CMPLNT_FR_DT',freq='M')).size().to_frame('total')

d_c_grouped2 = d_c_grouped.copy()
d_c_grouped2['Complaint Type'] = 'crime'
d_c_grouped2.set_index('Complaint Type', append=True, inplace=True)
d_c_grouped2.index.rename('created_date', level=0, inplace=True)


crimecorr = d_311_grouped.unstack(level=1)['total'].corrwith(d_c_grouped2['total'])
corridx = ['Complaint Type'] + list(crimecorr[crimecorr > .5].keys()) + ['crime']


d_all_grouped = d_c_grouped2.combine_first(d_311_grouped)


corr = d_all_grouped.unstack(level=1).corr().dropna(axis=1, how='all')


corr2 = corr.reset_index(level=0, drop=True)
corr2.columns = corr2.columns.droplevel()
corr2.reset_index(inplace=True)

corr3 = corr2[corr2['Complaint Type'].isin(corridx)][corridx].set_index('Complaint Type')


sns.set(style="white")

mask = np.zeros_like(corr3, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

f, ax = plt.subplots(figsize=(11, 9))

cmap = sns.diverging_palette(220, 10, as_cmap=True)

sns.heatmap(corr3, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})

f.suptitle('Correlation Heat Map')
f.subplots_adjust(left = 0.1, bottom = 0.28)

plt.savefig('CorrelationMap.png', format='png')
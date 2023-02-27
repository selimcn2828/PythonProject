#!/usr/bin/env python
# coding: utf-8

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from fbprophet import Prophet
import holidays


# In[2]:


#!pip install fbprophet


# In[3]:


df_daily = pd.read_csv("bike_sharing_daily.csv")
df_hourly = pd.read_csv("bike_sharing_hourly.csv")
df_deneme = pd.read_excel("Last_30_day_KPIs.xlsx", sheet_name="Sayfa1", engine='openpyxl')


# In[4]:


def fixing_datatypes(df):
    # Fixing the datatypes 
    df['dteday'] = df['dteday'].astype('datetime64')
    df.loc[:,'season':'mnth'] = df.loc[:,'season':'mnth'].astype('category')
    df[['holiday','workingday']] = df[['holiday','workingday']].astype('bool')
    df[['weekday','weathersit']] = df[['weekday','weathersit']].astype('category')
    return df


# In[5]:


df_daily = fixing_datatypes(df_daily)
df_hourly = fixing_datatypes(df_hourly)
df_hourly['hr'] = df_hourly['hr'].astype('category')


# In[6]:


datas = []
for col in df_deneme.columns[3:]:
    datas.append(df_deneme[['Date', col]])
datas.pop(0)    
print(datas[4])


# In[7]:


for data in datas:
    data.columns = ["ds", "y"]
    data = pd.DataFrame(data=data)
    print(data.head())


# In[8]:


df_daily = df_daily[["cnt","dteday"]]
df_daily.columns = ["y", "ds"]
df_daily.head()


# ## Lets Predict

# In[9]:


# display settings
pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)
pd.set_option("display.float_format", lambda x: "%.3f" % x)
pd.set_option("display.expand_frame_repr", False)
pd.set_option("display.width", 500)

def fit_predict_model(dataframe):
    
    # seasonality, holidays
    # it can integrate football matches to data.
    m = Prophet(yearly_seasonality = True, daily_seasonality = True)
    
    m.add_country_holidays(country_name="TR")
    
    m = m.fit(dataframe)
    
    forecast = m.predict(dataframe)
    forecast["fact"] = dataframe["y"].reset_index(drop = True)
    return forecast

preds = []
for i, data in enumerate(datas):
    preds.append(fit_predict_model(data))
preds.pop(0)


# In[10]:


print(preds[0].head())


# In[12]:


pd.options.plotting.backend = "plotly"
preds[0].plot(x='ds', y=[ "yhat_lower", "fact","yhat_upper", "yhat"])


# # Detecting Anomalies:
# * The light blue boundaries in the above graph are yhat_upper and yhat_lower.
# * If y value is greater than yhat_upper and less than yhat lower then it is an anomaly.
# * Also getting the importance of that anomaly based on its distance from yhat_upper and yhat_lower.

# In[17]:


def detect_anomalies(forecast):
    forecasted = forecast[["ds","trend", "yhat", "yhat_lower", "yhat_upper", "fact"]].copy()
    #forecast["fact"] = df["y"]

    forecasted["anomaly"] = 0
    forecasted.loc[forecasted["fact"] > forecasted["yhat_upper"], "anomaly"] = 1
    forecasted.loc[forecasted["fact"] < forecasted["yhat_lower"], "anomaly"] = -1

    #anomaly importances
    forecasted["importance"] = 0
    forecasted.loc[forecasted["anomaly"] ==1, "importance"] = \
        (forecasted["fact"] - forecasted["yhat_upper"])/forecast["fact"]
    forecasted.loc[forecasted["anomaly"] ==-1, "importance"] = \
        (forecasted["yhat_lower"] - forecasted["fact"])/forecast["fact"]
    
    return forecasted

pred = detect_anomalies(preds[0])


# In[20]:


# Finalde günlük anomali detection yapabiliyoruz. Bunu tüm KPI setlerine yedirmek lazım.
# Sezonsallık ve tatil günlerini modelimiz dikkate alıyor (maç günlerini de ekleyeiblirsin 
    # link --> (https://facebook.github.io/prophet/docs/seasonality,_holiday_effects,_and_regressors.html#modeling-holidays-and-special-events)

# ADD last 30 days trend info in a "trend_info" column 
    # how to get last 30 days trend info --> OLS (ordinary least square) bir şekilde trendin yönünü öğren (açıya bakabilirsin) -- KPI türüne dikkat

pred.sample(32)


# References:
# * http://www-personal.umich.edu/~mejn/cp/programs.html
# * https://towardsdatascience.com/anomaly-detection-time-series-4c661f6f165f
# * https://github.com/altair-viz/altair/issues/1270
# 

# **NOTE:**
# 
# * Otomatize etmek için:
# 
# -- Datanın hazırlanması
# * Data'nın otomatik alınması
#     * DB'ye direk erişimin olucak oraya sql atıcan ya da günlük olarak dosyayı ilgili kişi bir yere atıcak sen de ftp ile otomatik alıcan.
# 
# -- Model
# * Model'i kur ama her defasında train etme, yani modelinin dump'ını alarak modeli kaydet böylece zamandan kazanmış olursun. (modele ilk adımda ds-y)
#     * KPI bazlı yapmak için döngü yazmak şart.
#     * For döngüsünden kurtulmak için Spark Prophet'ı araştırırsan çözümü bulursun. (çok çok sonra)
#     
# * Sonuçları log halinde bir tabloya basman şart (tablo izini vs Elif hanım'dan talep et)
# 
# -- DAG
# * Bu bütün adımları her gün çalışıcak şekilde yapman için de Airflow'da dag yazılması lazım. 
#     * Bunun için de Nuhi abi & Veli abi (Gökhan abinin takımı) vs destek için Elif hanım ile görüş
# 
# * Çıktıları nasıl göstericeğin noktasında çalışman lazım. (tableu)

# In[ ]:





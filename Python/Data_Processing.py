# -*- coding: utf-8 -*-
# Python 3.7
# Joey G.

# =============================================================================
# HOUSE KEEPING

# Import libraries
import time
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.stattools import adfuller
from itertools import compress
from googletrans import Translator
from geopy.geocoders import Nominatim

# Confirm working directory
print("Working Directory: ", os.getcwd())

# =============================================================================
# DATA CLEANING
# NOTE: The following repeats some steps that were initially done in SQL queries

# Import original Kaggle files
df_items = pd.read_csv(r'YOUR_PATH\items.csv')
df_itemcategories = pd.read_csv(r'YOUR_PATH\item_categories.csv')
df_shops = pd.read_csv(r'YOUR_PATH\shops.csv')
df_train = pd.read_csv(r'YOUR_PATH\sales_train.csv')
df_test = pd.read_csv(r'YOUR_PATH\test.csv')
df_sample = pd.read_csv(r'YOUR_PATH\sample_submission.csv')


# Remove returns/cancellations
df_train = df_train[df_train['item_cnt_day'] >= 0]
# Identify and remove outliers
plt.figure()
sns.boxplot(x=df_train['item_cnt_day'])
df_train = df_train[df_train['item_cnt_day']<700]
plt.figure()
sns.boxplot(x=df_train['item_cnt_day'])
# Convert str to datetime and sort
df_train['date'] = pd.to_datetime(df_train['date'], format = '%d.%m.%Y')
df_train = df_train.sort_values(['shop_id','date'], ascending = [True,True]).reset_index(drop=True)

# Translate item categories from Russian to English
translator = Translator()
df_itemcategories['item_category_name'] = df_itemcategories['item_category_name'].apply(translator.translate, src='ru', dest='en').apply(getattr, args=('text',))
# Condense item_category_name
accessories = list(range(8))
consoles = list(range(10,18))
games = list(range(18,32))
cards = list(range(32,37))
movies = list(range(37,42))
books = list(range(42,55))
music = list(range(55,61))
gifts = list(range(61,73))
programs = list(range(73,79))
other = [8,9,79,80,81,82,83]
all_categories = [accessories,consoles, games, cards, movies, books, music, gifts, programs, other]
new_category_names = ['accessories', 'consoles', 'games', 'cards', 'movies', 'books', 'music', 'gifts', 'programs', 'other']
df_itemcategories['new_category'] = None
for i in range(len(all_categories)):
    df_itemcategories.loc[df_itemcategories['item_category_id'].isin(all_categories[i]), 'new_category'] = new_category_names[i]
# Join new categories to main dataframe
df_items = pd.merge(df_items, df_itemcategories[['item_category_id','new_category']], on='item_category_id')
df_train = pd.merge(df_train, df_items[['item_id','new_category']], on='item_id')

# Define important parameters
n_shops = df_shops.shape[0]
n_categories = df_itemcategories.shape[0]
n_items = df_items.shape[0]

# Merge duplicate stores: 10 <- 11, 23 -> 24, 39 <- 40
df_train['shop_id'] = df_train['shop_id'].replace({11: 10, 23: 24, 40: 39})
# Translate shop names from Russian to English
#df_shops['shop_name'] = df_shops['shop_name'].apply(translator.translate, src='ru', dest='en').apply(getattr, args=('text',))
# Keep only the city of the shop locations
df_shops['shop_name'] = df_shops['shop_name'].str.replace(r'[^ \nA-Za-z0-9À-ÖØ-öø-ÿЀ-ӿ/]+','')
df_shops['shop_name'] = df_shops['shop_name'].apply(lambda x: x.split()[0])
# Manual adjustments due to abbrev
df_shops['shop_name'].iloc[34:36] = 'Н Новгород'
df_shops['shop_name'].iloc[39:42] = 'Ростов На Дону'
# Retrieve coordinates of shops
geolocator = Nominatim(user_agent = 'jgronovius@gmail.com')
skip_shops = [9,12,55] # Non-address indices: Offsite Trade, Emergency Online Store, Digital Warehouse
lat = []
long = []
for i in range(len(df_shops['shop_name'])):
    if i not in skip_shops:
        cur_loc = geolocator.geocode(df_shops['shop_name'].iloc[i])
        lat.append(cur_loc.latitude), long.append(cur_loc.longitude)
    else:
        lat.append(float('NaN')), long.append(float('NaN'))
df_shops['lat'] = lat
df_shops['long'] = long

# Convert date_block_num int to date
yr1 = '2013'
mth1 = '01'
yr2 = '2015'
mth2 = '11' # stops with the month before
def month_as_date(df, start_year, start_month, end_year, end_month):
    date_interval = pd.date_range(start_year + '-' + start_month, 
                                  end_year + '-' + end_month, freq='M').to_list()
    date_map = {}
    for i in range(34):
        date_map[i] = date_interval[i]
    #df['date'] = df['date_block_num'].map(date_map)
    df = df.replace({'date_block_num': date_map})
    return df

# Aggregate monthly sales of shop-item pairings
df_train['sales']= df_train['item_price'] * df_train['item_cnt_day']
df_sales = df_train.groupby(['date_block_num', 'shop_id','item_id','new_category'],as_index=False).agg({'sales':'sum', 'item_cnt_day':'sum'})
df_sales = pd.merge(df_sales, df_shops[['shop_id','lat','long']], on='shop_id')
df_sales = month_as_date(df_sales, yr1, mth1, yr2, mth2)
# Aggregate total monthly sales
df_total = df_sales.groupby(['date_block_num'],as_index=False).agg({'sales':'sum', 'item_cnt_day':'sum'})
# Aggregate monthly sales per item
df_itemsales = df_sales.groupby(['date_block_num','item_id'],as_index=False).agg({'sales':'sum', 'item_cnt_day':'sum'})
df_itemsales = df_itemsales.sort_values(['item_id','date_block_num'], ascending = [True,True]).reset_index(drop=True)
# Aggregate monthly sales per item category
df_categorysales = df_sales.groupby(['date_block_num','new_category'],as_index=False).agg({'sales':'sum', 'item_cnt_day':'sum'})
df_categorysales = df_categorysales.sort_values(['new_category','date_block_num'], ascending = [True,True]).reset_index(drop=True)
# Aggregate monthly sales of shop-category pairings
df_shopcategorysales = df_sales.groupby(['date_block_num', 'shop_id','new_category'],as_index=False).agg({'sales':'sum', 'item_cnt_day':'sum', 'lat':'max', 'long':'max'})
df_shopcategorysales = df_shopcategorysales.sort_values(['shop_id','new_category','date_block_num'], ascending = [True,True,True]).reset_index(drop=True)
# Aggregate monthly sales per shop
df_shopsales = df_sales.groupby(['date_block_num','shop_id'],as_index=False).agg({'sales':'sum', 'item_cnt_day':'sum', 'lat':'max', 'long':'max'})
df_shopsales = df_shopsales.sort_values(['shop_id','date_block_num'], ascending = [True,True]).reset_index(drop=True)

# ADF Test for Stationarity
# For total monthly sales
adf_sales = adfuller(df_total['sales'].values, autolag=None)
print('ADF Statistic: %f' % adf_sales[0])
print('p-value: %f' % adf_sales[1])
print('Critical Values:')
for key, value in adf_sales[4].items():
	print('\t%s: %.3f' % (key, value))
# For total monthly item count
adf_cnt = adfuller(df_total['item_cnt_day'].values, autolag=None)
print('ADF Statistic: %f' % adf_cnt[0])
print('p-value: %f' % adf_cnt[1])
print('Critical Values:')
for key, value in adf_cnt[4].items():
	print('\t%s: %.3f' % (key, value))
# Per shop
adf_results = []
for i in range(n_shops):
    if i not in [11,23,40]: # these shops were merged
        sales = df_sales.loc[df_sales['shop_id']==i,'sales']
        result = adfuller(sales.values, autolag=None)
        adf_results.append(result)
adf_results = pd.DataFrame(adf_results, columns = ['ADF stat','p-value','used lag','nobs','critical values'])

# =============================================================================
# DATA VISUALIZATION
# Move to separate script later

sns.set(style='whitegrid')
cat_order = ['games','consoles','gifts','movies','accessories','music','cards','programs','books','other']

# Total monthly sales:
plt.figure()
g = sns.lineplot(x='date_block_num', y='sales', data=df_total, 
                  palette=sns.color_palette('Set2'))
g.set(xlabel='Month', ylabel='Monthly Sales (RUB)', title = 'Total Monthly Sales')

# Monthly sales per store:
# Shops with little observations: 0, 1, 8, 9, 11, 20, 23, 36
def plot_shop(df, shop, xaxis, yaxis):
    df = df[df['shop_id']==shop]
    plt.figure()
    g = sns.lineplot(x=xaxis, y=yaxis, data=df,
                 palette=sns.color_palette('Set2'))
    g.legend(['Shop ' + str(i)])
    g.set(xlabel='Month', ylabel='Monthly Sales (RUB)', title = 'Monthly Sales Per Shop ID')
# As separate plots
#for i in range(n_shops):
#    plot_shop(df_shopsales, i, 'date_block_num', 'sales')
# As subplot grid
plt.figure()
g = sns.relplot(x='date_block_num', y='sales', col='shop_id', col_wrap=6, 
                 kind='line', data=df_shopsales)
g.set(xlabel='Month', ylabel='Monthly Sales (RUB)')
g.set_xticklabels(rotation=-45, horizontalalignment = 'left')

# Monthly sales per category:
# As single plot
plt.figure()
g = sns.lineplot(x='date_block_num', y='sales', hue ='new_category', 
                  hue_order = cat_order,
                  data=df_categorysales, palette=sns.color_palette('deep', len(cat_order)))
g.legend(loc='right', bbox_to_anchor=(1.35, 0.5), ncol=1)
plt.setp(g.get_xticklabels(), rotation=-45, horizontalalignment='left')
g.set(xlabel='Month', ylabel='Monthly Sales (RUB)', title = 'Monthly Sales Per Category')
# As subplot grid with all other categories as silhouettes
plt.figure()
g = sns.relplot(
    data=df_categorysales,
    x="date_block_num", y="sales", col="new_category", hue="new_category", hue_order = cat_order,
    kind="line", palette="deep", linewidth=4, col_wrap=5, legend=False)
for new_category, ax in g.axes_dict.items():
    ax.text(.8, .85, new_category, transform=ax.transAxes, fontweight="bold")
    sns.lineplot(
        data=df_categorysales, x="date_block_num", y="sales", units="new_category",
        estimator=None, color=".7", linewidth=1, ax=ax,
    )
g.set_titles("")
g.set(xlabel='Month', ylabel='Monthly Sales (RUB)')
g.set_xticklabels(rotation=-45, horizontalalignment = 'left')
g.tight_layout()

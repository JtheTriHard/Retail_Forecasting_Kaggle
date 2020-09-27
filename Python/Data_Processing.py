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
from statsmodels.tsa.stattools import adfuller
from itertools import compress
from googletrans import Translator

# Confirm working directory
print("Working Directory: ", os.getcwd())

# =============================================================================
# DATA CLEANING
# NOTE: The following repeats some steps that were done thru SQL queries

# Import original Kaggle files
df_items = pd.read_csv(r'YOUR_PATH\items.csv')
df_itemcategories = pd.read_csv(r'YOUR_PATH\item_categories.csv')
df_shops = pd.read_csv(r'YOUR_PATH\shops.csv')
df_train = pd.read_csv(r'YOUR_PATH\sales_train.csv')
df_test = pd.read_csv(r'YOUR_PATH\test.csv')
df_sample = pd.read_csv(r'YOUR_PATH\sample_submission.csv')

# Define important parameters
n_shops = df_shops.shape[0]
n_categories = df_itemcategories.shape[0]
n_items = df_items.shape[0]

# Check for missing values
all_dfs = [df_items, df_itemcategories, df_shops, df_train, df_test, df_sample]

# Translate desc. from Russian to English
translator = Translator()
time_start = time.time()
df_itemcategories['item_category_name'] = df_itemcategories['item_category_name'].apply(translator.translate, src='ru', dest='en').apply(getattr, args=('text',))
time_category = time.time() - time_start

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

# Convert str to datetime and sort
df_train['date'] = pd.to_datetime(df_train['date'], format = '%d.%m.%Y')
df_train = df_train.sort_values(['shop_id','date'], ascending = [True,True]).reset_index(drop=True)

# Aggregate monthly sales of shop-item pairings
df_train['sales']= df_train['item_price'] * df_train['item_cnt_day']
df_sales = df_train.groupby(['date_block_num', 'shop_id','item_id','new_category'],as_index=False).agg({'sales':'sum'})
# Aggregate total monthly sales
df_total = df_sales.groupby(['date_block_num'],as_index=False).agg({'sales':'sum'})
# Aggregate monthly sales per item
df_itemsales = df_sales.groupby(['date_block_num','item_id'],as_index=False).agg({'sales':'sum'})
df_itemsales = df_itemsales.sort_values(['item_id','date_block_num'], ascending = [True,True]).reset_index(drop=True)
# Aggregate monthly sales per item category
df_categorysales = df_sales.groupby(['date_block_num','new_category'],as_index=False).agg({'sales':'sum'})
df_categorysales = df_categorysales.sort_values(['new_category','date_block_num'], ascending = [True,True]).reset_index(drop=True)
# Aggregate monthly sales of shop-category pairings
df_shopcategorysales = df_sales.groupby(['date_block_num', 'shop_id','new_category'],as_index=False).agg({'sales':'sum'})
df_shopcategorysales = df_shopcategorysales.sort_values(['shop_id','new_category','date_block_num'], ascending = [True,True,True]).reset_index(drop=True)
# Aggregate monthly sales per shop
df_shopsales = df_sales.groupby(['date_block_num','shop_id'],as_index=False).agg({'sales':'sum'})
df_shopsales = df_shopsales.sort_values(['shop_id','date_block_num'], ascending = [True,True]).reset_index(drop=True)


# ADF Test for Stationarity
# For total sales (result: not stationary)
result = adfuller(df_total['sales'].values, autolag=None)
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])
print('Critical Values:')
for key, value in result[4].items():
	print('\t%s: %.3f' % (key, value))
# Per shop (result: stationary)
adf_results = []
for i in range(n_shops):
    sales = df_sales.loc[df_sales['shop_id']==i,'sales']
    result = adfuller(sales.values, autolag=None)
    adf_results.append(result)
adf_results = pd.DataFrame(adf_results, columns = ['ADF stat','p-value','used lag','nobs','critical values'])

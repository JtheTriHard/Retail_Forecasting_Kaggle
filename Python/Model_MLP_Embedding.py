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
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler, StandardScaler,RobustScaler

# Confirm working directory
print("Working Directory: ", os.getcwd())

np.random.seed(123)
tf.random.set_seed(123)
df = pd.read_csv(r'YOUR_DIRECTORY')

# =============================================================================
# MODEL-SPECIFIC CLEANING

# Drop unnecessary features
df = df.drop(['item_price','sales','new_category'],axis=1) # cat temp removed
# Merging duplicate shops earlier broke consecutive shop IDs
df['shop_id'], remap = pd.factorize(df['shop_id'])
remap = remap.to_list()
# Retrieve month
df['month'] = pd.DatetimeIndex(df['date']).month
# Map month to unit circle
df['month_x'] = np.sin(2 * np.pi * df['month']/12)
df['month_y'] = np.cos(2 * np.pi * df['month']/12)
df = df.drop(['date','month'],axis=1)
sns.scatterplot(x='month_x', y='month_y',data=df)

# Reserve final 10% of months for validation
threshold = round(df['date_block_num'].max() * 0.9)
df_train = df[df['date_block_num'] <= threshold]
df_train = df_train.drop(['date_block_num'],axis=1)
df_test = df[df['date_block_num'] > threshold]
df_test = df_test.drop(['date_block_num'],axis=1)

# Scale data 
scaler = MinMaxScaler()
df_train['item_cnt_day'] = scaler.fit_transform(df_train[['item_cnt_day']])
df_test['item_cnt_day'] = scaler.transform(df_test[['item_cnt_day']])

# Convert to array
train_x = df_train.drop(['item_cnt_day'],axis=1).values
train_y = df_train['item_cnt_day'].values
test_x = df_test.drop(['item_cnt_day'],axis=1).values
test_y = df_test['item_cnt_day'].values

# =============================================================================
# MODEL

# Features to be embedded
cat_cols = ['shop_id','item_id']
# Features to send as regular numeric input
num_cols = ['month_x','month_y']

# Parameters
n_epochs = 250
batchsize = 16384
embed_dim = 3
n_num_feats = len(num_cols) # num features that are not embedded

tf.keras.backend.clear_session()
# Define embedded features input
inputs, embeddings = [], []
for i in cat_cols:
    cat_input = tf.keras.layers.Input(shape=(1,), name="".join([i.replace(" ", ""),"_inp"]))
    cat_dim  = df[i].max() +1
    inputs.append(cat_input)
    cat_input = tf.keras.layers.Embedding(cat_dim, embed_dim, input_length = 1,
            name="".join([i.replace(" ", ""),"_embed"]))(cat_input)
    cat_input = tf.reshape(cat_input,[-1,embed_dim])
    embeddings.append(cat_input)

# Define numeric features input    
num_input = tf.keras.layers.Input(shape=(n_num_feats), name="num_input")
inputs.append(num_input)
embeddings.append(num_input)

# MLP using Functional API
input_layer = tf.keras.layers.Concatenate(name="concat")(embeddings)
new_layer= tf.keras.layers.Dense(128,activation='tanh',name='MLP_1')(input_layer)
new_layer = tf.keras.layers.Dropout(0.2,name='Dropout_1')(new_layer)
new_layer= tf.keras.layers.Dense(64,activation='tanh',name='MLP_1')(input_layer)
new_layer = tf.keras.layers.Dropout(0.2,name='Dropout_1')(new_layer)
output_layer = tf.keras.layers.Dense(1,name='output_layer')(new_layer)
model = tf.keras.Model(inputs, output_layer, name = "MLP")
model.compile(optimizer='adam',loss='mse')
model.summary()
tf.keras.utils.plot_model(model, 'YOUR_PATH', show_shapes=True)
history = model.fit([train_x[:,0],train_x[:,1],train_x[:,2:]], train_y, batch_size = batchsize, epochs=n_epochs, 
                    validation_data=([test_x[:,0],test_x[:,1],test_x[:,2:]], test_y))

# Predict
test_yhat = model.predict([test_x[:,0],test_x[:,1],test_x[:,2:]])
test_yhat = np.rint(scaler.inverse_transform(test_yhat))

# Define plotting functions
def plot_train_history(history):
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epoch_range = range(len(loss))
    plt.figure()
    plt.plot(epoch_range, loss, label='Training loss')
    plt.plot(epoch_range, val_loss, label='Validation loss')
    #plt.title(title)
    plt.ylabel('MSE (Scaled)')
    plt.xlabel('Epochs')
    plt.legend()
    plt.show()
plot_train_history(history)

# Plot embeddings
# 2 = shop_id, 3 = item_id
def plot_embeddings(layer):
    u=model.layers[layer]
    weights = np.array(u.get_weights())
    categories = weights.shape[1]
    plt.figure()
    ax = plt.axes(projection='3d')
    ax.scatter3D(weights[0,:,0], weights[0,:,1], weights[0,:,2])
    labels = list(range(0,categories))
    for x, y, z, label in zip(weights[0,:,0], weights[0,:,1], weights[0,:,2], labels):
        ax.text(x, y, z, label)
    #for angle in range(0, 360):
        #ax.view_init(30, angle)
        #plt.draw()
        #plt.pause(.001)
plot_embeddings(2)
plot_embeddings(3)

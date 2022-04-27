import tensorflow as tf
from tensorflow import keras

import os

import pandas as pd

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tensorflow.keras.layers import Input, Dense, LeakyReLU, Dropout


df =  pd.read_csv('Train_updated.csv')
print("After reading data from csv shape : ", df.shape)
df['target'] = df['target'].astype(int)

Y = df['target']
X = df.drop(columns=['target'])


X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.1, shuffle=True)
print('X-Train shape', X_train.shape)
print('X-Test shape', X_test.shape)
y_train = tf.convert_to_tensor(y_train)
y_test = tf.convert_to_tensor(y_test)
print('Y-Train shape', y_train.shape)
print('Y-Test shape', y_test.shape)




input_ = Input(shape = (69), name='Input')

dense1_shared = Dense(128, name='dense1_shared')(input_)
lr1_shared = LeakyReLU(alpha=0.01, name='lr1_shared')(dense1_shared)
dense2_shared = Dense(256, name='dense2_shared')(lr1_shared)
lr2_shared = LeakyReLU(alpha=0.01, name='lr2_shared')(dense2_shared)
#drop1_shared = Dropout(0.05, name='drop1_shared')(lr2_shared)
#dense3_shared = Dense(1024, name='dense3_shared')(drop1_shared)
#lr3_shared = LeakyReLU(alpha=0.05, name='lr3_shared')(dense3_shared)
drop2_shared = Dropout(0.02, name='drop2_shared')(lr2_shared)
dense4_shared = Dense(512, name='dense4_shared')(drop2_shared)
lr4_shared = LeakyReLU(alpha=0.01, name='lr4_shared')(dense4_shared)


#dense1_3 = Dense(1024, name='dense1_3')(lr4_shared)
#lr1_3 = LeakyReLU(alpha=0.01, name='lr1_3')(dense1_3)
drop1_3 = Dropout(0.05, name='drop1_3')(lr4_shared)
dense2_3 = Dense(256, name='dense2_3')(drop1_3)
lr2_3 = LeakyReLU(alpha=0.05, name='lr2_3')(dense2_3)
drop2_3 = Dropout(0.02, name='drop2_3')(lr2_3)
dense3_3 = Dense(128, name='dense3_3')(drop2_3)
lr3_3 = LeakyReLU(alpha=0.01, name='lr3_3')(dense3_3)
drop3_3 = Dropout(0.01, name='drop3_3')(lr3_3)
dense3_4 = Dense(64, name='dense3_4')(drop3_3)
out3 = Dense(1, activation = 'sigmoid', name = 'out3')(dense3_4)
#out3 = Dense(2, activation='softmax', name = 'out3')(dense3_4)

model = tf.keras.models.Model(input_, out3)

# model.compile(
#      loss = {
#         'out3': 'binary_crossentropy'
#     },
#     optimizer='adam',
#     metrics = ['accuracy']
# )
model.compile(optimizer='adam', loss=tf.keras.losses.BinaryCrossentropy(),metrics=[tf.keras.metrics.AUC()])

_ = model.fit(
    X_train,
    y_train,
    batch_size = 32,
    epochs=10, 
    validation_data=(X_test, y_test),
    verbose=True)

df_test =  pd.read_csv('Test_updated.csv')
print("After reading test data from csv shape : ", df_test.shape)

predictions = model.predict(df_test)
print("predictions shape:", predictions.shape)

Y_pred = model.predict(X_train)
Y_pred_dis = np.where(Y_pred >= 0.5, 1, 0)
print("Train Report")    
print(classification_report(y_train,Y_pred_dis,digits=5))


Y_pred = model.predict(X_test)
Y_pred_dis = np.where(Y_pred >= 0.5, 1, 0)
print("Test Report")    
print(classification_report(y_test,Y_pred_dis,digits=5))

pd.DataFrame(predictions, columns=['target']).to_csv('predictions.csv', index=False)



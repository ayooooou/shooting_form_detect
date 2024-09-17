import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from tensorflow import keras
from keras import layers
import ast
import tkinter.messagebox as messagebox

ordinal_encoder = OrdinalEncoder()

#補空資料
def pad_list(lst, max_length):
    return lst + [0] * (max_length - len(lst))

#字串轉列表
def convert_string_to_list(df, column_names):
    for col in column_names:
        df[col] = df[col].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
    return df

def training():
    # 讀取資料
    train_df = pd.read_csv(r"opencv\shooting_detect\train_data\train.csv")
    
    # 將 motion 欄位進行編碼
    train_df[['motion']] = ordinal_encoder.fit_transform(train_df[['motion']])
    
    # 需要轉換為列表的欄位
    list_columns = ['Relbow_list', 'Rshoulder_list', 'Rbody_list', 'Rknee_list']
    
    # 將字串轉換為列表
    train_df = convert_string_to_list(train_df, list_columns)
    
    # 確保所有特徵列表長度一致
    max_length_train = max(train_df['Relbow_list'].apply(len).max(),
                           train_df['Rshoulder_list'].apply(len).max(),
                           train_df['Rbody_list'].apply(len).max(),
                           train_df['Rknee_list'].apply(len).max())
    
    # 使用 pad_list 將所有列表補齊到相同長度
    for col in list_columns:
        train_df[col] = train_df[col].apply(lambda x: pad_list(x, max_length_train))
    
    # 合併所有特徵列表
    features_train = np.concatenate([
        np.array(train_df['Relbow_list'].tolist()),
        np.array(train_df['Rshoulder_list'].tolist()),
        np.array(train_df['Rbody_list'].tolist()),
        np.array(train_df['Rknee_list'].tolist())
    ], axis=1)
    
    labels_train = train_df['motion'].values
    
    # 拆分訓練集和驗證集
    x_train, x_val, y_train, y_val = train_test_split(features_train, labels_train, test_size=0.2, random_state=42)
    
    
    model = keras.Sequential([
        layers.Dense(128, activation="relu", input_shape=(360,)),
        layers.Dropout(0.4),
        layers.Dense(64, activation="relu"),
        layers.Dropout(0.4),
        layers.Dense(32, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(16, activation="relu"),
        layers.Dense(1, activation="sigmoid")
    ])

    model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

    history = model.fit(
        x_train,
        y_train,
        epochs=200,
        batch_size=64,
        validation_data=(x_val, y_val),
    )       
# training()



def outputmotion():
    # 讀取資料
    train_df = pd.read_csv(r"opencv\shooting_detect\train_data\train.csv")
    test_df = pd.read_csv(r"opencv\shooting_detect\train_data\test.csv")
    
    # 需要轉換為列表的欄位
    list_columns = ['Relbow_list', 'Rshoulder_list', 'Rbody_list', 'Rknee_list']
    
    # 將字串轉換為列表
    train_df = convert_string_to_list(train_df, list_columns)
    
    # 確保所有特徵列表長度一致
    max_length_train = max(train_df['Relbow_list'].apply(len).max(),
                           train_df['Rshoulder_list'].apply(len).max(),
                           train_df['Rbody_list'].apply(len).max(),
                           train_df['Rknee_list'].apply(len).max())
    
    # 將字串轉換為列表
    test_df = convert_string_to_list(test_df, list_columns)
    
    # 使用 pad_list 將所有列表補齊到相同長度
    for col in list_columns:
        test_df[col] = test_df[col].apply(lambda x: pad_list(x, max_length_train))
    
    # 合併所有特徵列表
    features_test = np.concatenate([
        np.array(test_df['Relbow_list'].tolist()),
        np.array(test_df['Rshoulder_list'].tolist()),
        np.array(test_df['Rbody_list'].tolist()),
        np.array(test_df['Rknee_list'].tolist())
    ], axis=1)
    
    # 取得最後一行的特徵資料
    last_feature = features_test[-1].reshape(1, -1)  # 轉為 2D 資料 (1, n_features)
    
    # 預測
    from keras.models import load_model
    model = load_model('opencv\shooting_detect\my_model.keras')
    
    # 進行預測並輸出結果
    last_prediction = model.predict(last_feature)
    
    messagebox.showinfo("Prediction Result", f"Last row prediction: {last_prediction}")
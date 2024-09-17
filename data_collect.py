import numpy
import pandas as pd

def record(self, motion):
    train_df = pd.read_csv(r"opencv\shooting_detect\train_data\train.csv")
    test_df = pd.read_csv(r"opencv\shooting_detect\train_data\test.csv")
    
    if motion == "train":
        train_df.loc[len(train_df)] = [len(train_df), motion, self.Relbow_list, self.Rshoulder_list, self.Rbody_list, self.Rknee_list]
        train_df.to_csv(r"opencv\shooting_detect\train_data\train.csv", index=False)  # 將更改寫入到 CSV 檔案中
    if motion == "record":
        test_df.loc[len(test_df)] = [len(train_df),self.Relbow_list, self.Rshoulder_list, self.Rbody_list, self.Rknee_list]
        test_df.to_csv(r"opencv\shooting_detect\train_data\train.csv", index=False)  # 將更改寫入到 CSV 檔案中
# Python-Machine Learning-XGboost-Digits
請設計類神經網路+XGBoost分類器以分類0~9手寫數字


0到9手寫數字的二維影像轉成一維向量當成隨機(W,b)的類神經網路的輸入，f 非線性轉移函數自定，此類神經網路可得到對應此輸入的一維向量輸出(隨機特徵)a，將此隨機特徵當成XGBoost的輸入而其期望輸出就是手寫數字所對應的數字，所以XGBoost的期望輸出及XGBoost的輸出只有10類(0到9)可能。

1.用sklearn.datasets的digits去拿手寫辨識的資料庫

2.用xgboost的DMatrix來轉換成二進位並設定參數

3.儲存model

4.然後去print輸出xgboost的0~9和訓練的準確度。

# create_model.py
import numpy as np
import cloudpickle
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# 生成隨機資料
X = np.random.rand(100, 4)  # 100 筆資料，4 個特徵
y = np.random.randint(0, 2, size=100)  # 0/1 兩類

# 建立 pipeline
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression())
])

# 訓練模型
pipeline.fit(X, y)

# 儲存模型
with open("model.pkl", "wb") as f:
    cloudpickle.dump(pipeline, f)
print("model.pkl saved.")

import requests
import numpy as np
import json

# Triton inference API endpoint
url = "http://localhost:8187/v2/models/sklearn_pipeline/infer"

# 建立隨機測試資料
input_data = np.random.rand(1, 4).astype(np.float32).tolist()

# 建立 payload
payload = {
    "inputs": [
        {
            "name": "input__0",
            "shape": [1, 4],
            "datatype": "FP32",
            "data": input_data
        }
    ],
    "outputs": [
        {
            "name": "output__0"
        }
    ]
}

# 發送請求
response = requests.post(url, json=payload)

# 顯示結果
print("Status Code:", response.status_code)
print("Response:", response.json())

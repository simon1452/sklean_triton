import numpy as np
import tritonclient.http as httpclient

client = httpclient.InferenceServerClient(url="localhost:8187")

# 隨機一筆資料 (4 維)
input_data = np.random.rand(1, 4).astype(np.float32)

inputs = httpclient.InferInput("input__0", input_data.shape, "FP32")
inputs.set_data_from_numpy(input_data)

outputs = httpclient.InferRequestedOutput("output__0")

response = client.infer(model_name="sklearn_pipeline",
                        inputs=[inputs],
                        outputs=[outputs])

print("Prediction:", response.as_numpy("output__0"))

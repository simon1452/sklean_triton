import os
import cloudpickle
import joblib
import numpy as np
import triton_python_backend_utils as pb_utils

class TritonPythonModel:
    def initialize(self, args):
        # 找到 pkl 的路徑
        model_dir = args["model_repository"]  # Triton model repo 根目錄
        version = args["model_version"]
        model_path = os.path.join(model_dir, version, "model.pkl")

        # 載入 sklearn pipeline
        with open(model_path, 'rb') as f:
            self.model = cloudpickle.load(f)

    def execute(self, requests):
        responses = []
        for request in requests:
            # 取得輸入
            in_tensor = pb_utils.get_input_tensor_by_name(request, "input__0")
            input_data = in_tensor.as_numpy()

            # 預測
            preds = self.model.predict(input_data).astype(np.int32)

            # 包裝輸出
            out_tensor = pb_utils.Tensor("output__0", preds)
            responses.append(pb_utils.InferenceResponse([out_tensor]))
        return responses

    def finalize(self):
        print("Cleaning up...")
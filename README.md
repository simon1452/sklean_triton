conda activate time_series_model_3.9

docker run --rm --gpus all \
  -p8187:8187 -p8189:8189 -p8200:8200 \
  -v /home/develop/DMLOps/sklearn_triton/:/models \
  tritonserver:23.10-sklearn \
  tritonserver --model-repository=/models \
               --http-port=8187 \
               --grpc-port=8189 \
               --metrics-port=8200
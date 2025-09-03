Define Triton images

1\. Dockerfile

FROM registry.digiwincloud.com.cn/teamcloud/tritonserver:23.10-py3



COPY /requirements.txt /tmp/requirements.txt

\##RUN pip install --no-cache-dir -r /tmp/requirements.txtcond



RUN pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple \&\& \\

&nbsp;   pip install --no-cache-dir --ignore-installed --default-timeout=600 -r /tmp/requirements.txt



2\. docker build -t tritonserver:23.10-sklearn .

3\. conda activate time\_series\_model\_3.9

4\. docker run --rm --gpus all \\

&nbsp; -p8131:8131 -p8132:8132 -p8133:8133 \\

&nbsp; -v /home/develop/DMLOps/save\_data\_sm/deploy\_model/triton\_server/TSA/Time-Series-Forecasting:/models \\

&nbsp; tritonserver:23.10-sklearn \\

&nbsp; tritonserver --model-repository=/models \\

&nbsp;              --http-port=8131 \\

&nbsp;              --grpc-port=8132 \\

&nbsp;              --metrics-port=8133



conda activate time\_series\_model\_3.9



docker run --rm --gpus all   
-p8187:8187 -p8189:8189 -p8200:8200   
-v /home/develop/DMLOps/sklearn\_triton/:/models   
tritonserver:23.10-sklearn   
tritonserver --model-repository=/models   
--http-port=8187   
--grpc-port=8189   
--metrics-port=8200


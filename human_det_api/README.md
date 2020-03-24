### Human detection API

整个code 依赖pytorch1.2 可以利用anaconda进行安装也可以利用docker，这个docker是我训练采用的环境 `zhichenghuang/azureml-docker:latest`

```shell
# 从GitHub上将code clone
git clone https://github.com/ZhichengHuang/mmdetection.git
cd mmdetection
#初始化
bash human_det_api/begin.sh
```

API包装在`human_det_api`文件夹下，只要实例化`Human_Detector_API`类就可以了，其中score_thr是分类的阈值，默认0.75 ，config_file是整个模型的配置文件，保持默认就可以，checkpoint是模型的参数文件，在使用begin.sh的时候会自动下载，保持默认就可以。使用的时候传入一张图的地址，然后返回符合条件的框的坐标以及执行度。
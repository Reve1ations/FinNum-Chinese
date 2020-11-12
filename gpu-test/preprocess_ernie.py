import yaml
from data_process.data_process import data_process  # 导入data_process文件目录下的data_process.py文件中的data_process包

config = yaml.safe_load(open('config.yml', encoding='UTF-8'))

data_process(config)

##字段说明
	ID：唯一标识
	Label：类别标签，0：人工编写  1：大语言模型生成
	Content：数据内容
## （1）先将测试数据集拷贝到该文件的train_data中
## （2）docker build -t testmodel .
## （3）docker run -it -v 文件train_data的地址:/app/train_data --entrypoint /bin/bash testmodel
## （4）python /app/test.py --file_path /app/train_data/数据集名称
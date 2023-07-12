import json

# 打开源文件和目标文件
with open("raw/ultrachat_release_230407.json", "r") as source_file, open("processed/output.txt", "w") as target_file:
    # 逐行读取源文件
    for line in source_file:
        # 解析JSON数据
        data = json.loads(line)
        sentences = data["data"]

        # 将每个句子写入目标文件
        for sentence in sentences:
            target_file.write(sentence + "\n")

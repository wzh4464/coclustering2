import json
import multiprocessing

def process_line(line):
    result = []
    data = json.loads(line)
    sentences = data["data"]

    # 将每个句子写入目标文件
    for sentence_group in sentences:
        # sparse the sentence_group with "."
        for sentence in sentence_group.split("."):
            # if len(sentence) > 0:
            if len(sentence) > 0:
                # if not end with number then write to file
                if not sentence[-1].isdigit():
                    # remove the space at the beginning of the sentence
                    sentence = sentence.strip()
                    result.append(sentence)
    return result

def process_json_file(source_file_path, target_file_path=None):
    pool = multiprocessing.Pool()
    results = []

    # 打开源文件和目标文件
    with open(source_file_path, "r") as source_file, open(target_file_path, "w") as target_file:
        # 逐行读取源文件
        for result in pool.imap_unordered(process_line, source_file):
            results.extend(result)
            if target_file_path is not None:
                for sentence in result:
                    target_file.write(sentence + "\n")

    pool.close()
    pool.join()

    return results

if __name__ == "__main__":
    result = process_json_file("raw/ultrachat_release_230407.json")
    # print first 10 sentences
    for i in range(10):
        print(result[i])
        
        
# import json

# def process_json_file(source_file_path, target_file_path):
#     result = []

#     # 打开源文件和目标文件
#     with open(source_file_path, "r") as source_file, open(target_file_path, "w") as target_file:
#         # 逐行读取源文件
#         for line in source_file:
#             # 解析JSON数据
#             data = json.loads(line)
#             sentences = data["data"]

#             # 将每个句子写入目标文件
#             for sentence_group in sentences:
#                 # sparse the sentence_group with "."
#                 for sentence in sentence_group.split("."):
#                     # if len(sentence) > 0:
#                     if len(sentence) > 0:
#                         # if not end with number then write to file
#                         if not sentence[-1].isdigit():
#                             # remove the space at the beginning of the sentence
#                             sentence = sentence.strip()
#                             target_file.write(sentence + "\n")
#                             # print(sentence)
#                             result.append(sentence)
#     return result
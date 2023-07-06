import xml.etree.ElementTree as ET

# 解析 XML 文件
tree = ET.parse('微博情感分析评测/样例数据/labelled dataset/ipad.xml')
root = tree.getroot()

# 遍历微博元素
weibos = []
for weibo in root.findall('weibo'):
    weibo_id = weibo.get('id')
    sentences = []
    # 遍历句子元素
    for sentence in weibo.findall('sentence'):
        sentence_id = sentence.get('id')
        opinionated = sentence.get('opinionated')
        polarity = sentence.get('polarity')
        target_word_1 = sentence.get('target_word_1')
        target_begin_1 = sentence.get('target_begin_1')
        target_end_1 = sentence.get('target_end_1')
        target_polarity_1 = sentence.get('target_polarity_1')
        text = sentence.text.strip()
        # 将句子信息存储到字典中
        sentence_dict = {
            'id': sentence_id,
            'opinionated': opinionated,
            'polarity': polarity,
            'target_word_1': target_word_1,
            'target_begin_1': target_begin_1,
            'target_end_1': target_end_1,
            'target_polarity_1': target_polarity_1,
            'text': text
        }
        sentences.append(sentence_dict)
    # 将微博信息存储到字典中
    weibo_dict = {
        'id': weibo_id,
        'sentences': sentences
    }
    weibos.append(weibo_dict)

# 打印微博列表
print(weibos)
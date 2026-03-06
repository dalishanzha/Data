import networkx as nx
import json
# Fix for AttributeError: module 'networkx' has no attribute 'from_numpy_matrix'
# This occurs with networkx 3.0+ because textrank4zh uses deprecated methods.
if not hasattr(nx, 'from_numpy_matrix'):
    nx.from_numpy_matrix = nx.from_numpy_array

from textrank4zh import TextRank4Sentence, TextRank4Keyword


def compress_with_tr4zh(text, num_sentences=3, num_keywords=5):
    """
    Compresses Chinese text using the TextRank4ZH library to extract
    both key sentences (summary) and keywords.
    """
    # 1. Sentence Extraction (Text Compression)
    tr4s = TextRank4Sentence()
    tr4s.analyze(text=text, lower=True, source='all_filters')

    # Get the most important sentences
    # limit: number of sentences to return
    # weight_min: minimum weight threshold
    key_sentences = tr4s.get_key_sentences(num=num_sentences)

    summary = "".join([item.sentence for item in key_sentences])

    # 2. Keyword Extraction (Tag-style Compression)
    tr4w = TextRank4Keyword()
    tr4w.analyze(text=text, lower=True, window=2)

    # Get keywords
    keywords_list = tr4w.get_keywords(num=num_keywords, word_min_len=2)
    keywords = "/".join([item.word for item in keywords_list])

    return summary, keywords


# --- Main Execution Example ---
if __name__ == "__main__":

    '''
    #data = json.load(open("./data/conf/ori.json", 'r', encoding="utf-8"))
    data = json.load(open("./data/conf/compress.json", 'r', encoding="utf-8"))
    results = []
    for element in data:
        temp = {}
        idx = element["idx"]
        example_text = element["ori_text"]
        com_text_nospo = element["com_text_nospo"]
        com_text_spo_gpt = element["com_text_spo_gpt"]
        com_text_spo_santa = element["com_text_spo_santa"]
        spo_list_gpt = element["spo_list_gpt"]
        spo_list_santa = element["spo_list_santa"]
        print("--- Original Text ---")
        print(example_text.strip())
        
        # Perform compression
        summary, tags = compress_with_tr4zh(example_text, num_sentences=2, num_keywords=5)

        print("\n--- Compressed Summary (Key Sentences) ---")
        print(summary)

        print("\n--- Compressed Tags (Keywords) ---")
        print(tags)
        temp['idx'] = idx
        temp['ori_text'] = example_text
        temp['com_text_textrank'] = summary
        temp['com_text_nospo'] = com_text_nospo
        temp['com_text_spo_gpt'] = com_text_spo_gpt
        temp['com_text_spo_santa'] = com_text_spo_santa
        temp['spo_list_gpt'] = spo_list_gpt
        temp['spo_list_santa'] = spo_list_santa
        results.append(temp)

    with open(f"./data/conf/final_compress", "w", encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    '''

    example_text = '当前，基于人工智能算法的故障诊断框架已在故障诊断领域占据了重要地位。相较于传统故障诊断方法，AI算法可以处理来自多个传感器、数据源的信息，并且通过历史数据的不断增加与新数据的变化来自适应地调整模型参数和诊断方法。文献［10］搭建了堆叠自编码网络，提取风电机组发电机在正常状态下的数据采集与监控数据特征，并将网络输入数据与重构数据之间的误差值构成状态观测向量实现对发电机故障的判定。文献［11］从数据角度出发，使用卷积神经网络与门控循环神经网络对风电机组发电机正常状态下的数据采集与监控多特征进行提取，并对不平衡数据赋予不同的权重，提升了诊断准确率。'
    # Perform compression
    summary, tags = compress_with_tr4zh(example_text, num_sentences=2, num_keywords=5)

    print("\n--- Compressed Summary (Key Sentences) ---")
    print(summary)

    print("\n--- Compressed Tags (Keywords) ---")
    print(tags)



# -*- coding: utf-8 -*-
"""
Created on Mon Oct 29 12:25:50 2018

@author: xyz
"""

MODELDIR='ltp_data_v3.4.0'
ltp = ltp_api(MODELDIR)
sentences = SentenceSplitter.split(paragraph)

for sentence in sentences:
    if sentence:
        print('\n===================== 原句 =====================\n')
        print(sentence)
        # 第一种：类内计算
        words = ltp.ltp_segmentor(sentence)  # 分词
        postags = ltp.ltp_postagger(words)  # 词性
        arcs = ltp.ltp_parser(words,postags)  #依存
        netags = ltp.ltp_recognizer(words,postags)# 命名实体识别
        labeller = ltp.ltp_labeller(words,postags, arcs) #语义角色

        #ltp.get_result(sentence)
        #output = ltp.output
        #arcs = output['arcs']
        #netags = output['netags']
        #postags = output['postags']
        #labeller = output['role']
        #words = output['words']
        tuples_words = Parser2dataframe(words,postags,arcs)

        print('\n----- 搭配用语查找 -----\n')
        print(FindCollocation(tuples_words))
        print('\n----- 并列词查找 -----\n')
        print(FindSynonym(tuples_words))
        print('\n----- 核心观点抽取 -----\n')
        print(CoreExtraction(tuples_words, words))
        print('\n----- 实体名词搭配 -----\n')
        print(FindEntityCollocation(tuples_words))

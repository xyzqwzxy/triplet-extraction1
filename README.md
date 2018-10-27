paragraph = """环境很好，位置独立性很强，比较安静很切合店名，半闲居，偷得半日闲。点了比较经典的菜品，味道果然不错！烤乳鸽，
超级赞赞赞，脆皮焦香，肉质细嫩，超好吃。艇仔粥料很足，香葱自己添加，很贴心。金钱肚味道不错，不过没有在广州吃的烂，牙口不好的慎点。
凤爪很火候很好，推荐。最惊艳的是长寿菜，菜料十足，很新鲜，清淡又不乏味道，而且没有添加调料的味道，搭配的非常不错！"""
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
        print(CoreExtraction(tuples_words,words))
        print('\n----- 实体名词搭配 -----\n')
        print(FindEntityCollocation(tuples_words))

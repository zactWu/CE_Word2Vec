from gensim.models import KeyedVectors

wiki_file = "./diff_models/wiki_w2v_model.bin"
ce_file = "./diff_models/added_w2v_model.bin"

# model = Word2Vec.load(file)
# model.wv.save_word2vec_format("./models/wiki_w2v_model.bin")

# print("wiki语料预测结果：")
# model = KeyedVectors.load_word2vec_format(wiki_file, limit=500000)
# similarity_1 = model.most_similar("隧道")
# similarity_2 = model.most_similar('金属')
# li = ["金属", "合金", "隧道"]
# print("与隧道最相似的词为：", similarity_1)
# print("与金属最相似的词为：", similarity_2)
# print("金属, 合金, 隧道中差别最大的词为：", model.doesnt_match(li))
#
#
# print("土木语料预测结果：")
# model = KeyedVectors.load_word2vec_format(ce_file, limit=500000)
# similarity_1 = model.most_similar("隧道")
# similarity_2 = model.most_similar('金属')
# li = ["金属", "合金", "隧道"]
# print("与隧道最相似的词为：", similarity_1)
# print("与金属最相似的词为：", similarity_2)
# print("金属, 合金, 隧道中差别最大的词为：", model.doesnt_match(li))

model_ce = KeyedVectors.load_word2vec_format(ce_file)
model_wk = KeyedVectors.load_word2vec_format(wiki_file)
var_ce = len(model_ce.vocab)
var_wk = len(model_wk.vocab)
var = var_ce - var_wk
var_vs = model_ce.vector_size
var_vs2 = model_wk.vector_size
print("wiki语料对应模型词向量个数为：", var_wk)
print("添加了土木语料对应模型词向量个数为：", var_ce)
print("共添加词向量：", var)
print("词向量维度为：", var_vs, " 和", var_vs2)
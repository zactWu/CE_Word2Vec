from gensim.models import KeyedVectors, Word2Vec

file = "./models/wiki_w2v_model.bin"

# model = Word2Vec.load(file)
# model.wv.save_word2vec_format("./models/wiki_w2v_model.bin")


model = KeyedVectors.load_word2vec_format(file, limit=500000)
similarity_1 = model.most_similar('隧道')
similarity_2 = model.most_similar('金属')
li = ["金属", "合金", "隧道"]
print("与隧道最相似的词为：", similarity_1)
print("与金属最相似的词为：", similarity_2)
print("金属, 合金, 隧道中差别最大的词为：", model.doesnt_match(li))

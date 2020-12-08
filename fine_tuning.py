import logging
import os
import sys

from gensim.models import Word2Vec, KeyedVectors, word2vec

# pre_trained_file为基于百度百科语料的预训练词向量
file = "./models/added_w2v_model.model"
pre_trained_file = "sgns.target.word-word.dynwin5.thr10.neg5.dim300.iter5"
path_corpus = "./txt_out"
sentences = word2vec.PathLineSentences(path_corpus)

# 转换预训练词向量为gensim model
# model = KeyedVectors.load_word2vec_format(pre_trained_file, encoding="utf-8", limit=500000)
# model.init_sims(replace=True)
# model.save('./models/pre_trained_word.bin')

# 系统日志
# program = os.path.basename(sys.argv[0])
# logger = logging.getLogger(program)
# logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
# logging.root.setLevel(level=logging.INFO)
# logger.info("running %s" % ' '.join(sys.argv))

# 增量训练
model = Word2Vec.load(file)
model.wv.save_word2vec_format("./models/test.bin")
# model = KeyedVectors.load(file)
print(model.most_similar("同济大学"))
# model.build_vocab(sentences, update=True)
# model.train(sentences, total_examples=model.corpus_count, epochs=model.iter)
# model.save("incremental_w2v_model.model")

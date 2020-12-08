# -*- coding: utf-8 -*-
import multiprocessing
import logging
import os
import sys
from gensim.models import Word2Vec, word2vec

path_corpus = "./txt_out"
wiki_file = "./txt_out/#wiki_sentences.txt"

# 训练模型
program = os.path.basename(sys.argv[0])
logger = logging.getLogger(program)
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
logging.root.setLevel(level=logging.INFO)
logger.info("running %s" % ' '.join(sys.argv))

# sentences = word2vec.PathLineSentences(path_corpus)
sentences = word2vec.LineSentence(wiki_file)
model = Word2Vec(sentences, min_count=5, window=5, size=256,
                 workers=multiprocessing.cpu_count(), iter=10,
                 sg=1, )
# 保存模型要用Word2Vec！！！
model.save("./models/wiki_w2v_model.model")

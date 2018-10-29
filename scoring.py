import csv
import os
import sys
import MeCab
from collections import Counter
from gensim.models.doc2vec import LabeledSentence


TOPIC_DIR = './topic/'
INPUT_DIR = '../train0'
TOPIC_LIST = ['agreement', 'company', 'money', 'position', 'public', 'research']


# 全てのテーマファイルのリストを取得
def get_all_files(directory):
    for root, dirs, files in os.walk(directory):
        for file in files:
            yield os.path.join(root, file)

def read_document(path, encoding):
    with open(path, 'r', encoding=encoding, errors='ignore') as f:
        return f.read()

def corpus_to_sentences(corpus, encoding):
    docs = [read_document(x, encoding) for x in corpus]
    for idx, (doc, name) in enumerate(zip(docs, corpus)):
        yield split_into_words(doc, name)

def trim_doc(doc):
    lines = doc.splitlines()
    valid_lines = []
    for line in lines:
        if line == '':
            continue
        if line.startswith('<doc') or line.startswith('</doc'):
            continue
        if "colspan" in line or "|||||" in line:
            continue
        if '＠' in line:
            continue
        if line.startswith('％'):
            continue
        if line.startswith('F'):
            line = line[5:]
        if line.startswith('＃'):
            line = line[1:]
        if line.startswith('M'):
            line = line[5:]
        valid_lines.append(line)

    return ''.join(valid_lines)

def split_into_words(doc, name=''):
    mecab = MeCab.Tagger("-Ochasen")
    valid_doc = trim_doc(doc)
    lines = mecab.parse(valid_doc).splitlines()

    words = []
    for line in lines:
        chunks = line.split('\t')
        if len(chunks) > 3 and (chunks[3].startswith('動詞') or chunks[3].startswith('形容詞') or (
                chunks[3].startswith('名詞') and not chunks[3].startswith('名詞-数'))):
            words.append(chunks[0])

    return LabeledSentence(words=words, tags=[name])

# ファイルから辞書を作成
def make_topic_dic(path):
    file_name = os.path.basename(path)
    file, ext = os.path.splitext(file_name)
    with open(path, 'r') as f:
        reader = csv.reader(f)
        words = {row[0]: row[1] for row in reader}
    return (file, words)

def make_whole_topic_dic(TOPIC_DIR):
    files = list(get_all_files(TOPIC_DIR))
    topics = {make_topic_dic(file)[0]: make_topic_dic(file)[1] for file in files}
    return topics

def calc_topic_scores(sentences):
    sentences_scores = {}
    for sentence in sentences:
        counter = Counter()
        for word in sentence.words:
            counter[word] += 1
        scores = {}
        #topic score
        for topic in TOPIC_LIST:
            topic_sc = 0
            for key in counter.keys():
                if key in topics[topic].keys():
                    topic_sc += float(topics[topic][key]) * int(counter[key])
            scores[topic] = topic_sc
        sentences_scores[sentence.tags[0]] = scores
    return sentences_scores

if __name__ == "__main__":

    topics = make_whole_topic_dic(TOPIC_DIR)

    corpus = list(get_all_files(INPUT_DIR))
    sentences = list(corpus_to_sentences(corpus, 'utf-8'))

    sentence_scores = calc_topic_scores(sentences)

    for sentence in sentence_scores.keys():
        print(sentence, '-->', sentence_scores[sentence])


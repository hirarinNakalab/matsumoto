import csv
import os
import MeCab
import numpy as np
from scipy import interp
from collections import Counter
from gensim.models.doc2vec import LabeledSentence
from sklearn import metrics
import matplotlib.pyplot as plt

TOPIC_DIR = './topic/'
HOUHAN_DIR = '../houhan'
NUCC_DIR = '../nucc'
TOPIC_LIST = ['agreement', 'company', 'money', 'place', 'official', 'investigation']


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

def calc_topic_scores(sentences, topics):
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

        judging_sc = {}
        #houhan score
        judging_sc['houhan_sc'] = float(scores['money']) * float(scores['agreement'])
        #no purpose score
        judging_sc['nopurpose_sc'] = float(scores['company']) * (float(scores['place']) + float(scores['investigation']))
        #tell lie score
        judging_sc['telllie_sc'] = float(scores['official']) * (float(scores['place']) + float(scores['investigation']))

        sentences_scores[sentence.tags[0]] = judging_sc
    return sentences_scores

def validation_roc_curve(times, topics):
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)

    j = 0
    for i in range(times):
        houhan_labels = []
        houhan_values = []
        INPUT_DIR = '../test{:d}'.format(i)

        corpus = list(get_all_files(INPUT_DIR)) + list(get_all_files(NUCC_DIR))
        sentences = list(corpus_to_sentences(corpus, 'utf-8'))

        sentence_scores = calc_topic_scores(sentences, topics)

        for sentence in sentence_scores.keys():
            sent_dic = sentence_scores[sentence]
            keys = sent_dic.keys()
            if 'sample' in sentence:
                label = 1
            else:
                label = 2
            houhan_labels.append(label)
            houhan_values.append(float(sent_dic['houhan_sc']))
            print('{}   \t\t> '.format(sentence), end='\t')
            for key in keys:
                print('{}:{:.1f}'.format(key, float(sent_dic[key])), end='\t')
            print()

        y = np.array(houhan_labels)
        scores = np.array(houhan_values)

        fpr, tpr, thresholds = metrics.roc_curve(y, scores, pos_label=1, drop_intermediate=False)
        tprs.append(interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
        auc = metrics.auc(fpr, tpr)
        aucs.append(auc)

        plt.plot(fpr, tpr, lw=1, alpha=0.3,
                 label='ROC fold %d (AUC = %0.2f)' % (j, auc))
        print('\n----------------------------------------------\n')
        j += 1

    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
             label='Luck', alpha=.8)

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = metrics.auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    plt.plot(mean_fpr, mean_tpr, color='b',
             label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
             lw=2, alpha=.8)
    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                     label=r'$\pm$ 1 std. dev.')

    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic Curve')
    plt.legend(loc="lower right")

    plt.savefig('valid_roc.png')

def define_risk_class_list():
    HIGH = [3, 4, 5, 6, 19, 20, 21]
    MID = [1, 2, 7, 8, 22, 23, 24]
    LOW = [9, 10, 11, 12, 25, 26, 27]

    H_list = []
    M_list = []
    L_list = []
    for i in HIGH:
        H_list.append("sample{:03d}".format(int(i)))
    for i in MID:
        M_list.append("sample{:03d}".format(int(i)))
    for i in LOW:
        L_list.append("sample{:03d}".format(int(i)))

    H_list += ["sample_B", "sample_K", "sample_L"]
    M_list += ["sample_A", "sample_C", "sample_J"]
    L_list += ["sample_G", "sample_H", "sample_I"]
    return H_list, M_list, L_list

def middle_determination_plot(topics):

    _, M_list, _ = define_risk_class_list()

    labels = []
    values = []

    corpus = list(get_all_files(HOUHAN_DIR))
    sentences = list(corpus_to_sentences(corpus, 'utf-8'))

    sentence_scores = calc_topic_scores(sentences, topics)

    for sentence in sentence_scores.keys():
        print(sentence)
        sent_dic = sentence_scores[sentence]
        label = 1
        for m in M_list:
            if m in sentence:
                label = 2
        labels.append(label)
        values.append(float(sent_dic['nopurpose_sc']))

    y = np.array(labels)
    scores = np.array(values)

    fpr, tpr, thresholds = metrics.roc_curve(y, scores, pos_label=2, drop_intermediate=False)
    auc = metrics.auc(fpr, tpr)

    plt.plot(fpr, tpr, label='binary classification (area = %.2f)' % auc)
    plt.legend()
    plt.title('Sales Visit Detection - Receiver Operating Characteristic')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.grid(True)
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Luck', alpha=.8)
    plt.savefig('nopurpose.png')

def high_determination_plot(topics):

    H_list, _, _ = define_risk_class_list()

    labels = []
    values = []

    corpus = list(get_all_files(HOUHAN_DIR))
    sentences = list(corpus_to_sentences(corpus, 'utf-8'))

    sentence_scores = calc_topic_scores(sentences, topics)

    for sentence in sentence_scores.keys():
        sent_dic = sentence_scores[sentence]
        label = 1
        for h in H_list:
            if h in sentence:
                label = 2
        labels.append(label)
        values.append(float(sent_dic['telllie_sc']))

    y = np.array(labels)
    scores = np.array(values)

    fpr, tpr, thresholds = metrics.roc_curve(y, scores, pos_label=2, drop_intermediate=False)
    auc = metrics.auc(fpr, tpr)

    plt.plot(fpr, tpr, label='binary classification (area = %.2f)' % auc)
    plt.legend()
    plt.title('Sales Visit Detection - Receiver Operating Characteristic')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.grid(True)
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Luck', alpha=.8)
    plt.savefig('telllie.png')


if __name__ == "__main__":
    topics = make_whole_topic_dic(TOPIC_DIR)
    print("validation")
    validation_roc_curve(3, topics)
    print("middle")
    middle_determination_plot(topics)
    print("high")
    high_determination_plot(topics)


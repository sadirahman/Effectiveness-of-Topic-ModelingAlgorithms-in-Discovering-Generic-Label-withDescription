from nltk.tokenize import RegexpTokenizer
from stop_words import get_stop_words
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
from gensim import corpora, models
import nltk
from nltk.corpus import wordnet
import gensim
import pyLDAvis.gensim as gensimvis
import pyLDAvis
import warnings

warnings.filterwarnings('ignore')

tokenizer = RegexpTokenizer(r'\w+')

# create English stop words list
en_stop = get_stop_words('en')

# Create p_stemmer of class PorterStemmer
p_stemmer = PorterStemmer()
l_lemma = WordNetLemmatizer()

# create sample documents
# doc_a = "a motor vehicle with four wheels; usually propelled by an internal combustion engine"
# doc_b = "a wheeled vehicle adapted to the rails of railroad"
# doc_c = "the compartment that is suspended from an airship and that carries personnel and the cargo and the power plant"
# doc_d = "where passengers ride up and down"
# doc_e = "a conveyance for passengers or freight on a cable railway"

# doc_a = "a motor vehicle with four wheels; usually propelled by an internal combustion engine"
# doc_b = "a class of problems. Algorithms can perform calculation, data processing and automated reasoning tasks."
# doc_c = "A vehicle is a machine that transports people or cargo. Vehicles include wagons, bicycles, motor vehicles, railed vehicles, watercraft, amphibious vehicles, aircraft and spacecraft."
# doc_d = "doctor is a professional who practises medicine, which is concerned with promoting, maintaining, or restoring health through the study, diagnosis, and treatment of disease, injury, and other physical and mental impairments."
# doc_e = "A university is an institution of higher education and research which awards academic degrees in various academic disciplines. Universities typically provide undergraduate education and postgraduate education."

# doc_a = "A mobile phone also known as a wireless phone, cell phone, or a small portable radio telephone."
# doc_b = "The mobile phone can be used to communicate over long distances without wires. It works by communicating with a nearby base station which connects it to the main phone network."
# doc_c = "When moving, if the mobile phone gets too far away from the cell it is connected to, that cell sends a message to another cell to tell the new cell to take over the call."
# doc_d = "This is called a 'hand off', and the call continues with the new cell the phone is connected to. The hand-off is done so well and carefully that the user will usually never even know that the call was transferred to another cell."
# doc_e = "As mobile phones became more popular, they began to cost less money, and more people could afford them."

doc_set = [doc_a, doc_b, doc_c, doc_d, doc_e]

# list for tokenized documents in loop
texts = []

# loop through document list
for i in doc_set:
    # clean and tokenize document string
    raw = i.lower()
    tokens = tokenizer.tokenize(raw)

    # remove stop words from tokens
    stopped_tokens = [i for i in tokens if not i in en_stop]

    # stem tokens
    stemmed_tokens = [l_lemma.lemmatize(i) for i in stopped_tokens]

    #     pos_tagger = [nltk.pos_tag(i) for i in stemmed_tokens]

    #     nn_tagged = [(word,tag) for word, tag in pos_tagger
    #                 if tag.startswith('NN')]

    # add tokens to list

    texts.append(stemmed_tokens)

l = []
m = []

# for i in texts:
a = nltk.pos_tag(texts[0])
nn_tagged = [(word, tag) for word, tag in a if tag.startswith('NN') or tag.startswith('NNP')]

for i in nn_tagged:
    m.append(i[0])
l.append(m)

n = []
a = nltk.pos_tag(texts[1])
nn_tagged = [(word, tag) for word, tag in a if tag.startswith('NN') or tag.startswith('NNP')]

for i in nn_tagged:
    n.append(i[0])
l.append(n)

o = []
a = nltk.pos_tag(texts[2])
nn_tagged = [(word, tag) for word, tag in a if tag.startswith('NN') or tag.startswith('NNP')]

for i in nn_tagged:
    o.append(i[0])
l.append(o)

p = []
a = nltk.pos_tag(texts[3])
nn_tagged = [(word, tag) for word, tag in a if tag.startswith('NN') or tag.startswith('NNP')]

for i in nn_tagged:
    p.append(i[0])
l.append(p)

q = []
a = nltk.pos_tag(texts[4])
nn_tagged = [(word, tag) for word, tag in a if tag.startswith('NN') or tag.startswith('NNP')]

for i in nn_tagged:
    q.append(i[0])
l.append(q)
print(l)

# print(m)

# l=[]
#

# a=nltk.pos_tag(texts[0])
# nn_tagged = [(word,tag) for word, tag in a
#                 if tag.startswith('NN') or tag.startswith('NNP')]

# for i in nn_tagged:
#     l.append(i[0])

# print(l)
# ss=[l]
# print(ss)

# m=[]
# a=nltk.pos_tag(texts[1])
# nn_tagged = [(word,tag) for word, tag in a
#                 if tag.startswith('NN') or tag.startswith('NNP')]

# for i in nn_tagged:
#     m.append(i[0])

# print(m)
# print(ss+=m)


# turn our tokenized documents into a id <-> term dictionary
dictionary = corpora.Dictionary(l)

# convert tokenized documents into a document-term matrix
corpus = [dictionary.doc2bow(text) for text in l]

# generate LDA model
ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics=4, id2word=dictionary, passes=2000, random_state=1)
# ldamodel2 =  gensim.models.LdaMulticore(corpus,
#                                    num_topics = 4,
#                                    id2word = dictionary,
#                                    passes = 2000,
#                                    workers = 2)

# print(nltk.pos_tag(texts[0]))
# a=nltk.pos_tag(texts[0])
# nn_tagged = [(word,tag) for word, tag in a
#                 if tag.startswith('NN') or tag.startswith('NNP')]
# print(nn_tagged)
# l=[]
# for i in nn_tagged:
#     l.append(i[0])
# print(l)

# l=[]
# m=[]
# for i in texts:
#     a=nltk.pos_tag(i)
#     nn_tagged = [(word,tag) for word, tag in a
#                 if tag.startswith('NN') or tag.startswith('NNP')]

# #     abc=[a for i in nn_tagged]
# #     l.append(abc)

#     for i in nn_tagged:
#         l.append(i[0])


# print(l)


# for i in texts:
#     nn_vb_tagged = [(word,tag) for word, tag in i
#                 if tag.startswith('NN') or tag.startswith('NNP')]
#     print(nn_vb_tagged)
# print(pos_tagger)
# print(tokens)
# print(texts)
# print(dictionary)
# print(ldamodel)
# print(corpus)
v = ldamodel.print_topics(num_topics=4, num_words=2)
print(v)
# t1,t2,t3,t4
# for i in v:
#     if i==0:
#         t1=i
#     elif i==1:
#         t2=i
#     elif i==2:
#         t3=i
#     elif i==3:
#         t4=i


t1 = v[0][1]
t2 = v[1][1]
t3 = v[2][1]
t4 = v[3][1]
print(t1, "\n", t2, "\n", t3, "\n", t4, "\n")

# print(t1)
tv2 = t1.split()[0]
# print(tv2)
import re

tv3 = " ".join(re.findall("[a-zA-Z]+", tv2))
# print(tv3)
tv4 = re.split("[^a-zA-Z]*", tv3)
# print(tv4)
best_topic = ""
for item in tv4:
    best_topic = str(item)
    print(best_topic)

syns = wordnet.synsets("education")
des = []

n_doc_a = ""
n_doc_b = ""
n_doc_c = ""
n_doc_d = ""
n_doc_e = ""

for i in range(4):
    if i == 0:
        n_doc_a = syns[i].definition()
        print(n_doc_a)
    elif i == 1:
        n_doc_b = syns[i].definition()
    elif i == 2:
        n_doc_c = syns[i].definition()
    elif i == 3:
        n_doc_d = syns[i].definition()
    elif i == 4:
        n_doc_e = syns[i].definition()

#     des.append(syns[i].definition())
#     des.append(".")
# print(des)

n_doc_set = [n_doc_a, n_doc_b, n_doc_c, n_doc_d, n_doc_e]
n_texts = []
for i in n_doc_set:
    # clean and tokenize document string
    raw = i.lower()
    tokens = tokenizer.tokenize(raw)

    # remove stop words from tokens
    stopped_tokens = [i for i in tokens if not i in en_stop]

    # stem tokens
    stemmed_tokens = [l_lemma.lemmatize(i) for i in stopped_tokens]

    n_texts.append(stemmed_tokens)

print(n_texts)

n_l = []
n_m = []

# for i in texts:
n_a = nltk.pos_tag(n_texts[0])
n_nn_tagged = [(word, tag) for word, tag in n_a if tag.startswith('NN') or tag.startswith('NNP')]

for i in n_nn_tagged:
    n_m.append(i[0])
n_l.append(n_m)

n_n = []
n_a = nltk.pos_tag(n_texts[1])
n_nn_tagged = [(word, tag) for word, tag in n_a if tag.startswith('NN') or tag.startswith('NNP')]

for i in n_nn_tagged:
    n_n.append(i[0])
n_l.append(n_n)

n_o = []
n_a = nltk.pos_tag(n_texts[2])
n_nn_tagged = [(word, tag) for word, tag in n_a if tag.startswith('NN') or tag.startswith('NNP')]

for i in n_nn_tagged:
    n_o.append(i[0])
n_l.append(n_o)

n_p = []
n_a = nltk.pos_tag(n_texts[3])
n_nn_tagged = [(word, tag) for word, tag in n_a if tag.startswith('NN') or tag.startswith('NNP')]

for i in n_nn_tagged:
    n_p.append(i[0])
n_l.append(n_p)

n_q = []
n_a = nltk.pos_tag(n_texts[4])
n_nn_tagged = [(word, tag) for word, tag in n_a if tag.startswith('NN') or tag.startswith('NNP')]

for i in n_nn_tagged:
    n_q.append(i[0])
n_l.append(n_q)
print(n_l)

n_dictionary = corpora.Dictionary(n_l)

# convert tokenized documents into a document-term matrix
n_corpus = [n_dictionary.doc2bow(n_text) for n_text in n_l]

# generate LDA model
n_ldamodel = gensim.models.ldamodel.LdaModel(n_corpus, num_topics=1, id2word=n_dictionary, passes=2000, random_state=1)
n_v = n_ldamodel.print_topics(num_topics=1, num_words=1)
print(n_v)

# print(t2)
# print(t3)
# print(t4)
# print(ldamodel2.print_topics(num_topics=4, num_words=2))
# vis_data = gensimvis.prepare(ldamodel, corpus, dictionary)
# pyLDAvis.display(vis_data)
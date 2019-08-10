


import re
import nltk
from gensim import models, corpora
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from stop_words import get_stop_words
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer

NUM_TOPICS = 3
STOPWORDS = stopwords.words('english')
tokenizer = RegexpTokenizer(r'\w+')

# create English stop words list
en_stop = get_stop_words('en')

# Create p_stemmer of class PorterStemmer
p_stemmer = PorterStemmer()
l_lemma = WordNetLemmatizer()


def clean_text(text):
    tokenized_text = word_tokenize(text.lower())
    cleaned_text = [t for t in tokenized_text if t not in STOPWORDS and re.match('[a-zA-Z\-][a-zA-Z\-]{2,}', t)]
    return cleaned_text


# For gensim we need to tokenize the data and filter out stopwords
doc_a = "Drug addiction is a dependence syndrome. It is a condition where a person feels a strong desire to consume drugs and can’t do without them."
doc_b= "The feeling to consume it is more important for them than other daily chores and even their family"
doc_c= "If the addicted one does not use it for longer time he is likely to feel depressed and isolated."
doc_d= "Addiction is the state where mind and body just cannot do without it."
doc_e= "The brain changes are persistent that is why drug addiction is often defined as a form of mental disorder."

#2
# doc_a = "Sugar is bad to consume. My sister likes to have sugar, but not my father."
# doc_b = "My father spends a lot of time driving my sister around to dance practice."
# doc_c = "Doctors suggest that driving may cause increased stress and blood pressure."
# doc_d = "Sometimes I feel pressure to perform well at school, but my father never seems to drive my sister to do better."
# doc_e = "Health experts say that Sugar is not good for your lifestyle."

#1
# doc_a = "Traffic jam means a long line of vehicles that can not move or that can move very slowly. It is a common affair in the big cities of our country. There are many causes of traffic jam. Rapid growth of population and the increasing amount of vehicles are the main causes of it."
# doc_b = "Vehicles are much more than the roads can accommodate. The indiscriminate playing of rickshaw is another causes of it. Haphazard parking of vehicles alongside the pavement also causes of it. "
# doc_c = " Violation of traffic rules is also responsible for it. The drivers do not follow traffic rules. Traffic jam causes untold sufferings to people. Sometimes it raises our mental tension. It causes loss of our valuable time. We have to wait to reach our destination. "
# doc_d = " The students, the office-going people, the businessmen and the patients in the ambulance are the worst sufferers of it. Traffic jam can be removed by enforcing traffic jam strictly. "
# doc_e = "The narrow roads should be broadened. By pass roads should be constructed in the big towns. One way movement of vehicles and building of fly over can solve this problem. We can reduce it by raising public awareness."


data = [doc_a, doc_b, doc_c, doc_d, doc_e]
texts = []
for i in data:
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


# Build a Dictionary - association word to numeric id
dictionary = corpora.Dictionary(texts)

# Transform the collection of texts to a numerical form
corpus = [dictionary.doc2bow(text) for text in texts]

# Have a look at how the 20th document looks like: [(word_id, count), ...]

# [(12, 3), (14, 1), (21, 1), (25, 5), (30, 2), (31, 5), (33, 1), (42, 1), (43, 2),  ...

# Build the LDA model
lda_model = models.LdaModel(corpus=corpus, num_topics=NUM_TOPICS, id2word=dictionary, random_state=1)

# Build the LSI modelonepass=True
lsi_model = models.LsiModel(corpus=corpus, num_topics=NUM_TOPICS, id2word=dictionary, onepass=True)

print("LDA Model:")

for idx in range(NUM_TOPICS):
    # Print the first 10 most representative topics
    print("Topic #%s:" % idx, lda_model.print_topic(idx, 3))

print("=" * 20)

print("LSI Model:")

for idx in range(NUM_TOPICS):
    # Print the first 10 most representative topics
    print("Topic #%s:" % idx, lsi_model.print_topic(idx, 3))

print("=" * 20)
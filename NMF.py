import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.decomposition import NMF
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
#

data = [doc_a, doc_b, doc_c, doc_d, doc_e]

# convert the text to a tf-idf weighted term-document matrix

vectorizer = TfidfVectorizer(stop_words='english')

X = vectorizer.fit_transform(data)

idx_to_word = np.array(vectorizer.get_feature_names())

# apply NMF

nmf = NMF(n_components=3, solver="mu")

W = nmf.fit_transform(X)

H = nmf.components_

# print the topics

for i, topic in enumerate(H):
    print("Topic {}: {}".format(i + 1, ",".join([str(x) for x in idx_to_word[topic.argsort()[-3:]]])))
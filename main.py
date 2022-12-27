import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm
from annotated_text import annotated_text
import graphviz

@st.cache
def load_data_txt(path):
    f = open(path, "r")
    list_of_words = f.readlines()
    new_list = []
    for l in list_of_words:
        words = []
        for word in l[1:-3].split(', '):
            words += [word[1:-1]]
        new_list += [words]
    return new_list

@st.cache
def load_data_csv(path):
    df = pd.read_csv(path)
    return df

def compose_text(tokenized_text, best_words, doc):
    result = []
    tmp = ""
    for token in tqdm(tokenized_text):
        if token == doc[:len(token)].lower():
            if token in best_words:
                index = best_words.index(token)
                token_mod = doc[:len(token)]
                result += [(token_mod, index)]
            else:
                token_mod = doc[:len(token)]
                result += [token_mod]
            doc = doc[len(token):]
        else:
            counter = 1
            tmp = ""
            while token not in tmp.lower():
                tmp = doc[:counter]
                counter += 1
                if tmp[-1:] == '\n':
                    result += tmp[:-1]
                    result += tmp[-1:]
                    result += tmp[-1:]
                    doc = doc[len(tmp):]
                    counter = 1
                    tmp = ""
            doc = doc[len(tmp):]

            extra = tmp[:-len(token)]
            result += [extra]

            if token in best_words:
                index = best_words.index(token)
                token_mod = tmp[-len(token):]
                result += [(token_mod, str(index+1))]
            else:
                token_mod = tmp[-len(token):]
                result += [token_mod]
    return result

def create_graph(tokens, vocab):
    g = graphviz.Graph(format='svg')
    last = ''
    for t in tokens:
        if t in vocab.keys():
            if last == '':
                g.node(t, t)
            else:
                g.node(t, t)
                g.edge(t, last)
            last = t
    return g

def jaccard_similarity(top_words_a, top_words_b, k=1000):
    A = set(top_words_a[:k])
    B = set(top_words_b[:k])
    if len(A) == 0 and len(B) == 0:
        return 1
    return len(A & B) / len(A | B)

st.title('GNNs Explaination')

data_load_state = st.text('Loading data...')

words_omission = {'GCN': load_data_txt('./data/omission_gcn.txt'),
                  'GAT': load_data_txt('./data/omission_gat.txt'),
                  'SAGEGraph': load_data_txt('./data/omission_gat.txt'),
                  'SimpleGCN': load_data_txt('./data/omission_gcn.txt')
                  }

words_saliency = {'GCN': load_data_txt('./data/saliency_gcn.txt'),
                  'GAT': load_data_txt('./data/saliency_gat.txt'),
                  'SAGEGraph': load_data_txt('./data/saliency_gat.txt'),
                  'SimpleGCN': load_data_txt('./data/saliency_gcn.txt')
                  }

words_random = {'GCN': load_data_txt('./data/random_gcn.txt'),
                'GAT': load_data_txt('./data/random_gat.txt'),
                'SAGEGraph': load_data_txt('./data/random_gat.txt'),
                'SimpleGCN': load_data_txt('./data/random_gcn.txt')
                  }

data = load_data_csv('./data/test_data_text.csv')
train_data = load_data_csv('./data/train_data_text.csv')

vectorizer = TfidfVectorizer(lowercase=True, min_df=10, max_df=0.25, norm='l1')

vectorizer.fit(train_data['data'])
tokenize_func = vectorizer.build_analyzer()
vocab = vectorizer.vocabulary_

data_load_state.text('Loading data...done!')

option = st.selectbox("Select from examples",
                      data['data'])

st.subheader('Chosen Text')
st.text(option)

st.subheader('Labels')
st.text(data[data['data'] == option]['label'].iloc[0])

index = data.index[data['data'] == option]

option_expl = st.selectbox("Select explainability methods",
                      ['random', 'omission', 'saliency'])

if option_expl == 'omission':
    words = words_omission
elif option_expl == 'saliency':
    words = words_saliency
elif option_expl == 'random':
    words = words_random

st.session_state["k"] = st.slider('How many top words would you like to see?', min_value=5, max_value=50, value=10,
                                  step=1)
st.session_state['tokenized'] = tokenize_func(option)
subset_words = {}
for model in words.keys():
    words_selected = words[model][index[0]]
    k_tmp = min(st.session_state["k"], len(words_selected))
    subset_words[model] = words_selected[:k_tmp]

    st.subheader(model)
    annotated_text(*compose_text(st.session_state['tokenized'], subset_words[model], option))

st.subheader('Created Graph')
st.graphviz_chart(create_graph(st.session_state['tokenized'] , vocab))

st.subheader('Jaccard Similarity')

col = ['GCN', 'GAT', 'SAGEGraph', 'SimpleGCN']
df = pd.DataFrame([], columns=col)

for i, el1 in enumerate(col):
    row = {}
    for el2 in col:
        row[el2] = jaccard_similarity(subset_words[el1], subset_words[el2])
    df = pd.concat([df, pd.DataFrame([row.copy()])])
df.index = col
st.table(df)











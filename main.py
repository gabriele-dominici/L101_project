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
                result += [(str(token_mod), index)]
            else:
                token_mod = doc[:len(token)]
                result += [str(token_mod)]
            doc = doc[len(token):]
        else:
            counter = 1
            tmp = ""
            while token not in tmp.lower():
                tmp = doc[:counter]
                counter += 1
                if tmp[-1:] == '\n':
                    result += tmp[:-1]
                    result += '  ' + tmp[-1:]
                    # result += tmp[-1:]
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
st.markdown('This website is created to visualise the local explanations generated in the '
        '"[Demystifying Graph Neural Networks in Document Classification: Analysis of Local Explanations](https://github.com/gabriele-dominici/L101_web)" '
        'project.')
st.markdown('It is possible to select one of two document classification dataset, the interpretability methods used'
        ' and the models to inspect.')
data_load_state = st.text('Loading data... it can take a bit')

words_omission ={'20news_group': {'MLP': load_data_txt('./data/omission_mlp.txt'),
                  'GCN_2Layer_Mean': load_data_txt('./data/omission_gcn.txt'),
                  'GCN_3Layer_Mean': load_data_txt('./data/omission_gcn_3.txt'),
                  'GCN_2Layer_Max': load_data_txt('./data/omission_gcn_max.txt'),
                  'GAT_2Layer_Mean': load_data_txt('./data/omission_gat.txt'),
                  'GAT_3Layer_Mean': load_data_txt('./data/omission_gat_3.txt'),
                  'GAT_2Layer_Max': load_data_txt('./data/omission_gat_max.txt'),
                  'SAGEGraph_2Layer_Mean': load_data_txt('./data/omission_sage.txt'),
                  'SAGEGraph_3Layer_Mean': load_data_txt('./data/omission_sage_3.txt'),
                  'SAGEGraph_2Layer_Max': load_data_txt('./data/omission_sage_max.txt'),
                  'SGC_2Layer_Mean': load_data_txt('./data/omission_simple.txt'),
                  'SGC_3Layer_Mean': load_data_txt('./data/omission_simple_3.txt'),
                  'SGC_2Layer_Max': load_data_txt('./data/omission_simple_max.txt')
                                  },
                 'movie': {'MLP': load_data_txt('./data/omission_mlp_movie.txt'),
                           'GCN_2Layer_Mean': load_data_txt('./data/omission_gcn_movie.txt'),
                           'GCN_3Layer_Mean': load_data_txt('./data/omission_gcn_3_movie.txt'),
                           'GCN_2Layer_Max': load_data_txt('./data/omission_gcn_max_movie.txt'),
                           'GAT_2Layer_Mean': load_data_txt('./data/omission_gat_movie.txt'),
                           'GAT_3Layer_Mean': load_data_txt('./data/omission_gat_3_movie.txt'),
                           'GAT_2Layer_Max': load_data_txt('./data/omission_gat_max_movie.txt'),
                           'SAGEGraph_2Layer_Mean': load_data_txt('./data/omission_sage_movie.txt'),
                           'SAGEGraph_3Layer_Mean': load_data_txt('./data/omission_sage_3_movie.txt'),
                           'SAGEGraph_2Layer_Max': load_data_txt('./data/omission_sage_max_movie.txt'),
                           'SGC_2Layer_Mean': load_data_txt('./data/omission_simple_movie.txt'),
                           'SGC_3Layer_Mean': load_data_txt('./data/omission_simple_3_movie.txt'),
                           'SGC_2Layer_Max': load_data_txt('./data/omission_simple_max_movie.txt')
                                  }
                  }

words_saliency = {'20news_group': {'MLP': load_data_txt('./data/saliency_mlp.txt'),
                                   'GCN_2Layer_Mean': load_data_txt('./data/saliency_gcn.txt'),
                                   'GCN_3Layer_Mean': load_data_txt('./data/saliency_gcn_3.txt'),
                                   'GCN_2Layer_Max': load_data_txt('./data/saliency_gcn_max.txt'),
                                   'GAT_2Layer_Mean': load_data_txt('./data/saliency_gat.txt'),
                                   'GAT_3Layer_Mean': load_data_txt('./data/saliency_gat_3.txt'),
                                   'GAT_2Layer_Max': load_data_txt('./data/saliency_gat_max.txt'),
                                   'SAGEGraph_2Layer_Mean': load_data_txt('./data/saliency_sage.txt'),
                                   'SAGEGraph_3Layer_Mean': load_data_txt('./data/saliency_sage_3.txt'),
                                   'SAGEGraph_2Layer_Max': load_data_txt('./data/saliency_sage_max.txt'),
                                   'SGC_2Layer_Mean': load_data_txt('./data/saliency_simple.txt'),
                                   'SGC_3Layer_Mean': load_data_txt('./data/saliency_simple_3.txt'),
                                   'SGC_2Layer_Max': load_data_txt('./data/saliency_simple_max.txt')
                                    },
                     'movie': {'MLP': load_data_txt('./data/saliency_mlp_movie.txt'),
                               'GCN_2Layer_Mean': load_data_txt('./data/saliency_gcn_movie.txt'),
                               'GCN_3Layer_Mean': load_data_txt('./data/saliency_gcn_3_movie.txt'),
                               'GCN_2Layer_Max': load_data_txt('./data/saliency_gcn_max_movie.txt'),
                               'GAT_2Layer_Mean': load_data_txt('./data/saliency_gat_movie.txt'),
                               'GAT_3Layer_Mean': load_data_txt('./data/saliency_gat_3_movie.txt'),
                               'GAT_2Layer_Max': load_data_txt('./data/saliency_gat_max_movie.txt'),
                               'SAGEGraph_2Layer_Mean': load_data_txt('./data/saliency_sage_movie.txt'),
                               'SAGEGraph_3Layer_Mean': load_data_txt('./data/saliency_sage_3_movie.txt'),
                               'SAGEGraph_2Layer_Max': load_data_txt('./data/saliency_sage_max_movie.txt'),
                               'SGC_2Layer_Mean': load_data_txt('./data/saliency_simple_movie.txt'),
                               'SGC_3Layer_Mean': load_data_txt('./data/saliency_simple_3_movie.txt'),
                               'SGC_2Layer_Max': load_data_txt('./data/saliency_simple_max_movie.txt')
                             }
                    }

words_random = {'20news_group': {'MLP': load_data_txt('./data/random_mlp.txt'),
                                 'GCN_2Layer_Mean': load_data_txt('./data/random_gcn.txt'),
                                 'GCN_3Layer_Mean': load_data_txt('./data/random_gcn_3.txt'),
                                 'GCN_2Layer_Max': load_data_txt('./data/random_gcn_max.txt'),
                                 'GAT_2Layer_Mean': load_data_txt('./data/random_gat.txt'),
                                 'GAT_3Layer_Mean': load_data_txt('./data/random_gat_3.txt'),
                                 'GAT_2Layer_Max': load_data_txt('./data/random_gat_max.txt'),
                                 'SAGEGraph_2Layer_Mean': load_data_txt('./data/random_sage.txt'),
                                 'SAGEGraph_3Layer_Mean': load_data_txt('./data/random_sage_3.txt'),
                                 'SAGEGraph_2Layer_Max': load_data_txt('./data/random_sage_max.txt'),
                                 'SGC_2Layer_Mean': load_data_txt('./data/random_simple.txt'),
                                 'SGC_3Layer_Mean': load_data_txt('./data/random_simple_3.txt'),
                                 'SGC_2Layer_Max': load_data_txt('./data/random_simple_max.txt')
                                },
                'movie': {'MLP': load_data_txt('./data/random_mlp_movie.txt'),
                          'GCN_2Layer_Mean': load_data_txt('./data/random_gcn_movie.txt'),
                          'GCN_3Layer_Mean': load_data_txt('./data/random_gcn_3_movie.txt'),
                          'GCN_2Layer_Max': load_data_txt('./data/random_gcn_max_movie.txt'),
                          'GAT_2Layer_Mean': load_data_txt('./data/random_gat_movie.txt'),
                          'GAT_3Layer_Mean': load_data_txt('./data/random_gat_3_movie.txt'),
                          'GAT_2Layer_Max': load_data_txt('./data/random_gat_max_movie.txt'),
                          'SAGEGraph_2Layer_Mean': load_data_txt('./data/random_sage_movie.txt'),
                          'SAGEGraph_3Layer_Mean': load_data_txt('./data/random_sage_3_movie.txt'),
                          'SAGEGraph_2Layer_Max': load_data_txt('./data/random_sage_max_movie.txt'),
                          'SGC_2Layer_Mean': load_data_txt('./data/random_simple_movie.txt'),
                          'SGC_3Layer_Mean': load_data_txt('./data/random_simple_3_movie.txt'),
                          'SGC_2Layer_Max': load_data_txt('./data/random_simple_max_movie.txt')
                          }
                }
words_gnne = {'20news_group': {'GCN_2Layer_Mean': load_data_txt('./data/gnne_gcn.txt'),
                               'GCN_3Layer_Mean': load_data_txt('./data/gnne_gcn_3.txt'),
                               'GCN_2Layer_Max': load_data_txt('./data/gnne_gcn_max.txt'),
                               'GAT_2Layer_Mean': load_data_txt('./data/gnne_gat.txt'),
                               'GAT_3Layer_Mean': load_data_txt('./data/gnne_gat_3.txt'),
                               'GAT_2Layer_Max': load_data_txt('./data/gnne_gat_max.txt'),
                               'SAGEGraph_2Layer_Mean': load_data_txt('./data/gnne_sage.txt'),
                               'SAGEGraph_3Layer_Mean': load_data_txt('./data/gnne_sage_3.txt'),
                               'SAGEGraph_2Layer_Max': load_data_txt('./data/gnne_sage_max.txt'),
                               'SGC_2Layer_Mean': load_data_txt('./data/gnne_simple.txt'),
                               'SGC_3Layer_Mean': load_data_txt('./data/gnne_simple_3.txt'),
                               'SGC_2Layer_Max': load_data_txt('./data/gnne_simple_max.txt')
                                },
                'movie': {'GCN_2Layer_Mean': load_data_txt('./data/gnne_gcn_movie.txt'),
                          'GCN_3Layer_Mean': load_data_txt('./data/gnne_gcn_3_movie.txt'),
                          'GCN_2Layer_Max': load_data_txt('./data/gnne_gcn_max_movie.txt'),
                          'GAT_2Layer_Mean': load_data_txt('./data/gnne_gat_movie.txt'),
                          'GAT_3Layer_Mean': load_data_txt('./data/gnne_gat_3_movie.txt'),
                          'GAT_2Layer_Max': load_data_txt('./data/gnne_gat_max_movie.txt'),
                          'SAGEGraph_2Layer_Mean': load_data_txt('./data/gnne_sage_movie.txt'),
                          'SAGEGraph_3Layer_Mean': load_data_txt('./data/gnne_sage_3_movie.txt'),
                          'SAGEGraph_2Layer_Max': load_data_txt('./data/gnne_sage_max_movie.txt'),
                          'SGC_2Layer_Mean': load_data_txt('./data/gnne_simple_movie.txt'),
                          'SGC_3Layer_Mean': load_data_txt('./data/gnne_simple_3_movie.txt'),
                          'SGC_2Layer_Max': load_data_txt('./data/gnne_simple_max_movie.txt')
                          }
                }
predictions = {
                '20news_group': {'MLP': load_data_csv('./data/mlp_news.csv'),
                                 'GCN_2Layer_Mean': load_data_csv('./data/gcn_2_mean_news.csv'),
                                 'GCN_3Layer_Mean': load_data_csv('./data/gcn_3_mean_news.csv'),
                                 'GCN_2Layer_Max': load_data_csv('./data/gcn_2_max_news.csv'),
                                 'GAT_2Layer_Mean': load_data_csv('./data/gat_2_mean_news.csv'),
                                 'GAT_3Layer_Mean': load_data_csv('./data/gat_3_mean_news.csv'),
                                 'GAT_2Layer_Max': load_data_csv('./data/gat_2_max_news.csv'),
                                 'SAGEGraph_2Layer_Mean': load_data_csv('./data/sage_2_mean_news.csv'),
                                 'SAGEGraph_3Layer_Mean': load_data_csv('./data/sage_3_mean_news.csv'),
                                 'SAGEGraph_2Layer_Max': load_data_csv('./data/sage_2_max_news.csv'),
                                 'SGC_2Layer_Mean': load_data_csv('./data/simple_2_mean_news.csv'),
                                 'SGC_3Layer_Mean': load_data_csv('./data/simple_3_mean_news.csv'),
                                 'SGC_2Layer_Max': load_data_csv('./data/simple_2_max_news.csv')
                                },
                'movie': {'MLP': load_data_csv('./data/mlp_movie.csv'),
                          'GCN_2Layer_Mean': load_data_csv('./data/gcn_2_mean_movie.csv'),
                          'GCN_3Layer_Mean': load_data_csv('./data/gcn_3_mean_movie.csv'),
                          'GCN_2Layer_Max': load_data_csv('./data/gcn_2_max_movie.csv'),
                          'GAT_2Layer_Mean': load_data_csv('./data/gat_2_mean_movie.csv'),
                          'GAT_3Layer_Mean': load_data_csv('./data/gat_3_mean_movie.csv'),
                          'GAT_2Layer_Max': load_data_csv('./data/gat_2_max_movie.csv'),
                          'SAGEGraph_2Layer_Mean': load_data_csv('./data/sage_2_mean_movie.csv'),
                          'SAGEGraph_3Layer_Mean': load_data_csv('./data/sage_3_mean_movie.csv'),
                          'SAGEGraph_2Layer_Max': load_data_csv('./data/sage_2_max_movie.csv'),
                          'SGC_2Layer_Mean': load_data_csv('./data/simple_2_mean_movie.csv'),
                          'SGC_3Layer_Mean': load_data_csv('./data/simple_3_mean_movie.csv'),
                          'SGC_2Layer_Max': load_data_csv('./data/simple_2_max_movie.csv')
                        }
}

data = {'20news_group': load_data_csv('./data/test_data_text.csv'),
        'movie': load_data_csv('./data/test_data_movie.csv')}
train_data = {'20news_group': load_data_csv('./data/train_data_text.csv'),
              'movie': load_data_csv('./data/train_data_movie.csv')}

vectorizer = {'20news_group': TfidfVectorizer(lowercase=True, min_df=10, max_df=0.25, norm='l1'),
              'movie': TfidfVectorizer(lowercase=True, min_df=10, max_df=0.25, norm='l1')}

vectorizer['20news_group'].fit(train_data['20news_group']['data'])
vectorizer['movie'].fit(train_data['movie']['data'])
tokenize_func = {'20news_group': vectorizer['20news_group'].build_analyzer(),
                 'movie': vectorizer['movie'].build_analyzer()}
vocab = {'20news_group': vectorizer['20news_group'].vocabulary_,
         'movie': vectorizer['movie'].vocabulary_}

data_load_state.text('Loading data...done!')
dataset = st.selectbox("Select the dataset",
                      ['20news_group',
                       'movie'])

st.session_state["option"] = st.selectbox("Select from examples",
                      data[dataset]['data'])



st.subheader('Chosen Document')
st.markdown(st.session_state["option"])

st.subheader('Label')
label = data[dataset][data[dataset]['data'] == st.session_state["option"]]['label'].iloc[0]
if label == 1 and dataset == '20news_group':
    label_text = '**Atheism**'
elif label == 0 and dataset == '20news_group':
    label_text = '**Christian**'
elif label == 1 and dataset == 'movie':
    label_text = '**Positive**'
elif label == 0 and dataset == 'movie':
    label_text = '**Negative**'
st.markdown(label_text)

index = data[dataset].index[data[dataset]['data'] == st.session_state["option"]]
st.subheader('Explainations')

option_expl = st.selectbox("Select interpretability methods",
                      ['Random', 'Omission', 'Vanilla Gradient', 'GNNExplainer'])
col = []

if option_expl == 'Omission':
    words = words_omission[dataset]
elif option_expl == 'Vanilla Gradient':
    words = words_saliency[dataset]
elif option_expl == 'GNNExplainer':
    words = words_gnne[dataset]
elif option_expl == 'Random':
    words = words_random[dataset]

col = st.multiselect("Select models",
                      words.keys())

st.session_state["k"] = st.slider('How many top words would you like to see?', min_value=5, max_value=50, value=10,
                                  step=1)

st.session_state['tokenized'] = tokenize_func[dataset](st.session_state["option"])
subset_words = {}
for model in col:
    words_selected = words[model][index[0]]
    pred_label = predictions[dataset][model].iloc[index[0]][0]
    if pred_label == 1 and dataset == '20news_group':
        pred_label = '**Atheism**'
    elif pred_label == 0 and dataset == '20news_group':
        pred_label = '**Christian**'
    elif pred_label == 1 and dataset == 'movie':
        pred_label = '**Positive**'
    elif pred_label == 0 and dataset == 'movie':
        pred_label = '**Negative**'
    k_tmp = min(st.session_state["k"], len(words_selected))
    subset_words[model] = words_selected[:k_tmp]

    st.subheader(model)
    st.markdown(f'Prediction of the model: {pred_label}')
    try:
        annotated_text(*compose_text(st.session_state['tokenized'], subset_words[model], st.session_state["option"]))
    except Exception as e:
        print(e)
st.subheader('Created Graph')
st.markdown('The following graph is the one used by the models for this document')
st.graphviz_chart(create_graph(st.session_state['tokenized'], vocab[dataset]))

st.subheader('Jaccard Similarity')
st.markdown('The table shows the Jaccard Similarity between models chosen computed on the the local explanations shown ')
df = pd.DataFrame([], columns=col)

for i, el1 in enumerate(col):
    row = {}
    for el2 in col:
        row[el2] = jaccard_similarity(subset_words[el1], subset_words[el2])
    df = pd.concat([df, pd.DataFrame([row.copy()])])
df.index = col
st.table(df)





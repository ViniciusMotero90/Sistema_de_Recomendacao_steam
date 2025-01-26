import ast
import nltk
import sklearn
import numpy as np
import pandas as pd
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
pd.options.mode.chained_assignment = None

steam = pd.read_csv('steam.csv')

steam.dropna(inplace=True)
steam = steam[['appid','name','release_date','genres','steamspy_tags']]

def converter(obj):
    return obj.split(';')

steam['genres'] = steam['genres'].apply(converter)
steam['steamspy_tags'] = steam['steamspy_tags'].apply(converter)
steam['release_date'] = steam['release_date'].apply(lambda x:x.split('-'))
steam['genres'] = steam['genres'].apply(lambda x:[i.replace(" ","") for i in x])
steam['steamspy_tags'] = steam['steamspy_tags'].apply(lambda x:[i.replace(" ","") for i in x])

steam['tags'] = steam['release_date']  + \
                steam['genres'] + \
                steam['steamspy_tags']

steam_novos = steam[['appid','name','tags']]

steam_novos['tags'] = steam_novos['tags'].apply(lambda x:" ".join(x))
steam_novos['tags'] = steam_novos['tags'].apply(lambda x:x.lower())

parser_ps = PorterStemmer()

def stem(text):

    y = []

    for i in text.split():
        y.append(parser_ps.stem(i))
    return " ".join(y)

steam_novos['tags'] = steam_novos['tags'].apply(stem)


cv = CountVectorizer(max_features=5000,stop_words='english')

vectors = cv.fit_transform(steam_novos['tags']).toarray()

np.set_printoptions(threshold=np.inf)

similaridade = cosine_similarity(vectors)

def sistema_recomendacao(game):
    # Verifica se o jogo existe no dataset
    if game not in steam_novos['name'].values:
        print(f"Jogo '{game}' não encontrado no dataset.")
        return

    # Encontra o índice do jogo no DataFrame
    index = steam_novos[steam_novos['name'] == game].index[0]

    # Calcula a similaridade e ordena os resultados
    distancia = sorted(list(enumerate(similaridade[index])), reverse=True, key=lambda x: x[1])

    print(f"Jogos recomendados para '{game}':")
    for i in distancia[1:6]:  # Retorna os 5 jogos mais similares
        # Aqui acessamos corretamente o nome do jogo
        print(steam_novos.iloc[i[0]]['name'])

sistema_recomendacao('Jurassic Survival')
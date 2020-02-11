import pandas
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
from scipy.stats.stats import pearsonr
from scipy.spatial.distance import jaccard
from flask import Flask, flash, render_template, request, session, abort, jsonify, make_response    
from flask_bootstrap import Bootstrap
from flask_wtf import Form
from wtforms import SelectField, SubmitField
from wtforms import validators

app = Flask(__name__)
app.config.update(dict(
    SECRET_KEY="SECRET_KEY",
    WTF_CSRF_SECRET_KEY="WTF_CSRF_SECRET_KEY"
))
Bootstrap(app)

#config
MOVIES_FILE = "data/movies.csv"
USERS_FILE = "data/users.csv"
RATINGS_FILE = "data/ratings.csv"
DEFAULT_TOP_N_REC = 10
TRAIN_FRACTION = 0.8
TRAIN_SIZE = 700

# X TEST ONLY (non usare in produzione)
#np.random.seed(1) # toglie randomness

# var globali con i data frame principali
movies=None
users=None
ratings=None
feature_weights = None
movies_feature = None
movies_test = None
movies_train = None


# Importa i dati
def read_data(movies_file, users_file, ratings_file = None):
    global movies, users, ratings, feature_weights, movies_feature, movies_train, movies_test
    movies = pandas.read_csv(movies_file, sep="|", encoding='latin-1')
    users = pandas.read_csv(users_file, sep="|", encoding='latin-1')
    if(ratings_file): 
        ratings = pandas.read_csv(ratings_file, sep="|", encoding='latin-1') 
        
    feature_weights = pandas.read_csv("data/feature_weights.csv", sep="|", encoding='latin-1').iloc[0,0:]
    movies_feature = pandas.read_csv("data/movie_feature.csv", sep="|", encoding='latin-1')
    movies_test = pandas.read_csv("data/movie_test.csv", sep="|", encoding='latin-1')
    movies_train = pandas.read_csv("data/movie_train.csv", sep="|", encoding='latin-1')

# preparo dati generi per chiamata WS
def get_genres():
    genres=[None]*(len(movies_test.columns)-1)
    
    genres[0]=(0, "All")
    #aggiungo i generi in lista
    for j in range(len(movies_test.columns)-2) :
        genres[j+1]=(j+1, movies_test.columns[j+2])
        
    return genres
    
# preparo dati film divisi per genere per chiamata WS
def get_test_movies_by_genre():
    movies_by_gender = [None]*(len(movies_test.columns)-1)

    #aggiungo i generi in lista
    movies_by_gender[0]={"genre":"All",
                         "movies":[]
                         }
    for i in range(len(movies_test.index)) :
        movies_by_gender[0]["movies"].append((movies_test.iloc[i, 0], movies_test.iloc[i, 1]))
    
    for j in range(len(movies_test.columns)-2) :
        genre={"genre": movies_test.columns[j+2],
               "movies":[]
              }
        movies_by_gender[j+1]=genre
            
    #aggiungo i film ai generi
    for i in range(len(movies_test.index)) :
        for j in range(len(movies_test.columns)-2) :
            if movies_test.iloc[i,j+2] == 1 :
                movies_by_gender[j+1]["movies"].append((movies_test.iloc[i, 0], movies_test.iloc[i, 1]))
    
    return movies_by_gender


def get_test_movies():
    return movies_test.to_json(orient='records', force_ascii=False)


def get_random_recommendation(id, n=DEFAULT_TOP_N_REC):
    random = np.random.randint(0,len(movies_test.index)-1)
    return get_recommendation(movies_test.iloc[random, 0], n)

# contiene l'algoritmo di raccomandazione
def get_recommendation(id, n=DEFAULT_TOP_N_REC): 
    id=float(id)
    # film scelto    
    movie = movies_test[movies_test.movie_id==id]
    if movie.shape[0] == 0 :
        return "-1"
    
    m_feats = movie.iloc[0,2:]    

    # calcolo distanza con film nel modello
    diss=[0.0]*len(movies_train.index)
    for i in range(len(movies_train.index)):
        diss[i] = jaccard(movies_train.iloc[i,2:], np.asmatrix(m_feats))
   
    sim=diss[:]
    for i in range(len(diss)):
        sim[i]= 1/(1+sim[i])

    sim_sum = sum(sim) 
    
    for i in range(len(m_feats)):
        m_feats[i] = float(m_feats[i])
    
    for i in range(len(m_feats)):
        if m_feats[i]==0:
            m_feats[i] = sim[i]/sim_sum
        m_feats[i] *= feature_weights[i]
    
    mf = movies_feature.iloc[0:, 2:]
    
    # calcolo correlazione film utente/film nel training set
    m_cor=[0.0]*len(movies_feature.index)
    for i in range(len(m_cor)):
        m_cor[i] = pearsonr(mf.iloc[i,0:], m_feats.values)[0]
        
    res_idx = np.argsort(m_cor)[::-1][0:n]
    
    return (np.asmatrix(movies_train)[res_idx, 0:2].tolist(), np.asarray(m_cor)[res_idx].tolist()) 
        
    # res <- order(as.vector(m_cor), na.last=TRUE, decreasing=TRUE)[1:10] 
    # cbind(movies_train[res,2], m_cor[res])

read_data(MOVIES_FILE, USERS_FILE)

#movies_train = movies.sample(frac=TRAIN_FRACTION)
#movies_test = movies.loc[movies.index.difference(movies_train.index)]
#movies_train = movies_train[0:TRAIN_SIZE]
#movies_feature=model_training(movies_train)

movies_by_genre=get_test_movies_by_genre()
genres = get_genres()

class SelctionForm(Form):
   genre = SelectField("Genre:", 
                [validators.Required("Please enter your name.")],
                choices=genres)
   movie = SelectField("Movie:", [validators.Required("Please enter your name.")], choices=movies_by_genre[0]['movies'])   
   submit = SubmitField("Get recommendations!")

@app.route("/")
def index():
    movies_by_genre = get_test_movies_by_genre()
    form=SelctionForm()
    return render_template('choose.html', form=form, movies=movies_by_genre) 

@app.route("/movies/")
def movies():
    #return render_template('test.html', movies=get_test_movies())
    return get_test_movies()


@app.route("/movies/<string:id>/")
def get_movies(id):
    if id == "random":
        return jsonify(get_random_recommendation(id))
    else:
        return jsonify(get_recommendation(id))

@app.route("/recommend/", methods=['POST'])
def recommend():
    id=request.form.get("movie")
    if id == "random":
        recommended = get_random_recommendation(id)
    else:
        id=float(id)
        # film scelto    
        movie = movies_test[movies_test.movie_id==id].iloc[0,1]
        recommended = get_recommendation(id)

    return render_template('result.html', preferred_movie=movie, recommended=recommended)


if __name__ == "__main__":
    app.run(debug=True, host= '0.0.0.0', port=8081)

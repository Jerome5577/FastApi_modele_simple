
#%pip install fastapi uvicorn pydantic 

from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
import pickle
from joblib import load
import contractions
import re
from chat_words_local import chat_words_list, chat_words_map_dict
from nltk.stem import PorterStemmer
from nltk import word_tokenize

# =========================================================================================== 
# Remove URL
def remove_urls(data):
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    data = url_pattern.sub(r'', str(data))
    return data
# Remove USERNAME
def remove_username(data):
    username_pattern = re.compile(r'@\S+')
    data = username_pattern.sub(r'', str(data))
    return data
# Replaciong emojis with their corresponding sentiments
def emoji(tweet):
    # Smile -- :), : ), :-), (:, ( :, (-:, :') , :O
    tweet = re.sub(r'(:\s?\)|:-\)|\(\s?:|\(-:|:\'\)|:O)', ' positiveemoji ', tweet)
    # Laugh -- :D, : D, :-D, xD, x-D, XD, X-D
    tweet = re.sub(r'(:\s?D|:-D|x-?D|X-?D)', ' positiveemoji ', tweet)
    # Love -- <3, :*
    tweet = re.sub(r'(<3|:\*)', ' positiveemoji ', tweet)
    # Wink -- ;-), ;), ;-D, ;D, (;,  (-; , @-)
    tweet = re.sub(r'(;-?\)|;-?D|\(-?;|@-\))', ' positiveemoji ', tweet)
    # Sad -- :-(, : (, :(, ):, )-:, :-/ , :-|
    tweet = re.sub(r'(:\s?\(|:-\(|\)\s?:|\)-:|:-/|:-\|)', ' negetiveemoji ', tweet)
    # Cry -- :,(, :'(, :"(
    tweet = re.sub(r'(:,\(|:\'\(|:"\()', ' negetiveemoji ', tweet)
    return tweet
# +++++++++++++++++++++++++++++++++++++++++++++==========
# Processing the tweet
def process_tweet_phase1(tweet):
    tweet = remove_username(tweet)                                    # Removes usernames
    tweet = remove_urls(tweet)                                        # Remove URLs
    tweet = emoji(tweet)                                               # Replaces Emojis
    return tweet
# +++++++++++++++++++++++++++++++++++++++++++++
# Conversion of chat words
def convert_chat_words(data):
    tokens = word_tokenize(str(data))
    new_text = []
    for w in tokens:
        if w.upper() in chat_words_list:
            new_text.append(chat_words_map_dict[w.upper()])
        else:
            new_text.append(w)
    return " ".join(new_text)
# EXPAND CONTRACTIONS
def expend_contractions(data):
    new_text = ""
    for word in str(data).split():
        # using contractions.fix to expand the shortened words
        #expanded_words.append(contractions.fix(word))      
        new_text = new_text + " " + contractions.fix(word)
    return new_text 
# REMOVE MAXIMUM    
def remove_maximum(data):
    data = re.sub(r'[^a-zA-z]', r' ', data)
    data = re.sub(r"\s+", " ", str(data))
    return data
# +++++++++++++++++++++++++++++++++++++++++++++==========
def process_tweet_phase2(tweet):    
    #tweet = convert_numbers(tweet)    
    tweet = convert_chat_words(tweet)
    tweet = expend_contractions(tweet)                                           
    tweet = tweet.lower()                                             # Lowercases the string
    tweet = re.sub(r"\d+", " ", str(tweet))                           # Removes all digits
    tweet = re.sub('"'," ", str(tweet))                               # Remove (") 
    tweet = re.sub(r"\b[a-zA-Z]\b", "", str(tweet))                   # Removes all single characters
    tweet = re.sub(r"[^\w\s]", " ", str(tweet))                       # Removes all punctuations
    tweet = re.sub(r'(.)\1+', r'\1\1', str(tweet))                    # Convert more than 2 letter repetitions to 2 letter
    tweet = re.sub(r"\s+", " ", str(tweet))                           # Replaces double spaces with single space    
    tweet = re.sub(r"\b[a-zA-Z]\b", "", str(tweet))                   # Removes all single characters
    tweet = remove_maximum(tweet)
    return tweet
# +++++++++++++++++++++++++++++++++++++++++++++
# PORTESTEMMER STEMMING
def porter_stemmer(data):
    stemmer = PorterStemmer()
    tokens = word_tokenize(data)
    stem_text = ''    
    for word in tokens:
        stem = stemmer.stem(word)
        stem_text = stem_text+' '+stem
    
    return stem_text

# =========================================================================================== 
# load the model
#file_name = 'simple_model_stem.pkl'
#model = pickle.load(open(file_name,'rb'))
file_name = 'simple_model_stem.joblib'
model = load( file_name )

# create the input schema using pydantic basemodel
# Pydantic models are structures that ingest the data, parse it and make sure it conforms 
# to the fields’ constraints defined in it
class Tweet(BaseModel):
    tweet : str
    
# create FastAPI instance
app = FastAPI(
    title="Sentiment Model API",
    description="A simple API that use NLP model to predict the sentiment of a tweet"
            )

# create routes
# home route(/) 
@app.get("/")
def read_root():
    return {"msg":'TWEET SENTIMENT'}

# predict route
@app.get("/predict_tweet")
async def predict_tweet(tweet: Tweet):
    #data = input.dict()
    # clean the tweet
    #cleaned_text = process_tweet_phase1(data['Tweet'])
    cleaned_text = process_tweet_phase1(tweet)
    cleaned_text = process_tweet_phase2(cleaned_text)
    cleaned_text = porter_stemmer(cleaned_text)
    # prediction
    prediction = model.predict([cleaned_text])
    # output
    output = int(prediction[0])
    # output dictionary
    sentiments = {0: "Negative", 1: "Positive"}

    return {
        'prediction': sentiments[output]
        }

#if __name__=="__main__":
#    uvicorn.run(app, host="127.0.0.1", port=8000)


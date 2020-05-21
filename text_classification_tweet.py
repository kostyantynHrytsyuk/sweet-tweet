from nltk import pos_tag
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import TweetTokenizer, word_tokenize
import pandas as pd
import re
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn import metrics, svm, naive_bayes
from string import punctuation


def get_data_from_file(file_path, head_names=None):
    with open(file_path, encoding='utf8', errors='ignore') as f:
        data = pd.read_csv(f, dtype={'title': str, 'description': str},
                           engine='python', header=None)
        return data


def get_clean_tweets(tweets):
    clean_t = []
    stop_words = ['REM_URL', 'REM_USER'] + stopwords.words('english') + list(punctuation) + list('...')

    for i,j in tweets.iterrows():
        t = j[1]
        # Making all text in tweet lowercase
        t = t.lower()
        # Cleaning out links
        t = re.sub('((www\.[^\s]+)|(https?://[^\s]+))', 'REM_URL', t)

        # Removing Twitter attributes - # symbol near hashtags and mentions
        t = re.sub(r'#([^\s]+)', r'\1', t)
        t = re.sub('@[^\s]+', 'REM_USER', t)

        # Tokenizing tweet
        t = word_tokenize(t)
        t = [w for w in t if w not in stop_words]

        # Remove repeating characters in words
        repeat_pattern = re.compile(r'(\w)\1*')
        t = [repeat_pattern.sub(r'\1', w) for w in t]

        # Removing word with length less than 3
        t = [w for w in t if len(w) >= 3]
        clean_t.append((t, j[0]))

    return pd.DataFrame(clean_t)


def fix_polarity(p):
    if p == 0:
        return 'negative'
    elif p == 2:
        return 'neutral'
    else:
        return 'positive'


def text_pre_processing(df):
    # Giving columns valuable names
    df = df.rename(columns={0: 'polarity',
                            1: 'tweet_id',
                            2: 'tweet_date',
                            3: 'query',
                            4: 'username',
                            5: 'tweet_text'})

    # Selecting column with sentiment and text of tweet
    df = df[['polarity', 'tweet_text']]

    df['polarity'] = df['polarity'].map(fix_polarity)

    # Data cleaning and text pre-processing

    # Removing empty rows
    df['tweet_text'].dropna(inplace=True)

    return get_clean_tweets(df)


df = get_data_from_file('training.1600000.processed.noemoticon.csv')
#nltk.download('popular')

train_tweets = text_pre_processing(df)
train_tweets = train_tweets.rename(columns={0: 'tweet_text',
                                            1: 'polarity'})

tag_map = {'N': wordnet.NOUN, 'J': wordnet.ADJ, 'V': wordnet.VERB, 'R': wordnet.ADV}

for i, tweet in train_tweets.iterrows():
    clean_words = []

    words_lem = WordNetLemmatizer()
    for word, tag in pos_tag(tweet[0]):
        if word.isalnum():
            k = tag_map.get(tag[0])
            if not k:
                k = tag_map.get('N')
            w = words_lem.lemmatize(word, k)
            clean_words.append(w)
    train_tweets.loc[i, 'tweet_text'] = str(clean_words)

X_train, X_test, y_train, y_test = train_test_split(
    train_tweets['tweet_text'], train_tweets['polarity'],
    test_size=0.3)

encode = LabelEncoder()

# TF-IDF

tfidf_vec = TfidfVectorizer()
tfidf_vec.fit(train_tweets['tweet_text'])

X_train_tfidf = tfidf_vec.transform(X_train)
X_test_tfidf = tfidf_vec.transform(X_test)

# NB

NB_clf = naive_bayes.MultinomialNB()
NB_clf.fit(X_train_tfidf, y_train)

NB_predict = NB_clf.predict(X_test_tfidf)
print('Manual Vectorizer NB', round(metrics.accuracy_score(NB_predict, y_test)*100), '%')

# SVM

SVM_clf = svm.SVC(kernel='linear', gamma='auto')
SVM_clf.fit(X_train_tfidf, y_train)

SVM_predict = SVM_clf.predict(X_test_tfidf)

print('Manual Vectorizer SVM', round(metrics.accuracy_score(SVM_predict, y_test)*100), '%')

# Automated

cv = CountVectorizer(lowercase=True, stop_words='english', ngram_range=(1,1),
                     tokenizer=TweetTokenizer().tokenize)

text_counts = cv.fit_transform(df.loc[:, 5])

X_train, X_test, y_train, y_test = train_test_split(
    text_counts, df.loc[:, 0], test_size=0.1)

# NB

NB_clf = naive_bayes.MultinomialNB()
NB_clf.fit(X_train, y_train)

NB_predict = NB_clf.predict(X_test)
print('Count Vectorizer NB', round(metrics.accuracy_score(NB_predict, y_test)*100) ,'%')

# SVM

SVM_clf = svm.SVC(kernel='linear', gamma='auto')
SVM_clf.fit(X_train, y_train)

SVM_predict = SVM_clf.predict(X_test)

print('Count Vectorizer SVM', round(metrics.accuracy_score(SVM_predict, y_test)*100), '%')

# Dirty

text_tf = TfidfVectorizer().fit_transform(df.loc[: , 5])

X_train, X_test, y_train, y_test = train_test_split(
    text_tf, df.loc[:, 0], test_size=0.1)

# NB

NB_clf = naive_bayes.MultinomialNB()
NB_clf.fit(X_train, y_train)

NB_predict = NB_clf.predict(X_test)
print('TF-IDF Vectorizer NB', round(metrics.accuracy_score(NB_predict, y_test)*100), '%')

# SVM

SVM_clf = svm.SVC(kernel='linear', gamma='auto')
SVM_clf.fit(X_train, y_train)

SVM_predict = SVM_clf.predict(X_test)

print('TF-IDF Vectorizer SVM', round(metrics.accuracy_score(SVM_predict, y_test)*100), '%')

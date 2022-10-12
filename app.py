from sre_parse import Tokenizer
import streamlit as st 
import pandas as pd 
import numpy as np
import plotly.express as px
import seaborn as sns
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import string
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
# from sklearn.feature_extraction.text TfidTransformer

def stem_words(text):
    stemmer = PorterStemmer()
    text = [stemmer.stem(word) for word in text]
    return text

def remove_puntuation(tokens):
    text = [word for word in tokens if word not in string.punctuation]
    return text

def tokenise(text):
    tokens = nltk.word_tokenize(text)
    return tokens

def remove_stopwords(tokens):
    stop_words = set(nltk.corpus.stopwords.words('english'))
    filtered_tokens = [w for w in tokens if not w in stop_words]
    return filtered_tokens

@st.cache(allow_output_mutation=True)
def load_data():
    df = pd.read_csv('data/anthems.csv')
    df.drop(['Alpha-2', 'Alpha-3'], axis=1, inplace=True)
    return df

def countplot_px(df, x, y, title):
    fig = px.bar(df, x=x, y=y, title=title)
    st.plotly_chart(fig)

def main():

    st.set_page_config(page_title='National anthem classifier', page_icon='🎶', layout='wide', initial_sidebar_state='auto')
    st.title('National anthem classifier')

    selected_country = st.selectbox(
        label='Search anthem by country',
        options=load_data()['Country'].unique(),
        key='select_country'
    )
    anthem_display_button = st.button('Display anthem',key='anthem_display_button')
    anthems = load_data()
    anthems['length'] = anthems['Anthem'].apply(len)
    if anthem_display_button:
        st.write(anthems.iloc[anthems[anthems['Country'] == selected_country].index[0]]['Anthem'])


    
    c1,c2 = st.columns(2)
    with c1:
        c1.metric(label='Number of countries', value=load_data()['Country'].nunique(),delta=0)
        st.plotly_chart(px.bar(x=anthems.groupby('Continent').count()['Country'],
        title='Number of countries per continent'),labels={'x':'Continent', 'y':'Number of countries'},
        width=800, height=500,
        use_container_width=True,
        group_label='Continent'
            
        )

    with c2:
        c2.metric(label='Number of anthems', value=load_data()['Anthem'].nunique(),delta=0)
        length_country_fig = px.bar(anthems, y='length', x='Country', title='Length of anthem by country',labels={'x':'Country', 'y':'Length of anthem'})
        st.plotly_chart(length_country_fig)

    c3,c4 = st.columns(2)
    with c3:
        st.plotly_chart(px.bar(anthems, x='Continent', y='Country', color='Continent', title='Countries by continent'))
        df = px.data.medals_long()

    with c4:
        st.plotly_chart(px.bar(anthems, x='Continent', y='length', color='Continent', title='Length of anthem by continent'))
        df = px.data.medals_long()

    st.subheader('anthmes with tokenisation')
    
    tokenise_anthmes = anthems['Anthem'].apply(tokenise)
    tokenise_anthmes = tokenise_anthmes.apply(remove_puntuation)
    tokenise_anthmes = tokenise_anthmes.apply(remove_stopwords)
    tokenise_anthmes = tokenise_anthmes.apply(stem_words)
    tokenise_anthmes = tokenise_anthmes.apply(lambda x: ' '.join(x))
    anthems['tokenised_anthems'] = tokenise_anthmes
    st.write(anthems)

    # pipeline = Pipeline([
    #     ('bow', CountVectorizer()),
    #     ('tfidf', TfidTransformer()),
    #     ('classifier', MultinomialNB())
    # ])

    # X = anthems['tokenised_anthems']
    # y = anthems['Continent']

    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # pipeline.fit(X_train, y_train)

    # predictions = pipeline.predict(X_test)

    # st.write(classification_report(y_test, predictions))
    
    
   

if __name__ == '__main__':
    main()
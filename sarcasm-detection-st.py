import pickle
from tensorflow.python.keras.preprocessing.text import Tokenizer
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from nltk.tokenize import word_tokenize
import pandas as pd
import re
import nltk
import string
from nltk.corpus import stopwords
from keras.models import load_model

nltk.download('punkt')
nltk.download('stopwords')

model = load_model("my_model1.h5")

def CleanTokenize(df):
    head_lines = list()
    lines = df["headline"].values.tolist()
    
    for line in lines:
        line = clean_text(line)
        # tokenize the text
        tokens = word_tokenize(line)
        # remove puntuations
        table = str.maketrans('', '', string.punctuation)
        stripped = [w.translate(table) for w in tokens]
        # remove non alphabetic characters
        words = [word for word in stripped if word.isalpha()]
        stop_words = set(stopwords.words("english"))
        # remove stop words
        words = [w for w in words if not w in stop_words]
        head_lines.append(words)
    return head_lines

def clean_text(text):
    text = text.lower()
    
    pattern = re.compile('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
    text = pattern.sub('', text)
    text = " ".join(filter(lambda x:x[0]!='@', text.split()))
    emoji = re.compile("["
                           u"\U0001F600-\U0001FFFF"  # emoticons
                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                           u"\U0001F680-\U0001F6FF"  # transport & map symbols
                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           u"\U00002702-\U000027B0"
                           u"\U000024C2-\U0001F251"
                           "]+", flags=re.UNICODE)
    
    text = emoji.sub(r'', text)
    text = text.lower()
    text = re.sub(r"i'm", "i am", text)
    text = re.sub(r"he's", "he is", text)
    text = re.sub(r"she's", "she is", text)
    text = re.sub(r"that's", "that is", text)        
    text = re.sub(r"what's", "what is", text)
    text = re.sub(r"where's", "where is", text) 
    text = re.sub(r"\'ll", " will", text)  
    text = re.sub(r"\'ve", " have", text)  
    text = re.sub(r"\'re", " are", text)
    text = re.sub(r"\'d", " would", text)
    text = re.sub(r"\'ve", " have", text)
    text = re.sub(r"won't", "will not", text)
    text = re.sub(r"don't", "do not", text)
    text = re.sub(r"did't", "did not", text)
    text = re.sub(r"can't", "can not", text)
    text = re.sub(r"it's", "it is", text)
    text = re.sub(r"couldn't", "could not", text)
    text = re.sub(r"have't", "have not", text)
    text = re.sub(r"[,.\"\'!@#$%^&*(){}?/;`~:<>+=-]", "", text)
    return text



with open('tokenizer.pickle', 'rb') as handle:
    tokenizer_obj = pickle.load(handle)




def predict_sarcasm(s):
    x_final = pd.DataFrame({"headline":[s]})
    print(x_final)
    test_lines = CleanTokenize(x_final)
    print(test_lines)
    test_sequences = tokenizer_obj.texts_to_sequences(test_lines)
    print(test_sequences)
    test_review_pad = pad_sequences(test_sequences, maxlen=25, padding='post')
    print(test_review_pad)
    pred = model.predict(test_review_pad)
    pred*=100
    print("value: " + str(pred))
    if pred[0][0]>=50: return "Sarcastic" 
    else: return"Not Sarcastic"

predict_sarcasm("The engine feels good. Much faster than before. Amazing")
print("running")


import streamlit as st

st.set_page_config(page_title='Satirical Headlines Detector')


st.markdown(""" <style> .head {
font-size:70px ; color: #FF9633;
text-align: center;
margin-top:0px;
margin-bottom:5px;
} 
</style> """, unsafe_allow_html=True)

st.markdown(""" <style> .text1 {
font-size:20px ; color: #FF9633;
margin-bottom: 0;
} 
</style> """, unsafe_allow_html=True)

st.markdown("<h2 class='head'>Satirical Headline Detector</div>", unsafe_allow_html=True)

st.markdown("<p class='text1'>Enter Headline</p>", unsafe_allow_html=True)
hl = st.text_input('')

st.markdown("<br>", unsafe_allow_html=True)

if hl:
    txt1 = f"""<p class='text1'>The current headline is: <b><i>{hl.upper()}</i></b></p>"""
    st.markdown(txt1, unsafe_allow_html=True)

st.markdown("<br><br>", unsafe_allow_html=True)

st.markdown(""" <style> .sar {
font-size:25px ; color: red;
margin-bottom: 0;
} 
</style> """, unsafe_allow_html=True)

st.markdown(""" <style> .nsar {
font-size:25px ; color: green;
margin-bottom: 0;
} 
</style> """, unsafe_allow_html=True)

if hl:
    output = predict_sarcasm(hl)
    if output == "Sarcastic":
        txt2 = f"""<p class='sar'>✖ The headline is <b><i>SATIRE</i></b></p>"""
        st.markdown(txt2, unsafe_allow_html=True)
    else:
        txt2 = f"""<p class='nsar'>✓ The headline is <b><i>NOT SATIRE</i></b></p>"""
        st.markdown(txt2, unsafe_allow_html=True)
else:
    txt3 = f"""<p class='text1'>Enter news headline to check whether it is satire or not</p>"""
    st.markdown(txt3, unsafe_allow_html=True)
st.markdown("<br><br>", unsafe_allow_html=True)

col1, col2, col3 , col4, col5 = st.columns(5)

with col1:
    pass
with col2:
    pass
with col4:
    pass
with col5:
    pass
with col3 :
    center_button = st.button('About the project')

about = f"""<br><div><p class='text1' style='text-align:justify;'>Type a news headline in the textbox
to check whether it is satirical or not. The prediction is performed 
with the help of a natively built  LSTM model (RNN) that is 
trained with thousands of labelled news headlines from The HuffPost and The Onion.</p>
  
  <p class='text1'>This website is built by Sanjit Kapoor using Streamlit</p>"""

if center_button:

    st.markdown(about, unsafe_allow_html=True)
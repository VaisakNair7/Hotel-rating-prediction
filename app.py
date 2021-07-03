from pywebio.platform.flask import webio_view
from pywebio import STATIC_PATH
from flask import Flask, send_from_directory
from pywebio.input import textarea
from pywebio.output import *
from pywebio import start_server
import argparse

import nltk
import numpy as np
from nltk.corpus import stopwords
import re
from nltk.stem.porter import PorterStemmer
#nltk.download('stopwords')

app = Flask(__name__)

stopword = stopwords.words('english')
execption = ['they','them','their','theirs','what','which','who','why', 'how','when','where','how', 'few', 'no','nor','not','can',
             'will',"don't",'should', "should've","aren't",'couldn','only','too','very','don',"aren't",'couldn','but','did',"couldn't",
             "didn't",'doesn',"doesn't", 'hadn',"hadn't",'hasn',"hasn't",'haven',"haven't",'isn',"isn't",'mightn',"mightn't",
             'mustn',"mustn't",'needn', "needn't","shan't",'shouldn',"shouldn't",'wasn',"wasn't",'weren',"weren't","won't",'wouldn',
             "wouldn't",'ain','aren', 'didn', 'don']
stop = [x for x in stopword if x not in execption]

# Load the Pipeline
import joblib
model = joblib.load('tfidf_model')

# Used for validating input in textarea
def validateInput(x):
    if x.isspace() or x.isnumeric():
        return 'Please enter a valid review.'


def predict():

    # About.
    put_collapse('About', put_tabs([
        {'title':'How it works', 'content': 'This model is trained on 20k reviews crawled from Tripadvisor using TFIDFVectorizer and Logistic Regression. The front end is created using PyWebIO which is a Python library that allows you to build simple web applications with minimal use of HTML and Javascript.'},
        {'title':'Contact', 'content': [  
            put_link('LinkedIn', url = 'https://www.linkedin.com/in/vaisaksnair/', new_window = True),
            put_html('<br/>'),
            put_link('GitHub', url = 'https://github.com/VaisakNair7/Hotel-rating-prediction/tree/main', new_window = True),
            put_text('Mail - vaisaksnair98@gmail.com')
        ]}
    ]))

    put_html('<br/>')
    put_html('<br/>')
    
    # Get input in textarea
    review = textarea('Hotel review(1-5) prediction based on review', validate = validateInput, 
    placeholder = 'Enter your review here.', required = True)
    
    # Clean the text
    ps = PorterStemmer()
    text = re.sub('[^a-zA-Z]',' ', review)
    text = text.lower()
    text = text.split()
    text = [ps.stem(x) for x in text if x not in stop]
    text = ' '.join(text)

    # Prediction
    result = model.predict_proba([text])

    one = round(result[0][0] * 100, 2)
    two = round(result[0][1] * 100, 2)
    three = round(result[0][2] * 100, 2)
    four = round(result[0][3] * 100, 2)
    five = round(result[0][4] * 100, 2)

    put_markdown('# Here are the predictions!')
    put_html('<br/>')
    put_markdown('Your review : %r' % review)
    put_html('<br/>')
    put_markdown('## Ratings')
    put_html('<br/>')

    # processbar is used to display progress bar representing the probabilities of predcitions.
    put_text('1 / 5 :')
    put_processbar('bar1');
    set_processbar('bar1', one / 100)

    put_text('')
    put_text('2 / 5 :')
    put_processbar('bar2');
    set_processbar('bar2', two / 100)

    put_text('')
    put_text('3 / 5 :')
    put_processbar('bar3');
    set_processbar('bar3', three / 100)

    put_text('')
    put_text('4 / 5 :')
    put_processbar('bar4');
    set_processbar('bar4', four / 100)

    put_text('')
    put_text('5 / 5 :')  
    put_processbar('bar5');
    set_processbar('bar5', five / 100)


app.add_url_rule('/', 'webio_view', webio_view(predict), methods = ['GET', 'POST', 'OPTIONS'])

if __name__ == '__main__':                            #comment this block to run in your localhost
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--port", type=int, default = 8080)
    args = parser.parse_args()

    start_server(predict, port = args.port)
    
#app.run(host = 'localhost', port = 80, debug = True) #uncomment this to run in your localhost

    










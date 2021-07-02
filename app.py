from pywebio.platform.flask import webio_view
from pywebio import STATIC_PATH
from flask import Flask, send_from_directory
from pywebio.input import textarea
from pywebio.output import *
from pywebio import start_server

import nltk
import numpy as np
from nltk.corpus import stopwords
import re
from nltk.stem.porter import PorterStemmer

app = Flask(__name__)

stopword = stopwords.words('english')
execption = ['they','them','their','theirs','what','which','who','why', 'how','when','where','how', 'few', 'no','nor','not','can',
             'will',"don't",'should', "should've","aren't",'couldn','only','too','very','don',"aren't",'couldn','but','did',"couldn't",
             "didn't",'doesn',"doesn't", 'hadn',"hadn't",'hasn',"hasn't",'haven',"haven't",'isn',"isn't",'mightn',"mightn't",
             'mustn',"mustn't",'needn', "needn't","shan't",'shouldn',"shouldn't",'wasn',"wasn't",'weren',"weren't","won't",'wouldn',
             "wouldn't",'ain','aren', 'didn', 'don']
stop = [x for x in stopword if x not in execption]

import joblib
model = joblib.load('tfidf_model')

def validateInput(x):
    if x.isspace() or x.isnumeric():
        return 'Please enter a valid review.'


def predict():

    review = textarea('Hotel review(1-5) prediction based on review', validate = validateInput, 
    placeholder = 'Enter your review here.', required = True)

    ps = PorterStemmer()
    text = re.sub('[^a-zA-Z]',' ', review)
    text = text.lower()
    text = text.split()
    text = [ps.stem(x) for x in text if x not in stop]
    text = ' '.join(text)

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

app.run(host='localhost', port=80, debug = True)

    










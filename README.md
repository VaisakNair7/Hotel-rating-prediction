# Hotel rating prediction.

### Website link - https://hotel-rating-prediction.herokuapp.com/
 
The aim of this project is to predict rating(1-5) based on user input hotel review.

It is trained on Kaggle dataset 'tripadvisor_hotel_reviews.csv' which contains 20k hotel reviews extracted from Tripadvisor.
### Dataset link - (https://www.kaggle.com/andrewmvd/trip-advisor-hotel-reviews)

### Demo Screenshot
![hotel](https://user-images.githubusercontent.com/37840005/134374413-1a7aef08-aa67-4c5e-b852-c1e5d87a4bd7.PNG)

This model uses TFIDFVectorizer to vectorize words and Logistic Regression (OnevsRest) for multiclass classification which had roc auc score of 0.87 and is deployed on Heroku cloud platform.

To run this project on your local machine first install the libraries present in requirements.txt using the command (pip install -r requirements.txt) and then type (python app.py) in the terminal. Make sure to comment and uncomment the blocks in app.py as instructed in the file before running it.

The front end is created using Flask and PyWebIO library. PyWebIO provides a series of imperative functions to obtain user input and output on the browser, turning the browser into a “rich text terminal”. It allows you to build simple web applications without the knowledge of HTML and Javascript.



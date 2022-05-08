# Room-Booking-Chatbot

Very Intelligent Booking Chat Interface. It can answer your related queries on rooms, booking, requests and hotels.

## Classification Algorithm

For a training data of less than 1k size, using neural nets or sequential models seemed to be an overkill, so we started with    basic featurisation technique — TfidfVectorizer. We also decided to contest Tf-Idf against average taken over the GloVe(Global Vectors for Word Representations) for each of the words in the questions. We used LabelEncoder to convert the intents to numbers or labels.

For having less training data was that, we were free to explore all types of classifiers like Logistic Regression, k-Nearest Neighbours, Naive Bayes, SVM, SGD Classifier and XGBoost. We also carried out extensive experiments to fine-tune the hyperparameters and achieve their best configuration.

![Image](https://github.com/basel-ay/Room-Booking-Chatbot/blob/main/static/screenshots/Classifier%20Scores.png)

Clearly, SVM and Logistic Regression are the top two classifiers on this training data. Although SVM has the best F1 score and test score, its mean cross-validation score is very low. This shows that it has extreme behaviour on this data. Hence we go for logistic regression.

## Setup

1. Download GloVe vectors from [this](https://nlp.stanford.edu/data/glove.6B.zip) link. Unzip and keep the file **glove.6B.100d.txt** in **models** folder.
2. Run ``` conda env create -f environment.yml ```
3. Run ``` conda activate botEnv ```
4. Run the bot flask application by ``` python app.py ```
5. If you face any issues with the existing models, you can train afresh by deleting the **.joblib** files in the **models** folder and run ``` python botmodel.py ``` 

![Image1](https://github.com/basel-ay/Room-Booking-Chatbot/blob/main/static/screenshots/shot1.png)
![Image2](https://github.com/basel-ay/Room-Booking-Chatbot/blob/main/static/screenshots/shot2.png)
![Image3](https://github.com/basel-ay/Room-Booking-Chatbot/blob/main/static/screenshots/shot3.png)

# Room-Booking-Chatbot

Very Intelligent Booking Chat Interface. It can answer your related queries on rooms, booking, requests and hotels.

## Setup

1. Download GloVe vectors from [this](https://nlp.stanford.edu/data/glove.6B.zip) link. Unzip and keep the file **glove.6B.100d.txt** in **models** folder.
2. Run ``` conda env create -f environment.yml ```
3. Run ``` conda activate botEnv ```
4. Run the bot flask application by ``` python app.py ```
5. If you face any issues with the existing models, you can train afresh by deleting the **.joblib** files in the **models** folder and run ``` python botmodel.py ``` 

![Image1](https://github.com/basel-ay/Room-Booking-Chatbot/blob/main/static/screenshots/shot1.png)
![Image2](https://github.com/basel-ay/Room-Booking-Chatbot/blob/main/static/screenshots/shot2.png)
![Image3](https://github.com/basel-ay/Room-Booking-Chatbot/blob/main/static/screenshots/shot3.png)

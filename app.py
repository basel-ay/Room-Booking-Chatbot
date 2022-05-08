import os
from flask import Flask, render_template, request, jsonify

import utilities as ut
from botmodel import BotModel

# bot variables to be stored in Database
user_name = None
user_email = None
user_phone = None
number_of_rooms = None
room_type = None
checkin = None
checkout = None
special_requests = None
feedback = None

app = Flask(__name__)

class VicciBot(object):
    '''   This is the main bot engine class   '''

    def __init__(self):
        self.botmodel = BotModel()
        self.botmodel.initialize()

    def response(self, user_query):
        user_query = user_query.strip().lower()
        bot_responses, served = [], False
        global user_name, user_email, user_phone, checkin, checkout, room_type, special_requests, feedback, number_of_rooms
        tag, conf, resp = self.botmodel.response(user_query)

        if conf < 0.3:
            # Too low confidence of the intent classifier 
            bot_responses.append("Sorry. I did not understand your query.")

        else:
            if tag=='name':
                try:
                    user_name = ut.extract_names(user_query)
                
                    try:
                        get_user_name = f'Hello {user_name}. How are you?'
                        bot_responses.append(get_user_name)
                    except:
                        bot_responses.append('Error! Please try again')

                except:
                    bot_responses.append('Error! Please type your name again')

            elif tag=='check_IN_OUT':
                if user_name != None:
                    try:
                        checkin, checkout = ut.extract_dates(user_query)

                        try:
                            bot_responses.append(f'Very well. Checkin: {checkin} Checkout: {checkout}')
                            bot_responses.append('And we allow only 2 adults in a room.')
                            bot_responses.append('Can you give me your phone number and email? phone: 0000000000 and email: xxxxx@XXXXX.com')
                        except:
                            bot_responses.append('Error! Please try again')

                    except:
                        bot_responses.append('Error! Please type dates again')
                else:
                       bot_responses.append('Please tell me your name first!')

            elif tag=='confirm':
                try:
                    lower_case_response = user_query.lower()
                    if 'yes' in lower_case_response :
                        bot_responses.append('Type "book room"')
                    elif 'no' in lower_case_response:
                        bot_responses.append('Thank you. Please leave a feedback with "Feedback." word as prefix.')
                    
                except:
                    bot_responses.append('Error! Please try again')

            elif tag=='phone_number_and_email':
                if user_name != None:
                    if checkin != None and checkout != None:
                        try:
                            user_phone = ut.extract_phone_numbers(user_query)
                            user_email = ut.extract_email_addresses(user_query)
                            bot_responses.append('How many rooms do you want to book?')
                        except:
                            bot_responses.append('Please try again')
                    else:
                        bot_responses.append('Please type checkin and checkout first!')
                else:
                       bot_responses.append('Please tell me your name first!')

            elif tag=='number_of_rooms':
                if user_name != None:
                    if checkin != None and checkout != None:
                        if user_phone != None and user_email != None :
                            try:
                                number_of_rooms = user_query
                                bot_responses.append('What type of room you want?')
                            except:
                                bot_responses.append('Error! Please try again') 
                        else:
                            bot_responses.append('Please type your phone and email first!')
                    else:
                        bot_responses.append('Please type checkin and checkout first!')
                else:
                       bot_responses.append('Please tell me your name first!')

            elif tag=='room_type':
                if user_name != None:
                    if checkin != None and checkout != None:
                        if user_phone != None and user_email != None :
                            if number_of_rooms != None:
                                try:
                                    room_type = user_query
                                    bot_responses.append('Any special requests you want to add?')
                                    bot_responses.append('Type "Special." word as a prefix then type what you want.')
                                except:
                                    bot_responses.append('Error! Please try again') 
                            else:
                                bot_responses.append('Please type number of rooms first!')        
                        else:
                            bot_responses.append('Please type your phone and email first!')
                    else:
                        bot_responses.append('Please type checkin and checkout first!')
                else:
                       bot_responses.append('Please tell me your name first!')

            elif tag=='special_requests':
                        if user_name != None:
                            if checkin != None and checkout != None:
                                if user_phone != None and user_email != None :
                                    if room_type != None:
                                        if number_of_rooms != None:
                                            try:
                                                special_requests = user_query
                                                bot_responses.append('Done!')
                                                bot_responses.extend([f'Name: {user_name}', f'From: {checkin} To: {checkout}\n', f'Number of rooms: {number_of_rooms}',f'Room Type: {room_type}', f'Phone: {user_phone}', f'Email: {user_email}', f'Special: {special_requests}'])
                                                bot_responses.append('Thank you. Please leave a feedback with "Feedback." word as prefix.')
                                            except:
                                                bot_responses.append('Error! Please try again')
                                        else:
                                            bot_responses.append('Please type number of rooms first!')
                                    else:
                                        bot_responses.append('Please type room type first!')        
                                else:
                                    bot_responses.append('Please type your phone and email first!')
                            else:
                                bot_responses.append('Please type checkin and checkout first!')
                        else:
                            bot_responses.append('Please tell me your name first!')     

            elif tag=='feedback':
                feedback = user_query
                bot_responses.append('Thanks! Your feedback saved!')
            else:
                bot_responses.extend(resp)

        
        return jsonify(bot_responses)


vbot = VicciBot()

def lastUpdateTime(folder):
    # This function returns the latest last updated timestamp of all the static files 
    return str(max(os.path.getmtime(os.path.join(root_path, file)) \
        for root_path, dirs, files in os.walk(folder) \
            for file in files))

@app.route("/")
def home():
    return render_template("home.html", last_updated=lastUpdateTime('static/'))

@app.route("/get")
def get_bot_response():    
    user_query = request.args.get('user_query')    
    return vbot.response(user_query)

if __name__=='__main__':
    app.run(debug=False)
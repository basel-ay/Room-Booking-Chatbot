import re
import random
import nltk
from nltk.corpus import stopwords
stop = stopwords.words('english')

string = 'phone: 222248 email: x@.com'

def extract_phone_numbers(string):

    r = re.compile(
        r'\w\d \w\w \w\w \w\w \w\d|(?<=[^\d][^_][^_] )[^_]\d[^ ]\d[^ ][^ ]+|(?<= [^<]\w\w \w\w[^:]\w[^_][^ ][^,][^_] )(?: *[^<]\d+)+')
    phone_numbers = r.findall(string)
    return [re.sub(r'\D', '', number) for number in phone_numbers][0] # min 6 digits

def extract_email_addresses(string):
    r = re.compile(r'[\w\.-]+@[\w\.-]+')
    return r.findall(string)[0]

def extract_dates(string):
        dates = re.findall(r'\d+\S\d+\S\d+', string)
        if dates:
            if len(dates) == 2:
                return dates[0], dates[1]

def generate_room_number():
    return random.randint(3, 6)

def ie_preprocess(document):
    document = ' '.join([i for i in document.split() if i not in stop])
    sentences = nltk.sent_tokenize(document)
    sentences = [nltk.word_tokenize(sent) for sent in sentences]
    sentences = [nltk.pos_tag(sent) for sent in sentences]
    return sentences

def extract_names(document):
    names = []
    document = document.title()
    sentences = ie_preprocess(document)
    for tagged_sentence in sentences:
        for chunk in nltk.ne_chunk(tagged_sentence):
            if type(chunk) == nltk.tree.Tree:
                if chunk.label() == 'PERSON':
                    names.append(' '.join([c[0] for c in chunk]))
    return names[0]

# print(extract_phone_numbers(string))
# print(extract_email_addresses(string))
# print(extract_dates(string))
# print(extract_names(string))



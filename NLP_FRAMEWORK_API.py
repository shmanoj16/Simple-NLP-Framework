from flask import Flask, jsonify,request
from flask_cors import CORS
import re
import sys
import pandas as pd
import numpy as np
import json
import requests
import urllib.parse
from textblob import TextBlob
import nltk
import spacy
from gensim.summarization import keywords
app = Flask(__name__)
CORS(app)


@app.route('/language_detector', methods=['GET','POST'])
def lang_detection():
    #Using TextBlob Library
    value = request.args.get('value')
    value=urllib.parse.quote(value)
    raw_text= TextBlob(f"{value}")
    languages_dict= {'af':'Afrikaans',
                 'sq':'Albanian',
                 'am':'Amharic',
                 'ar':'Arabic',
                 'hy':'Armenian',
                 'az':'Azerbaijani',
                 'eu':'Basque',
                 'be':'Belarusian',
                 'bn':'Bengali',
                 'bs':'Bosnian',
                 'bg':'Bulgarian',
                 'ca':'Catalan',
                 'ceb':'Cebuano','zh':'Chinese',
                 'zh-CN':'Chinese(Simplified)','zh-TW':'Chinese (Traditional)','co':'Corsican','hr':'Croatian','cs':'Czech','da':'Danish','nl':'Dutch','en':'English','eo':'Esperanto','et':'Estonian','fi':'Finnish','fr':'French','fy':'Frisian','gl':'Galician',
                 'ka':'Georgian','de':'German','el':'Greek','gu':'Gujarati','ht':'Haitian Creole','ha':'Hausa','haw':'Hawaiian','iw':'Hebrew','he':'Hebrew','hi':'Hindi','hmn':'Hmong','hu':'Hungarian','is':'Icelandic','ig':'Igbo','id':'Indonesian','ga':'Irish','it':'Italian','ja':'Japanese','jv':'Javanese','kn':'Kannada','kk':'Kazakh',
                 'km':'Khmer','ko':'Korean','ku':'Kurdish','ky':'Kyrgyz','lo':'Lao','la':'Latin','lv':'Latvian','lt':'Lithuanian','lb':'Luxembourgish','mk':'Macedonian','mg':'Malagasy','ms':'Malay','ml':'Malayalam','mt':'Maltese','mi':'Maori','mr':'Marathi',
                 'mn':'Mongolian','my':'Myanmar (Burmese)','ne':'Nepali','no':'Norwegian','ny':'Nyanja (Chichewa)','ps':'Pashto','fa':'Persian','pl':'Polish','pt':'Portuguese (Portugal, Brazil)','pa':'Punjabi','ro':'Romanian','ru':'Russian','sm':'Samoan','gd':'Scots Gaelic','sr':'Serbian','st':'Sesotho',
                 'sn':'Shona','sd':'Sindhi','si':'Sinhala (Sinhalese)','sk':'Slovak','sl':'Slovenian','so':'Somali','es':'Spanish','su':'Sundanese','sw':'Swahili','sv':'Swedish','tl':'Tagalog (Filipino)','tg':'Tajik','ta':'Tamil','te':'Telugu','th':'Thai','tr':'Turkish','uk':'Ukrainian','ur':'Urdu','uz':'Uzbek','vi':'Vietnamese','cy':'Welsh','xh':'Xhosa','yi':'Yiddish','yo':'Yoruba','zu':'Zulu'
                }
    language_name=(languages_dict[raw_text.detect_language()])
    lang_dict={'language':language_name}
    return jsonify({'data':lang_dict})
#Sample Request : 10.168.126.50:4545/language_detector?value=Ich möchte ein Bier  | Response: German
#Sample Request : 10.168.126.50:4545/language_detector?value=Jedno pivo, hvala/molim  | Response :  Croatian
#Sample Request : 10.168.126.50:4545/language_detector?value=nandri | Response : Tamil

@app.route('/sentiment_analysis', methods=['GET','POST'])
def sentiment_analysis():
    #Using TextBlob Library
    value = request.args.get('value')
    raw_text= TextBlob(f"{value}")
    sentiment = raw_text.sentiment.polarity
    if sentiment<0:
        confidence=-(sentiment*100)
        sentiment='Negative'
    elif sentiment>0:
        confidence=sentiment*100
        sentiment='Positive'
    else:
        confidence=100
        sentiment='Neutral'
    sentiment_value={'tag_name':sentiment,'confidence':confidence}
    return jsonify({'data':sentiment_value})
#Sample Request : 10.168.126.50:4545/sentiment_analysis?value=Hello World 
#Response : Neutral  , Confidence : 100

@app.route('/entity_extraction', methods=['GET','POST'])
def entity_extraction():
    #Using Spacy Library
    nlp= spacy.load('en')
    value = request.args.get('value')
    text=nlp(f"{value}")
    extracted_entity_list=[]
    entity=""
    for word in text.ents:
        extracted_entity=(word.text,word.label_)
        extracted_entity_list.append(extracted_entity)
        entity={'entity':extracted_entity_list}
    return jsonify({'data':entity})
#Sample Request : 10.168.126.50:4545/entity_extraction?value=Google situated in Seattle and started in 1998 by Larry page 

@app.route('/keyword_extraction', methods=['GET','POST'])
def keyword_extraction():
    #Using Gensim Library
    value = request.args.get('value')
    text=(f"{value}")
    entity={'keywords':keywords(text).split('\n')}
    return jsonify({'data':entity})
#Sample Request : 10.168.126.50:4545/keyword_extraction?value=Challenges in natural language processing frequently involve... speech recognition, natural language understanding, natural language... generation (frequently from formal, machine-readable logical forms),... connecting language and machine perception, dialog systems, or some... combination thereof. 
# Where 10.168.126.50 is your IP
if __name__ == '__main__':
    app.run(host='0.0.0.0',port=4545,debug=True)

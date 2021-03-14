import json,csv,re, time
import keras
import keras.preprocessing.text as kpt
from keras.preprocessing.text import Tokenizer
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import goslate
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize

# emojis: saif mohaamed https://saifmohammad.com/WebPages/ArabicSA.html

class CleanDataClass(object):
    """docstring for CleanDataClass"""
    def __init__(self):
        super(CleanDataClass, self).__init__()
        self.vocabulary = self.csvToDict(self.lexicfile)

    # dataset + vocabulary
    start = time.time()
    # datasetfile = "datasets/grid-export.csv"
    lexicfile = "datasets/vocabulaire.csv"

    # leave some poctuaton? why? "?!"
    numbers_pattern, numbers_replacement = (r'\d+', ' number ') #before must replace numbers with lettres
    ponctuation_pattern, ponctuation_replacement = (r'[^\w\s]', ' ')
    ponctuation_pattern2, ponctuation_replacement2 = (r'[\-]{2,}', '-')
    ponctuation_pattern3, ponctuation_replacement3 = (r'\s{1,}', ' ')
    tag_pattern, tag_replacement = (r'@\w+', '')
    link_pattern, link_replacement = (r'https://[\w\./]+', '')
    # to do: treat hashtags : try to seperate words: testHashtag == test Hashtag

    # remove all non chars
    symbole_pattern, symbole_replacement = (r'[^ا-يةه]', '')

    def csvToDict(self, file):
        posts = []
        mydict = csv.DictReader(open(file))
        result = [d for d in mydict]
        return result

    def translate(self, text):
        gs = goslate.Goslate()
        return gs.translate(text, 'ar')

    def treat(self, string):
        string = re.sub(self.tag_pattern, self.tag_replacement , string)
        string = re.sub(self.link_pattern, self.link_replacement , string)
        string = re.sub(self.numbers_pattern, self.numbers_replacement , string)
        string = re.sub(r'[?!]', 'i12' , string)
        string = re.sub(self.ponctuation_pattern, self.ponctuation_replacement , string)
        string = re.sub(self.ponctuation_pattern2, self.ponctuation_replacement2 , string)
        string = re.sub(self.ponctuation_pattern3, self.ponctuation_replacement3, string)

        # replace repeated lettres
        string = re.sub(r'(\w)\1+', r'\1', string)

        #translate
        string = self.translate(string)

        #remove repeated letters
        string = ''.join([string[i] for i in range(len(string)-1) if string[i+1]!= string[i]]+[string[-1]])
        string = re.sub(r'(.)\1+', r'\1', string) 

        # remove non chars
        string = re.sub(self.symbole_pattern, self.symbole_replacement, string)

        #tokenisation
        sent_tokenize_list = word_tokenize(string)
        #remove stop words
        stop_words = stopwords.words(['arabic'])
        string = (' ').join([word for word in sent_tokenize_list if word not in stop_words])



        search = ["أ","إ","آ","ة","_","-","/",".","،"," و "," يا ",'"',"ـ","'","ى","\\",'\n', '\t','&quot;','?','؟','!']
        replace = ["ا","ا","ا","ه"," "," ","","",""," و"," يا","","","","ي","",' ', ' ',' ',' ? ',' ؟ ',' ! ']
        
        #remove tashkeel
        p_tashkeel = re.compile(r'[\u0617-\u061A\u064B-\u0652]')
        string = re.sub(p_tashkeel,"", string)
        
        #remove longation
        p_longation = re.compile(r'(.)\1+')
        subst = r"\1\1"
        string = re.sub(p_longation, subst, string)
        
        string = string.replace('وو', 'و')
        string = string.replace('يي', 'ي')
        string = string.replace('اا', 'ا')
        
        for i in range(0, len(search)):
            string = string.replace(search[i], replace[i])
        
        #trim    
        string = string.strip()


        return string

    def setPosts(self):
        self.posts = self.csvToDict(self.datasetfile)

    def getSentences(self, voca=True):
        self.setPosts()
        for post in self.posts:
            post['Post'] = self.treat(post['Post'])
        return self.posts

    def getSentencesAsArray(self, posts):
        result = [self.treat(post).split(' ') for post in posts]
        return result

    def getMaxSequenceLength(self, data= []):
        return len(max(self.getSentencesAsArray(data)))

    def setVocabularyPolarity(self):
        self.lexicon = {}
        for word in self.vocabulary:
            p = 1 if word['Polarity'] == '--' else float(word['Polarity'].replace(',','.'))
            self.lexicon[word['Expression']] = p
            for x in word['Related words']:
                if not x in self.lexicon:
                    self.lexicon[x] = p
            for x in word['Different spellings']:
                if not x in self.lexicon:
                    self.lexicon[x] = p

    def getVocabularyPolarity(self):
        self.setVocabularyPolarity()
        return self.lexicon

    def countOverlapin(self):
        file = 'datasets/texts_all.txt'
        with open(file, 'rU') as f:
            results =[]
            lines = f.readlines()
            keys = re.sub('\n', '' , lines[0])
            keys = keys.split('\t')
            for rec in lines:
                rec = re.sub('\n', '' , rec)
                rec = rec.split('\t')
                d = {}
                for i in range(len(keys)):
                    d[keys[i]] = rec[i]
                results.append(d)
            total = len(lines)-1
            sa_emo=0
            sa_emo_age = 0
            sa_emo_topic = 0
            sa_emo_gender = 0
            sa_all =0
            for rec in results:
                if rec['polarityClass'] and rec['polarityClass']!= '--' and rec['EmotionClasse'] and rec['EmotionClasse']!= '--':
                    sa_emo +=1

                if rec['polarityClass'] and rec['polarityClass'] != '--' and rec['EmotionClasse'] and rec['EmotionClasse']!= '--' and rec['userAge'] and rec['userAge']!= '--':
                    sa_emo_age +=1


                if rec['polarityClass'] and rec['polarityClass']!= '--' and rec['EmotionClasse'] and rec['EmotionClasse']!= '--' and rec['Topic'] and rec['Topic']!= '--':
                    sa_emo_topic +=1


                if rec['polarityClass'] and rec['polarityClass']!= '--' and rec['EmotionClasse'] and rec['EmotionClasse']!= '--' and rec['gender'] and rec['gender']!= '--':
                    sa_emo_gender +=1


                if rec['polarityClass'] and rec['polarityClass']!= '--' and rec['EmotionClasse'] and rec['EmotionClasse']!= '--' and rec['gender'] and rec['gender']!= '--' and rec['Topic'] and rec['Topic']!= '--' and rec['userAge'] and rec['userAge']!= '--':
                    sa_all +=1

        print(sa_emo, sa_emo_age, sa_emo_gender, sa_emo_topic, sa_all)
        print(sa_emo*100/total, sa_emo_age*100/total, sa_emo_gender*100/total, sa_emo_topic*100/total, sa_all*100/total)
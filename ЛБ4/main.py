import pymorphy2
from matplotlib import pyplot
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from wordcloud import WordCloud
import nltk
nltk.download('stopwords')
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

punc_str = "!()-[]{};:@#$%^',.\|/*-<>_~'"  # Краткий сисок знаков пунктуации
morph = pymorphy2.MorphAnalyzer()  # Функция для лемматизации
raw_text = open(r'text.txt', encoding='utf-8-sig').read()  # Читаем текст из файла
sentences = sent_tokenize(raw_text, language='russian')  # Токенизируем по предложениям
stop_words = set(stopwords.words('russian'))  # Стоп-лист слов на русском


def lemmatize(text):  # Функция для Vectorizer - лемматизирует элементы
    words = word_tokenize(text, language='russian')  # Токенизация по словам
    filtered_words = list()
    # Если слово не в стоп-листе и не является знаком препинания
    for word in words:
        if (word not in stop_words) and (word not in punc_str):
            filtered_words.append(word)  # Оно проходит фильтрацию
    # Возвращаем лемму слова
    unique_lemms = list()
    for filt_word in filtered_words:
        p = morph.parse(filt_word)[0].normal_form # Находим лемму слова
        if p not in unique_lemms: #Если она не встречалась ранее - попадет на вывод
            unique_lemms.append(p)
    return unique_lemms #Возврат уникальных лемм


count_vectorizer = CountVectorizer()  #Функция векторизации
string = [''.join(line) for line in sentences]  # Проводим ее по предложениям
bag_of_words = count_vectorizer.fit_transform(string)  # Создаем массив предложений
feature_names = count_vectorizer.get_feature_names_out()  # Получаем имена столбцов - слов
print('Токен Векторизция')
print(pd.DataFrame(bag_of_words.toarray(), columns=feature_names[:50]))  #Выводим результата в датафрейме

count_vectorizer_lemms = CountVectorizer(tokenizer=lemmatize)  #Векторизация с предварительной лемматизцией
bag_of_words = count_vectorizer_lemms.fit_transform(string)
feature_names = count_vectorizer_lemms.get_feature_names_out()
print('Лемма Векторизация')
print(pd.DataFrame(bag_of_words.toarray(), columns=feature_names[:37]))

tfidf_vectorizer = TfidfVectorizer()  #TF-IDF векторизация
bag_of_words = tfidf_vectorizer.fit_transform(string)
feature_names = tfidf_vectorizer.get_feature_names_out()
print('TF-IDF Векторизция')
print(pd.DataFrame(bag_of_words.toarray(), columns=feature_names[:50]))

lemmed = lemmatize(raw_text)
text = ''
for word in lemmed:
    text += ' ' + word

cloud = WordCloud(collocations=False).generate(text)
pyplot.imshow(cloud)
pyplot.axis('off')
cloud.to_file('cloud.png')

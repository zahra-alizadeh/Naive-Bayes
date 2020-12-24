import csv
import requests
from bs4 import BeautifulSoup
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string

URL = 'https://mstajbakhsh.ir'
pagination = f'https://mstajbakhsh.ir/page/'


# get data from mstajbakhsh.ir
def scraping():
    fileWriter = csv.writer(open('datasetfCat.csv', mode='w', encoding='utf-8'), delimiter=',', quotechar='"',
                            quoting=csv.QUOTE_MINIMAL)
    fileWriter.writerow(['category', 'text'])
    nextPageURL = f'https://mstajbakhsh.ir/page/'
    page = requests.get(URL)
    content = BeautifulSoup(page.content, 'html.parser')
    pageCount = content.select('div.pagination-centered ul li')
    for i in range(2, len(pageCount) + 1):
        content = BeautifulSoup(page.content, 'html.parser')
        articlesLink = content.select('article div.post-actions a')
        for j in range(0, len(articlesLink)):
            url = articlesLink[j]['href']
            print(url)
            page = requests.get(url)
            content = BeautifulSoup(page.content, 'html.parser')
            postsCategories = content.select('article header.post-head ul.post-category a')
            text = content.find('article')
            text = text.text.strip().lower()
            text = cleanData(text)
            category = []
            for postCategories in postsCategories:
                category.append(postCategories.text.strip())

            fileWriter.writerow([category[0], text])
        nextPageURL = pagination + str(i)
        print(nextPageURL)
        page = requests.get(nextPageURL)


# clean and preprocess of data
def cleanData(data):
    tokens = word_tokenize(data)
    stopWords = set(stopwords.words('english'))
    stopWords.add("'\\n")
    stopWords.add("==")
    stopWords.add("mir")
    stopWords.add("saman")
    stopWords.add("m")
    stopWords.add("'m")
    stopWords.add("phd")
    stopWords.add("''")
    stopWords.add("’")

    tokens_without_sw = [word.lower() for word in tokens if not word in stopWords and not word in string.punctuation]
    text = " ".join(tokens_without_sw)
    return text


# train and test data
def training():
    data = pd.read_csv('datasetfCat.csv', encoding='utf-8')
    categories = data.category.value_counts()
    # print(categories)

    x_train, x_test, y_train, y_test = train_test_split(data.text, data.category, test_size=0.2)

    print(f'train data size : {len(x_train)}')
    print(f'test data size : {len(x_test)}')

    vectorizer = CountVectorizer(binary=True)
    x_train_vect = vectorizer.fit_transform(x_train)
    nb = MultinomialNB()
    model = nb.fit(x_train_vect, y_train)
    score = nb.score(x_train_vect, y_train)
    print(score)

    x_test_vect = vectorizer.transform(x_test)
    prediction = nb.predict(x_test_vect)
    # print(f'predicted categories : {prediction}')
    # print(f'test categories real value : {y_test}')

    print("Accuracy: {:.2f}%".format(accuracy_score(y_test, prediction) * 100))

    URL = input("Enter a url : ")
    page = requests.get(URL)
    content = BeautifulSoup(page.content, 'html.parser')

    data = cleanData(content.text)
    data_series = pd.Series(data)
    test_vect = vectorizer.transform(data_series)
    predictedValue = model.predict(test_vect)
    print(predictedValue)


# scraping()
training()

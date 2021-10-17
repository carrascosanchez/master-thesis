#!/usr/bin/env python
# coding: utf-8

# ## Master in Business Analytics &amp; Big Data - Master's thesis
# ### "Sales prediction and anomaly detection in times of COVID-19 using data from the stock market and the international press”
# 
# Full PDF can be read at this link:
# <br>
# https://www.linkedin.com/posts/sergio-carrasco-sanchez_final-masters-project-activity-6746471092820475904-YLCE

# ## Data loading and cleaning
# ### Sales data load

# In[1]:


get_ipython().system('pip install --upgrade pip')


# In[2]:


get_ipython().system('pip install pandas')

import pandas as pd


# We proceed to load sales data from CSV file.

# In[3]:


sales_data_df = pd.read_csv('./datasets/orders.csv')


# In[4]:


sales_data_df


# We remove the least interesting columns.

# In[5]:


sales_data_df = sales_data_df.drop(['id', 'user', 'payment_method', 'subtotal', 'discount', 'shipping_costs'], axis=1)


# In[6]:


sales_data_df = sales_data_df.rename({'total': 'sales'}, axis=1)


# In[7]:


sales_data_df


# We group the sales by day.

# In[8]:


def daily_sales(data):
    data = data.copy()
    data.date = data.date.apply(lambda x: str(x)[:-9])
    data = data.groupby('date')['sales'].sum().reset_index()
    data.date = pd.to_datetime(data.date)
    return data


# In[9]:


daily_sales_df = daily_sales(sales_data_df)


# In[10]:


daily_sales_df


# In[11]:


daily_sales_df = daily_sales_df.set_index(daily_sales_df.date)


# In[12]:


daily_sales_df = daily_sales_df.drop('date', axis=1)


# In[13]:


daily_sales_df


# ### Nasdaq Composite Index Data Load

# We import the  Composite index data from the same time period as the sales data.

# In[14]:


get_ipython().system('pip install yfinance')

import yfinance as yf


# In[15]:


min_date = min(daily_sales_df.index)
max_date = max(daily_sales_df.index)


# In[16]:


min_date


# In[17]:


max_date


# In[18]:


ticker = '^IXIC'
ticker_name = 'NASDAQ Composite'


# In[19]:


stock_data_df = yf.download(ticker, start=min_date, end=max_date)


# In[20]:


stock_data_df.reset_index(inplace=True)


# In[21]:


stock_data_df['Date'] = stock_data_df['Date'].dt.date


# In[22]:


stock_data_df = stock_data_df.set_index(stock_data_df['Date'])


# In[23]:


stock_data_df = stock_data_df.drop(['Date'], axis=1)


# In[24]:


stock_data_df.index.name = 'date'


# In[25]:


stock_data_df


# We combine the sales data and the index data in the same dataframe.

# In[26]:


daily_sales_df = pd.merge(left=daily_sales_df,
                          right=stock_data_df,
                          left_index=True,
                          right_index=True,
                          how='inner')


# In[27]:


daily_sales_df = daily_sales_df.rename({'Open':'stock_open',
                                        'High':'stock_high',
                                        'Low':'stock_low',
                                        'Close':'stock_close',
                                        'Adj Close':'stock_adjclose',
                                        'Volume':'stock_volume'},
                                       axis=1)


# In[28]:


daily_sales_df = daily_sales_df[['sales',
                                 'stock_open',
                                 'stock_high',
                                 'stock_low',
                                 'stock_close',
                                 'stock_adjclose',
                                 'stock_volume']]


# In[29]:


daily_sales_df


# ### Loading data from the digital newspaper "The Economic Times"

# In[30]:


get_ipython().system('pip install requests')
get_ipython().system('pip install beautifulsoup4')

import requests
from bs4 import BeautifulSoup
import time
import datetime
from dateutil import rrule
from calendar import monthrange
import csv


# In[31]:


def read_url(year, month, starttime):
    url = f'https://economictimes.indiatimes.com/archivelist/year-{year},month-{month},starttime-{starttime}.cms'
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')    
    return soup


# In[32]:


def get_starttime(year, month, day):
    date1 = '1899-12-30'
    timestamp1 = time.mktime(datetime.datetime.strptime(date1, '%Y-%m-%d').timetuple())
    
    date2 = str(year) + '-' + str(month) + '-' + str(day)
    timestamp2 = time.mktime(datetime.datetime.strptime(date2, '%Y-%m-%d').timetuple())
    
    starttime = ((timestamp2 - timestamp1) / 86400)
    return str(starttime).replace(".0", "")


# In[33]:


headlines_from = '2020-01-01'
headlines_to = '2020-10-31'


# In[34]:


headlines_datetime_from = datetime.datetime.strptime(headlines_from, '%Y-%m-%d')
headlines_datetime_to = datetime.datetime.strptime(headlines_to, '%Y-%m-%d')


# In[35]:


"""
for dt in rrule.rrule(rrule.MONTHLY, dtstart=headlines_datetime_from, until=headlines_datetime_to):
    year = int(dt.strftime('%Y'))
    month = int(dt.strftime('%m'))
    
    for day in range(1, (monthrange(year, month)[1] + 1)):
        starttime = get_starttime(year, month, day)
        date_str_eng = str(year) + '-' + '{:02d}'.format(month) + '-' + '{:02d}'.format(day)
        
        # print(f'Date: {year}-{month}-{day}')
        
        headlines = []

        soup = read_url(year, month, starttime)

        for td in soup.findAll('td', {'class':'contentbox5'}):
            for headline in td.findAll('a'):
                if 'archive' not in headline.get('href'):
                    if len(headline.contents) > 0:
                        if headline.contents[0] not in headlines:
                            headlines.append(headline.contents[0])

        time.sleep(1)

        file = open(f'./datasets/economic_news_headlines_{date_str_eng}.csv', 'w')
        with file:
            write = csv.writer(file, escapechar='\\', quoting=csv.QUOTE_NONE)
            for item in headlines:
                write.writerow([item,])
"""


# We detect negative words found in economic news headlines.

# In[36]:


get_ipython().system('pip install stop-words')

from stop_words import get_stop_words
import collections


# In[37]:


stop_words = get_stop_words('en')


# In[38]:


banned_chars = ['\\', '`', '"', '*', '_', '{', '}', '[', ']', '(', ')', '>', '#',
                '+', ':', '-', '.', ',', '¿', '?', '¡', '!', '$', '\'', '«', '»', '|']


# In[39]:


negative_economic_words = ['coronavirus', 'sars-cov-2', 'covid-19', 'covid19', 'virus', 'pandemic',
                           'lockdown', 'outbreak', 'curfew', 'quarantine', 'crisis', 'fears', 'violence',
                           'death', 'cases', 'fall', 'hit', 'impact']


# In[40]:


number_common_words = 20


# In[41]:


negative_economic_words_df = pd.DataFrame(columns=['date', 'negative_economic_words'])


# In[42]:


date_from = '2020-01-01'
date_to = '2020-10-31'


# In[43]:


for date in pd.date_range(date_from, date_to, freq='d'):
    date_str_eng = date.strftime('%Y-%m-%d')
    
    # print(f'Date: {date_str_eng}')
    # print()
    
    file = open(f'./datasets/economic_news_headlines_{date_str_eng}.csv', 'rt')

    headlines = []
    
    with file:
        csv_reader = csv.reader(file, escapechar='\\')

        for line in csv_reader:
            headlines.append(line)
    
    word_count = {}
    
    for headline in headlines:
        for word in headline[0].lower().split():
            for ch in banned_chars:
                if ch in word:
                    word = word.replace(ch, '')

            if (word != '') & (word not in stop_words):
                if word not in word_count:
                    word_count[word] = 1
                else:
                    word_count[word] += 1
    
    negative_words_count = 0
    word_counter = collections.Counter(word_count)
    
    # print(f"Top {number_common_words} most common words:")
    # print()
    
    for word, count in word_counter.most_common(number_common_words):
        # print(f'{word}: {count}')

        if word in negative_economic_words:
            negative_words_count += count
    
    # print()
    # print(f"Negative words: {negative_words_count}")
    # print()
    
    negative_economic_words_df = negative_economic_words_df.append({'date':date,
                                                                    'negative_economic_words':negative_words_count},
                                                                   ignore_index=True)


# In[44]:


negative_economic_words_df


# In[45]:


negative_economic_words_df.to_csv('./datasets/negative_economic_words.csv', index=False)


# We create a word cloud with all the negative words extracted from the press headlines.

# In[46]:


headlines = []


# In[47]:


for date in pd.date_range(date_from, date_to, freq='d'):
    date_str_eng = date.strftime('%Y-%m-%d')
    
    file = open(f'./datasets/economic_news_headlines_{date_str_eng}.csv', 'rt')

    with file:
        csv_reader = csv.reader(file, escapechar='\\')

        for line in csv_reader:
            headlines.append(line)


# In[48]:


word_count = {}
negative_word_count = {}

for headline in headlines:
    for word in headline[0].lower().split():
        for ch in banned_chars:
            if ch in word:
                word = word.replace(ch, '')
        
        if (word != '') & (word not in stop_words):
            if word not in word_count:
                word_count[word] = 1
            else:
                word_count[word] += 1
            
            if word in negative_economic_words:
                if word not in negative_word_count:
                    negative_word_count[word] = 1
                else:
                    negative_word_count[word] += 1


# In[49]:


number_common_words = 25

word_counter = collections.Counter(word_count)
negative_word_counter = collections.Counter(negative_word_count)

most_common_words = {}
most_common_negative_words = {}

for word, count in word_counter.most_common(number_common_words):
    most_common_words[word] = count

for word, count in negative_word_counter.most_common(number_common_words):
    most_common_negative_words[word] = count
    print(f'{word}: {count}')


# In[50]:


get_ipython().system('pip install wordcloud')
get_ipython().system('pip install matplotlib')

from wordcloud import WordCloud
import matplotlib.pyplot as plt


# In[51]:


wc = WordCloud(background_color='white',
               max_font_size=256,
               random_state=42,
               width=800,
               height=400
              ).generate_from_frequencies(most_common_negative_words)
plt.figure(figsize=(12, 8))
plt.imshow(wc, interpolation="bilinear")
plt.axis('off')
plt.show()


# We create the "negative_economic_words_df" dataframe that contains the number of negative words in the economic press from January to October 2020, and we join this data to the "daily_sales_df" dataframe that already contained the sales and Nasdaq Composite index values for the same period of time.

# In[52]:


negative_economic_words_csv_df = pd.read_csv('./datasets/negative_economic_words.csv')


# In[53]:


negative_economic_words_csv_df = negative_economic_words_csv_df.set_index(negative_economic_words_csv_df.date)


# In[54]:


negative_economic_words_csv_df = negative_economic_words_csv_df.drop('date', axis=1)


# In[55]:


negative_economic_words_df = pd.DataFrame(columns=['date'])


# In[56]:


for date in pd.date_range(min_date, max_date, freq='d'):
    negative_economic_words_df = negative_economic_words_df.append({'date':date}, ignore_index=True)


# In[57]:


negative_economic_words_df = negative_economic_words_df.set_index(negative_economic_words_df.date)


# In[58]:


negative_economic_words_df = negative_economic_words_df.drop('date', axis=1)


# In[59]:


negative_economic_words_df = pd.merge(left=negative_economic_words_df,
                                      right=negative_economic_words_csv_df[{'negative_economic_words'}],
                                      left_index=True,
                                      right_index=True,
                                      how='outer')


# In[60]:


negative_economic_words_df = negative_economic_words_df.fillna(0)


# In[61]:


negative_economic_words_df[{'negative_economic_words'}]


# In[62]:


daily_sales_df


# In[63]:


daily_sales_df = pd.merge(left=daily_sales_df,
                          right=negative_economic_words_df,
                          left_index=True,
                          right_index=True,
                          how='inner')


# In[64]:


daily_sales_df


# We check the contrast between the values of the "sales" and "negative_economic_words" columns, mainly as of March 13.

# In[65]:


daily_sales_df[(daily_sales_df.index >= '2020-03-01') & (daily_sales_df.index <= '2020-03-31')]


# ### Loading data from the digital newspaper ABC.es

# In[66]:


def read_url(date, page):
    url = f'https://www.abc.es/hemeroteca/dia-{date}/pagina-{page}'
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    html = soup.text
    contents = not("Sugerencias" in html)
    return contents, soup


# In[67]:


headlines_from = '2020-01-01'
headlines_to = '2020-10-31'


# In[68]:


headlines_timestamp_from = datetime.datetime.strptime(headlines_from, '%Y-%m-%d')
headlines_timestamp_to = datetime.datetime.strptime(headlines_to, '%Y-%m-%d')


# In[69]:


"""
for date in pd.date_range(headlines_timestamp_from, headlines_timestamp_to, freq='d'):
    date_str_eng = date.strftime('%Y-%m-%d')
    date_str_spa = date.strftime('%d-%m-%Y')
    
    # print(f'Date: {date_str_eng}')
    # print()
    
    headlines = []
    page = 1
    
    contents, soup = read_url(date_str_spa, page)
    
    while contents:
        # print(f'Page: {page}')
        # print()
        for headline in soup.findAll('a', {'class':'titulo'}):
            # print(headline.contents[0])
            headlines.append(headline.contents[0])            
        time.sleep(1)
        page += 1
        contents, soup = read_url(date_str_spa, page)
        # print()
    
    file = open(f'./datasets/news_headlines_{date_str_eng}.csv', 'w')
    
    with file:
        write = csv.writer(file, escapechar='\\', quoting=csv.QUOTE_NONE)
        for item in headlines:
            write.writerow([item,])
"""


# The "stop_words" and "banned_chars" lists are built with the words (in Spanish) and symbols to avoid, and "negative_words" with the most common negative words (in Spanish) to be counted.

# In[70]:


stop_words = get_stop_words('es')


# In[71]:


banned_chars = ['\\', '`', '"', '*', '_', '{', '}', '[', ']', '(', ')', '>', '#',
                '+', ':', '-', '.', ',', '¿', '?', '¡', '!', '$', '\'', '«', '»', '|']


# In[72]:


negative_words = ['coronavirus', 'sars-cov-2', 'covid-19', 'covid19', 'crisis', 'recesión', 'quiebra', 'caída',
                  'erte', 'ertes', 'cierre', 'cierra', 'ruina', 'alarma', 'medidas', 'casos', 'cuarentena', 'confinamiento',
                  'colapso', 'contagios', 'pandemia', 'epidemia', 'muertos', 'muertes', 'muere', 'fallecidos']


# In[73]:


number_common_words = 25


# In[74]:


negative_words_df = pd.DataFrame(columns=['date', 'negative_words'])


# In[75]:


date_from = '2020-01-01'
date_to = '2020-10-31'


# In[76]:


for date in pd.date_range(date_from, date_to, freq='d'):
    date_str_eng = date.strftime('%Y-%m-%d')
    
    # print(f'Date: {date_str_eng}')
    # print()
    
    file = open(f'./datasets/news_headlines_{date_str_eng}.csv', 'rt')

    headlines = []
    
    with file:
        csv_reader = csv.reader(file, escapechar='\\')

        for line in csv_reader:
            headlines.append(line)
    
    word_count = {}
    
    for headline in headlines:
        for word in headline[0].lower().split():
            for ch in banned_chars:
                if ch in word:
                    word = word.replace(ch, '')

            if (word != '') & (word not in stop_words):
                if word not in word_count:
                    word_count[word] = 1
                else:
                    word_count[word] += 1
    
    negative_words_count = 0
    word_counter = collections.Counter(word_count)
    
    # print(f"Top {number_common_words} most common words:")
    # print()
    
    for word, count in word_counter.most_common(number_common_words):
        # print(f'{word}: {count}')

        if word in negative_words:
            negative_words_count += count
    
    # print()
    # print(f"Negative words: {negative_words_count}")
    # print()
    
    negative_words_df = negative_words_df.append({'date':date, 'negative_words':negative_words_count}, ignore_index=True)


# In[77]:


negative_words_df


# In[78]:


negative_words_df.to_csv('./datasets/negative_words.csv', index=False)


# We make a word cloud with all the negative words extracted from the press headlines.

# In[79]:


headlines = []


# In[80]:


for date in pd.date_range(date_from, date_to, freq='d'):
    date_str_eng = date.strftime('%Y-%m-%d')
    
    file = open(f'./datasets/news_headlines_{date_str_eng}.csv', 'rt')

    with file:
        csv_reader = csv.reader(file, escapechar='\\')

        for line in csv_reader:
            headlines.append(line)


# In[81]:


word_count = {}
negative_word_count = {}

for headline in headlines:
    for word in headline[0].lower().split():
        for ch in banned_chars:
            if ch in word:
                word = word.replace(ch, '')

        if (word != '') & (word not in stop_words):
            if word not in word_count:
                word_count[word] = 1
            else:
                word_count[word] += 1
            
            if word in negative_words:
                if word not in negative_word_count:
                    negative_word_count[word] = 1
                else:
                    negative_word_count[word] += 1


# In[82]:


number_common_words = 25

word_counter = collections.Counter(word_count)
negative_word_counter = collections.Counter(negative_word_count)

most_common_words = {}
most_common_negative_words = {}

for word, count in word_counter.most_common(number_common_words):
    most_common_words[word] = count

for word, count in negative_word_counter.most_common(number_common_words):
    most_common_negative_words[word] = count
    print(f'{word}: {count}')


# In[83]:


wc = WordCloud(background_color='white',
               max_font_size=256,
               random_state=42,
               width=800,
               height=400
              ).generate_from_frequencies(most_common_negative_words)
plt.figure(figsize=(12, 8))
plt.imshow(wc, interpolation="bilinear")
plt.axis('off')
plt.show()


# Finally, the "negative_words_df" dataframe is created that contains the number of negative words in the general press from January to October 2020, and this data is joined to the "daily_sales_df" dataframe that already contained the sales values, of the Nasdaq index. Composite and the number of negative words in the economic press for the same period of time.

# In[84]:


negative_words_csv_df = pd.read_csv('./datasets/negative_words.csv')


# In[85]:


negative_words_csv_df = negative_words_csv_df.set_index(negative_words_csv_df.date)


# In[86]:


negative_words_csv_df = negative_words_csv_df.drop('date', axis=1)


# In[87]:


negative_words_df = pd.DataFrame(columns=['date'])


# In[88]:


for date in pd.date_range(min_date, max_date, freq='d'):
    negative_words_df = negative_words_df.append({'date':date}, ignore_index=True)


# In[89]:


negative_words_df = negative_words_df.set_index(negative_words_df.date)


# In[90]:


negative_words_df = negative_words_df.drop('date', axis=1)


# In[91]:


negative_words_df = pd.merge(left=negative_words_df,
                             right=negative_words_csv_df[{'negative_words'}],
                             left_index=True,
                             right_index=True, how='outer')


# In[92]:


negative_words_df = negative_words_df.fillna(0)


# In[93]:


daily_sales_df = pd.merge(left=daily_sales_df,
                          right=negative_words_df[{'negative_words'}],
                          left_index=True,
                          right_index=True,
                          how='inner')


# In[94]:


daily_sales_df


# We check the contrast between the values of the “sales” and “negative_words” columns, mainly as of March 12th.

# In[95]:


daily_sales_df[(daily_sales_df.index >= '2020-03-01') & (daily_sales_df.index <= '2020-03-31')]


# ## Exploratory data analysis

# In[96]:


daily_sales_df.info()


# In[97]:


daily_sales_df.describe()


# By observing the histograms, there is already a strong correlation between the variables "stock_open", "stock_high", "stock_low", "stock_close" and "stock_adjclose", as well as between the variables "negative_economic_words" and "negative_words".

# In[98]:


daily_sales_df.hist(figsize=(15, 15))


# We proceed to carry out some additional checks on the data.

# In[99]:


get_ipython().system('pip install pandas-profiling')

from pandas_profiling import ProfileReport


# In[100]:


profile = ProfileReport(daily_sales_df, title='Pandas Profiling Report')


# In[101]:


profile


# Next, the study of the correlation between variables is deepened by creating a heat map.

# In[102]:


get_ipython().system('pip install seaborn')

import seaborn as sns


# We check the possible correlation between variables using a heat map, for which we use the Pandas corr method.

# In[103]:


corr = daily_sales_df.corr()


# In[104]:


plt.subplots(figsize=(10, 10))
sns.heatmap(corr, annot=True, cmap='YlGnBu')
plt.show()


# As we can see, the correlation heat map provides us with a visual description of the relationship between the variables. Now, we do not want a set of independent variables that has a more or less similar relationship with the dependent variables. If we look at the dependent variable "sales" we make sure that there is no strong correlation with any independent variable.

# However, we will try to eliminate the strong dependency between the independent variables "stock_open", "stock_high", "stock_low", "stock_close" and "stock_adjclose" by adding a new variable called "stock_mean" as the mean of the variables "stock_high" and "stock_low", and the resulting correlation is checked again.

# In[105]:


daily_sales_df['stock_mean'] = (daily_sales_df.stock_low + daily_sales_df.stock_high) / 2


# In[106]:


daily_sales_df = daily_sales_df[['sales', 'stock_volume', 'stock_mean', 'negative_words', 'negative_economic_words']]


# In[107]:


corr = daily_sales_df.corr()


# In[108]:


plt.subplots(figsize=(10, 10))
sns.heatmap(corr, annot=True, cmap='YlGnBu')
plt.show()


# Although a strong correlation between the variables "negative_words" and "negative_economic_words" is still observed, they will be maintained for the application of the predictive model.

# We observe in a graph the behavior of daily sales.

# In[109]:


figsize=(18, 8)
plt.figure(figsize=figsize)
plt.title('Daily Sales')
plt.plot(daily_sales_df.index, daily_sales_df.sales)
plt.show()


# We look at the behavior of the NASDAQ index stocks.

# In[110]:


figsize=(18, 8)
plt.figure(figsize=figsize)
plt.title(f'{ticker_name} Low & High Mean')
plt.plot(daily_sales_df.index, daily_sales_df.stock_mean)
plt.show()


# We verify that it has a behavior that is certainly similar to that of sales.

# We verify that in mid-March 2020 there is a sharp decline in sales, and a strong increase a month later, with sustained growth in the future compared to previous periods.

# Then, the same values are observed in a period of time that contains the initial moment of the pandemic (between January and May 2020), verifying that they have a certainly similar behavior.

# In[111]:


pandemic_period_date_from = '2020-01-01'
pandemic_period_date_to = '2020-05-31'


# In[112]:


figsize=(18, 8)
plt.figure(figsize=figsize)
plt.title('Pandemic Sales')
plt.plot(daily_sales_df.index[(daily_sales_df.index >= pandemic_period_date_from) & (daily_sales_df.index <= pandemic_period_date_to)], daily_sales_df['sales'][(daily_sales_df.index >= pandemic_period_date_from) & (daily_sales_df.index <= pandemic_period_date_to)])
plt.show()


# In[113]:


figsize=(18, 8)
plt.figure(figsize=figsize)
plt.title(f'Pandemic {ticker_name} Low & High Mean')
plt.plot(daily_sales_df.index[(daily_sales_df.index >= pandemic_period_date_from) & (daily_sales_df.index <= pandemic_period_date_to)], daily_sales_df.stock_mean[(daily_sales_df.index >= pandemic_period_date_from) & (daily_sales_df.index <= pandemic_period_date_to)])
plt.show()


# Finally, the graph of the values of the negative words is observed, verifying that it has an inverse behavior to that of sales and the Nasdaq Composite index.

# In[114]:


figsize=(18, 8)
plt.figure(figsize=figsize)
plt.title('Negative Words')
plt.plot(daily_sales_df.index[(daily_sales_df.index >= pandemic_period_date_from) & (daily_sales_df.index <= pandemic_period_date_to)], daily_sales_df['negative_words'][(daily_sales_df.index >= pandemic_period_date_from) & (daily_sales_df.index <= pandemic_period_date_to)])
plt.show()


# ## Prediction model

# The training and test data sets are prepared with the independent (exogenous) and dependent (endogenous) variables for our SARIMAX time series predictive model.

# In[115]:


daily_sales_df = daily_sales_df.reset_index()


# In[116]:


daily_sales_df


# We prepare the <strong>X</strong> dataset with the independent (exogenous) variables.

# In[117]:


X = daily_sales_df[['date', 'stock_volume', 'stock_mean', 'negative_words', 'negative_economic_words']]


# In[118]:


X


# We prepare the dataset <strong>y</strong> with the dependent variable (endogenous).

# In[119]:


y = daily_sales_df[['date', 'sales']]


# In[120]:


y


# We prepare the <strong>train_sales_vals</strong> and <strong>train_exog_vals</strong> datasets with the training data sets from the period 01/01/2016 to 12/31/2019. We dismissed sales prior to 01/01/2016 as they presented irregular sales values (these were the business start-up years).

# In[121]:


train_date_from = '2016-01-01'
train_date_to = '2019-12-31'


# In[122]:


train_size = int(len(X[(X.date >= train_date_from) & (X.date <= train_date_to)]))


# In[123]:


train_size


# In[124]:


train_sales_vals = y.sales[(y.date >= train_date_from) & (y.date <= train_date_to)]
train_sales_vals = train_sales_vals.reset_index(drop=True)


# In[125]:


train_exog_vals = X[(X.date >= train_date_from) & (X.date <= train_date_to)][['stock_volume', 'stock_mean', 'negative_words', 'negative_economic_words']]
train_exog_vals = train_exog_vals.reset_index(drop=True)


# We prepare the <strong>test_sales_vals</strong> and <strong>test_exog_vals</strong> datasets with the test data sets from the period 01/01/2020 to 30/10/2020.

# In[126]:


test_date_from = '2020-01-01'
test_date_to = '2020-10-30'


# In[127]:


test_size = int(len(y[(y.date >= test_date_from) & (y.date <= test_date_to)]))


# In[128]:


test_size


# In[129]:


test_sales_vals = y.sales[(y.date >= test_date_from) & (y.date <= test_date_to)]
test_sales_vals = test_sales_vals.reset_index(drop=True)


# In[130]:


test_exog_vals = X[(X.date >= test_date_from) & (X.date <= test_date_to)][['stock_volume', 'stock_mean', 'negative_words', 'negative_economic_words']]
test_exog_vals = test_exog_vals.reset_index(drop=True)


# Next, we proceed to work on the prediction problem. It is important to remember that, since it is a time series prediction problem, it is necessary to test the seasonality of the time series in order to apply the SARIMAX model.

# It starts from the observation of the seasonality of the sales values in the period of one year.

# In[131]:


get_ipython().system('pip install pip install statsmodels')

import statsmodels.api as sm


# In[132]:


seas_d = sm.tsa.seasonal_decompose(y['sales'], model='add', period=365)


# In[133]:


fig = seas_d.plot()
fig.set_figwidth(18)
fig.set_figheight(12)
plt.show()


# We check the seasonality of the sales values in the monthly period.

# In[134]:


seas_d = sm.tsa.seasonal_decompose(y[y.date >= '2020-01-01']['sales'], model='add', period=30)


# In[135]:


fig = seas_d.plot()
fig.set_figwidth(18)
fig.set_figheight(12)
plt.show()


# We check the seasonality of the sales values in the weekly period.

# In[136]:


seas_d = sm.tsa.seasonal_decompose(y[y.date >= '2020-01-01']['sales'], model='add', period=7)


# In[137]:


fig = seas_d.plot()
fig.set_figwidth(18)
fig.set_figheight(12)
plt.show()


# It is verified that, at least graphically, a clear seasonality is observed in the sales data in the annual, monthly and weekly periods.

# If we make the data stationary, then the model can make predictions based on the fact that the mean and variance will remain the same in the future. A stationary series is easier to predict. To check if the data is stationary, we will use the <strong>Augmented Dickey-Fuller (ADF)</strong> test. It is the most popular statistical method to find if the series is stationary or not. Also called the unit root test.

# In[138]:


get_ipython().system('pip install stattools')

from statsmodels.tsa.stattools import adfuller


# In[139]:


def test_adf(series, title=''):
    dfout={}
    dftest=sm.tsa.adfuller(series.dropna(), autolag='AIC', regression='ct')
    
    for key, val in dftest[4].items():
        dfout[f'critical value ({key})'] = val
    
    if dftest[1] <= 0.05:
        print('Strong evidence against Null Hypothesis')
        print('Reject Null Hypothesis - Data is Stationary')
        print('Data is Stationary for', title)
    else:
        print('Strong evidence for Null Hypothesis')
        print('Accept Null Hypothesis - Data is not Stationary')
        print('Data is NOT Stationary for', title)


# We check that the sales data of the training set (from 01/01/2016 to 12/31/2019) are stationary.

# In[140]:


test_adf(train_sales_vals, 'Train Sales')


# We check that the sales data of the test set (from 01/01/2020 onwards) are not stationary.

# In[141]:


test_adf(test_sales_vals, 'Test Sales')


# We verify that the sales data of the test set (from 01/01/2020 onwards) become stationary by applying the <strong>logarithmic transformation</strong>.

# In[142]:


get_ipython().system('pip install numpy')

import numpy as np


# In[143]:


test_adf(np.log10(test_sales_vals), 'Log10 Test Sales')


# Based on this, we apply the logarithmic transformation to all datasets to stabilize the variance in the data and make it stationary before feeding it to the model.

# In[144]:


train_sales_log = np.log10(train_sales_vals)


# In[145]:


test_sales_log = np.log10(test_sales_vals)


# In[146]:


train_exog_log = np.log10(train_exog_vals)


# In[147]:


test_exog_log = np.log10(test_exog_vals)


# In[148]:


from numpy import inf


# In[149]:


train_exog_log[train_exog_log.negative_words == -inf] = 0


# In[150]:


test_exog_log[test_exog_log.negative_words == -inf] = 0


# We apply ARIMA and SARIMAX to our data and see which one works better. For both ARIMA and SARIMA or SARIMAX, we need to know the AR and MA terms to correct any autocorrelation in the differentiated series.

# We observe the graphs of the autocorrelation function (ACF) and partial autocorrelation (PACF) of the differentiated series.

# In[151]:


fig, ax = plt.subplots(2, 1, figsize=(18, 12))
fig=sm.tsa.graphics.plot_acf(train_sales_log, lags=50, ax=ax[0])
fig=sm.tsa.graphics.plot_pacf(train_sales_log, lags=50, ax=ax[1])
plt.show()


# In[152]:


fig, ax = plt.subplots(2, 1, figsize=(18, 12))
fig=sm.tsa.graphics.plot_acf(test_sales_log, lags=50, ax=ax[0])
fig=sm.tsa.graphics.plot_pacf(test_sales_log, lags=50, ax=ax[1])
plt.show()


# We see that the PACF plot has a significant peak at lag 1 and lag 2, which means that all higher order autocorrelations are effectively explained by the lag 1 and lag 2 autocorrelations.

# We use pyramid auto Arima to perform a step-by-step search for the term AR and MA that gives the lowest value of AIC.

# In[153]:


get_ipython().system('pip install pmdarima')

from pmdarima.arima import auto_arima


# In[154]:


stepwise_model = auto_arima(
    train_sales_log,
    exogenous=train_exog_log,
    start_p=0, start_q=0,
    start_P=0, start_Q=0,
    max_p=7, max_q=7,
    d=1, D=1,
    m=7,
    seasonal=True,
    trace=True,
    error_action='ignore',
    suppress_warnings=True,
    stepwise=True)


# In[155]:


stepwise_model.summary()


# The model suggested by auto_arima is SARIMAX, and the values of p, d, q and P, D, Q are 4, 1, 0 and 2, 1, 0, respectively.

# At this point, the next data point is predicted and the training data is traversed to predict the next data and add the next data point after the prediction for an additional forecast. This is like a moving window of daily level data.

# In[156]:


import warnings

warnings.filterwarnings('ignore')


# In[157]:


predictions = list()
predict_log = list()


# In[158]:


with warnings.catch_warnings():
    warnings.simplefilter("ignore")

    for t in range(len(test_sales_log)):
        stepwise_model.fit(train_sales_log)
        output = stepwise_model.predict(n_periods=1)
        predict_log.append(output[0])
        yhat = 10**output[0]
        predictions.append(yhat)
        obs = test_sales_log.iloc[t]
        train_sales_log = train_sales_log.append(pd.Series(obs), ignore_index=True)
        print('t=%f, predicted=%f, expected=%f' % (t, output[0], obs))
        obs = test_exog_log.iloc[t]
        train_exog_log = train_exog_log.append(pd.Series(obs), ignore_index=True)


# ## Results
# ### Results graph display

# The mean square error (RMSE) is used to evaluate the model.

# In[159]:


get_ipython().system('pip install python-math')
get_ipython().system('pip install scikit-metrics')

import math
from sklearn.metrics import mean_squared_error


# In[160]:


error = math.sqrt(mean_squared_error(test_sales_log, predict_log))
print('Test RMSE: %.3f' % error)


# Next, for visualization, let's create a data frame with the actual data available and the results of the prediction.

# In[161]:


predicted_df = pd.DataFrame()
predicted_df['date'] = daily_sales_df['date'][(daily_sales_df.date >= test_date_from) & (daily_sales_df.date <= test_date_to)]
predicted_df['sales'] = test_sales_vals.values
predicted_df['predicted'] = predictions


# In[162]:


predicted_df


# We draw the graph of the predictions in relation to the actual sales.

# In[163]:


figsize=(18, 8)
plt.figure(figsize=figsize)
plt.title('Actual Sales vs Predicted Sales')
plt.plot(predicted_df.date, predicted_df.sales, label='Sales')
plt.plot(predicted_df.date, predicted_df.predicted, color='red', label='Predicted')
plt.legend(loc='upper right')
plt.show()


# We see that the prediction fairly faithfully reproduces the peak of declines in sales in mid-March, as well as the rebound in sales the following month.

# ## Anomaly detection and visualization

# Once with the forecast results and actual data, we proceed to detect anomalies. To do this, the following steps are followed:
# 
# 1. Calculation of the error term (real-prediction).
# 2. Calculation of the moving average and the moving standard deviation (the window is one week).
# 3. Classification of the data with an error of 1.5, 1.75 and 2 standard deviations as limits for low, medium and high anomalies (5% of the data points would be anomalies identified according to this property).

# In[227]:


def detect_classify_anomalies(df, window):
    df = df.copy()
    df.replace([np.inf, -np.inf], np.NaN, inplace=True)
    df.fillna(0, inplace=True)
    df['error'] = df['sales'] - df['predicted']
    df['percentage_change'] = ((df['error']) / df['sales']) * 100
    df['meanval'] = df['error'].rolling(window=window).mean()
    df['deviation'] = df['error'].rolling(window=window).std()
    df['-3s'] = df['meanval'] - (2 * df['deviation'])
    df['3s'] = df['meanval'] + (2 * df['deviation'])
    df['-2s'] = df['meanval'] - (1.75 * df['deviation'])
    df['2s'] = df['meanval'] + (1.75 * df['deviation'])
    df['-1s'] = df['meanval'] - (1.5 * df['deviation'])
    df['1s'] = df['meanval'] + (1.5 * df['deviation'])
    cut_list = df[['error', '-3s', '-2s', '-1s', 'meanval', '1s', '2s', '3s']]
    cut_values = cut_list.values
    cut_sort = np.sort(cut_values)
    df['impact'] = [(lambda x: np.where(cut_sort == df['error'].iloc[x])[1][0])(x) for x in range(len(df['error']))]
    severity = {0:3, 1:2, 2:1, 3:0, 4:0, 5:1, 6:2, 7:3}
    region = {0:'NEGATIVE', 1:'NEGATIVE', 2:'NEGATIVE', 3:'NEGATIVE', 4:'POSITIVE', 5:'POSITIVE', 6:'POSITIVE', 7:'POSITIVE'}
    df['color'] = df['impact'].map(severity)
    df['region'] = df['impact'].map(region)
    df['anomaly_points_level_1'] = np.where(df['color'] == 1, df['error'], np.nan)
    df['anomaly_points_level_2'] = np.where(df['color'] == 2, df['error'], np.nan)
    df['anomaly_points_level_3'] = np.where(df['color'] == 3, df['error'], np.nan)
    df = df.sort_values(by='date', ascending=False)
    df.date = pd.to_datetime(df['date'].astype(str), format='%Y-%m-%d')
    return df


# In[228]:


classify_df = detect_classify_anomalies(predicted_df, 7)


# In[229]:


classify_df.reset_index(inplace=True)


# In[230]:


classify_df = classify_df.drop('index', axis=1)


# In[231]:


classify_df.date = classify_df.date.dt.strftime('%Y-%m-%d')


# In[232]:


classify_df


# Here is a function to display the results. Again, the importance of a clear and comprehensive display helps business users comment on anomalies and makes the results actionable.

# The first chart has the error term with the specified upper and lower bound, with the anomalies highlighted it would be easy for a user to interpret / validate. The second graph has actual and predicted values with anomalies highlighted.

# In[233]:


get_ipython().system('pip install plotly')

import plotly.graph_objects as go
from plotly.offline import plot


# In[234]:


def plot_anomaly(df, metric_name):
    dates = df.date
    
    bool_array_level_1 = (abs(df['anomaly_points_level_1']) > 0)
    sales_level_1 = df["sales"][-len(bool_array_level_1):]
    anomaly_points_level_1 = bool_array_level_1 * sales_level_1
    anomaly_points_level_1[anomaly_points_level_1 == 0] = np.nan
    
    bool_array_level_2 = (abs(df['anomaly_points_level_2']) > 0)
    sales_level_2 = df["sales"][-len(bool_array_level_2):]
    anomaly_points_level_2 = bool_array_level_2 * sales_level_2
    anomaly_points_level_2[anomaly_points_level_2 == 0] = np.nan
    
    bool_array_level_3 = (abs(df['anomaly_points_level_3']) > 0)
    sales_level_3 = df["sales"][-len(bool_array_level_3):]
    anomaly_points_level_3 = bool_array_level_3 * sales_level_3
    anomaly_points_level_3[anomaly_points_level_3 == 0] = np.nan
    
    color_map = {0:'rgba(228, 222, 249, 0.65)', 1:'yellow', 2:'orange', 3:'red'}
    
    table = go.Table(
        domain = dict(x=[0, 1],
                      y=[0, 0.3]),
        columnwidth = [1, 2],
        header = dict(height=20,
                      values=[['<b>Date</b>'], ['<b>Sales</b>'],
                              ['<b>Predicted</b>'], ['<b>% Difference</b>'], ['<b>Severity (0-3)</b>']],
                      font=dict(color=['rgb(45, 45, 45)'] * 5, size=14),
                      fill=dict(color='#d562be')),
        cells = dict(values=[df.round(2)[k].tolist() for k in ['date', 'sales', 'predicted',
                                                               'percentage_change', 'color']],
                     line=dict(color='#506784'),
                     align=['center'] * 5,
                     font=dict(color=['rgb(40, 40, 40)'] * 5, size=12),
                     suffix=[None] + [''] + [''] + ['%'] + [''],
                     height=27,
                     fill=dict(color=[df['color'].map(color_map)])))
    
    error = go.Scatter(name='Error',
                       x=dates,
                       y=df['error'],
                       xaxis='x1',
                       yaxis='y1',
                       mode='lines',
                       marker=dict(size=12,
                                   line=dict(width=1),
                                   color='darkred'),
                       text='Error')
    
    mvingavrg = go.Scatter(name='Moving Average',
                           x=dates,
                           y=df['meanval'],
                           mode='lines',
                           xaxis='x1',
                           yaxis='y1',
                           marker=dict(size=12,
                                       line=dict(width=1),
                                       color='green'),
                           text='Moving average')
    
    anomalies_level_1 = go.Scatter(name='Anomaly Level 1',
                           x=dates,
                           xaxis='x1',
                           yaxis='y1',
                           y=df['anomaly_points_level_1'],
                           mode='markers',
                           marker=dict(color='yellow',
                                       size=11,
                                       line=dict(color='yellow',
                                                 width=1)))
    
    anomalies_level_2 = go.Scatter(name='Anomaly Level 2',
                           x=dates,
                           xaxis='x1',
                           yaxis='y1',
                           y=df['anomaly_points_level_2'],
                           mode='markers',
                           marker=dict(color='orange',
                                       size=11,
                                       line=dict(color='orange',
                                                 width=1)))
    
    anomalies_level_3 = go.Scatter(name='Anomaly Level 3',
                           x=dates,
                           xaxis='x1',
                           yaxis='y1',
                           y=df['anomaly_points_level_3'],
                           mode='markers',
                           marker=dict(color='red',
                                       size=11,
                                       line=dict(color='red',
                                                 width=1)))
    
    upper_bound = go.Scatter(name='Upper Confidence Interval',
                             x=dates,
                             showlegend=False,
                             xaxis='x1',
                             yaxis='y1',
                             y=df['3s'],
                             marker=dict(color='#444'),
                             line=dict(color=('rgb(23, 96, 167)'),
                                       width=2,
                                       dash='dash'),
                             fillcolor='rgba(68, 68, 68, 0.3)',
                             fill='tonexty')
    
    lower_bound = go.Scatter(name='Confidence Interval',
                             x=dates,
                             xaxis='x1',
                             yaxis='y1',
                             y=df['-3s'],
                             marker=dict(color='#444'),
                             line=dict(color=('rgb(23, 96, 167)'),
                                       width=2,
                                       dash='dash'),
                             fillcolor='rgba(68, 68, 68, 0.3)',
                             fill='tonexty')
    
    sales = go.Scatter(name='Sales',
                       x=dates,
                       y=df['sales'],
                       xaxis='x2',
                       yaxis='y2',
                       mode='lines',
                       marker=dict(size=12,
                                   line=dict(width=1),
                                   color='blue'))
    
    predicted = go.Scatter(name='Predicted',
                           x=dates,
                           y=df['predicted'],
                           xaxis='x2',
                           yaxis='y2',
                           mode='lines',
                           marker=dict(size=12,
                                       line=dict(width=1),
                                       color='silver'))
    
    anomalies_map_level_1 = go.Scatter(name='Anomaly Sales Level 1',
                               showlegend=False,
                               x=dates,
                               y=anomaly_points_level_1,
                               mode='markers',
                               xaxis='x2',
                               yaxis='y2',
                               marker=dict(color='yellow',
                                           size=11,
                                           line=dict(color='yellow',
                                                     width=1)))
    
    anomalies_map_level_2 = go.Scatter(name='Anomaly Sales Level 2',
                               showlegend=False,
                               x=dates,
                               y=anomaly_points_level_2,
                               mode='markers',
                               xaxis='x2',
                               yaxis='y2',
                               marker=dict(color='orange',
                                           size=11,
                                           line=dict(color='orange',
                                                     width=1)))
    
    anomalies_map_level_3 = go.Scatter(name='Anomaly Sales Level 3',
                               showlegend=False,
                               x=dates,
                               y=anomaly_points_level_3,
                               mode='markers',
                               xaxis='x2',
                               yaxis='y2',
                               marker=dict(color='red',
                                           size=11,
                                           line=dict(color='red',
                                                     width=1)))
    
    axis = dict(showline=True,
                zeroline=False,
                showgrid=True,
                mirror=True,
                ticklen=4,
                gridcolor='#ffffff',
                tickfont=dict(size=10))
    
    layout = dict(width=950,
                  height=950,
                  autosize=False,
                  title=metric_name,
                  margin=dict(l=0, r=0, t=50, b=10),
                  showlegend=True,
                  legend=dict(font=dict(size=10)),
                  xaxis1=dict(axis,
                              **dict(domain=[0, 1],
                                     anchor='y1',
                                     showticklabels=True)),
                  xaxis2=dict(axis,
                              **dict(domain=[0, 1],
                                     anchor='y2',
                                     showticklabels=True)),
                  yaxis1=dict(axis,
                              **dict(domain=[0.70, 1],
                                     anchor='x1',
                                     hoverformat='.2f')),
                  yaxis2=dict(axis,
                              **dict(domain=[0.34, 0.64],
                                     anchor='x2',
                                     hoverformat='.2f')))
    
    fig = go.Figure(data=[table, upper_bound, lower_bound, sales,
                          predicted, mvingavrg, error,
                          anomalies_level_1, anomalies_level_2, anomalies_level_3,
                          anomalies_map_level_1, anomalies_map_level_2, anomalies_map_level_3],
                    layout=layout)
    
    return plot(fig, filename='./output/anomaly_detection.html')


# In[235]:


plot_anomaly(classify_df, 'Daily Sales Anomaly Detection')


# By using a moving average and standard deviation here, you avoid continuous false anomalies during scenarios such as big sales days. The first peak or dip is highlighted, after which the thresholds are adjusted. Also, the table providing actual data predicted the change and conditional formatting based on the level of anomalies.

# Finally, an alternative is proposed to detect anomalies, in this case due to different levels of percentage of error between the real and predicted values.

# In[236]:


def detect_classify_anomalies_percentages(df, window):
    df = df.copy()
    df.replace([np.inf, -np.inf], np.NaN, inplace=True)
    df.fillna(0, inplace=True)
    df['error'] = abs(df['sales'] - df['predicted'])
    df['percentage_change'] = ((df['error']) / df['sales']) * 100
    df['25p_error'] = (df['sales'] * 0.25)
    df['50p_error'] = (df['sales'] * 0.5)
    df['75p_error'] = (df['sales'] * 0.75)
    df['100p_error'] = (df['sales'] * 1)
    cut_list = df[['error', '25p_error', '50p_error', '75p_error', '100p_error']]
    cut_values = cut_list.values
    cut_sort = np.sort(cut_values)
    df['impact'] = [(lambda x: np.where(cut_sort == df['error'].iloc[x])[1][0])(x) for x in range(len(df['error']))]
    severity = {0:0, 1:1, 2:2, 3:3, 4:4}
    df['color'] = df['impact'].map(severity)
    df['anomaly_points_level_1'] = np.where(df['color'] == 1, df['error'], np.nan)
    df['anomaly_points_level_2'] = np.where(df['color'] == 2, df['error'], np.nan)
    df['anomaly_points_level_3'] = np.where(df['color'] == 3, df['error'], np.nan)
    df['anomaly_points_level_4'] = np.where(df['color'] == 4, df['error'], np.nan)
    df = df.sort_values(by='date', ascending=False)
    df.date = pd.to_datetime(df['date'].astype(str), format='%Y-%m-%d')
    return df


# In[237]:


classify_df = detect_classify_anomalies_percentages(predicted_df, 7)


# In[238]:


classify_df.reset_index(inplace=True)


# In[239]:


classify_df = classify_df.drop('index', axis=1)


# In[240]:


classify_df.date = classify_df.date.dt.strftime('%Y-%m-%d')


# In[241]:


classify_df


# In[242]:


def plot_anomaly_percentages(df, metric_name):
    dates = df.date
    
    bool_array_level_1 = (abs(df['anomaly_points_level_1']) > 0)
    sales_level_1 = df["sales"][-len(bool_array_level_1):]
    anomaly_points_level_1 = bool_array_level_1 * sales_level_1
    anomaly_points_level_1[anomaly_points_level_1 == 0] = np.nan
    
    bool_array_level_2 = (abs(df['anomaly_points_level_2']) > 0)
    sales_level_2 = df["sales"][-len(bool_array_level_2):]
    anomaly_points_level_2 = bool_array_level_2 * sales_level_2
    anomaly_points_level_2[anomaly_points_level_2 == 0] = np.nan
    
    bool_array_level_3 = (abs(df['anomaly_points_level_3']) > 0)
    sales_level_3 = df["sales"][-len(bool_array_level_3):]
    anomaly_points_level_3 = bool_array_level_3 * sales_level_3
    anomaly_points_level_3[anomaly_points_level_3 == 0] = np.nan
    
    bool_array_level_4 = (abs(df['anomaly_points_level_4']) > 0)
    sales_level_4 = df["sales"][-len(bool_array_level_4):]
    anomaly_points_level_4 = bool_array_level_4 * sales_level_4
    anomaly_points_level_4[anomaly_points_level_4 == 0] = np.nan
    
    color_map = {0:'rgba(228, 222, 249, 0.65)', 1:'yellow', 2:'orange', 3:'red', 4:'darkred'}
    
    table = go.Table(
        domain = dict(x=[0, 1],
                      y=[0, 0.45]),
        columnwidth = [1, 2],
        header = dict(height=20,
                      values=[['<b>Date</b>'], ['<b>Sales</b>'],
                              ['<b>Predicted</b>'], ['<b>% Difference</b>'], ['<b>Severity (0-3)</b>']],
                      font=dict(color=['rgb(45, 45, 45)'] * 5, size=14),
                      fill=dict(color='#d562be')),
        cells = dict(values=[df.round(2)[k].tolist() for k in ['date', 'sales', 'predicted',
                                                               'percentage_change', 'color']],
                     line=dict(color='#506784'),
                     align=['center'] * 5,
                     font=dict(color=['rgb(40, 40, 40)'] * 5, size=12),
                     suffix=[None] + [''] + [''] + ['%'] + [''],
                     height=27,
                     fill=dict(color=[df['color'].map(color_map)])))
    
    sales = go.Scatter(name='Sales',
                       x=dates,
                       y=df['sales'],
                       xaxis='x1',
                       yaxis='y1',
                       mode='lines',
                       marker=dict(size=12,
                                   line=dict(width=1),
                                   color='blue'))
    
    predicted = go.Scatter(name='Predicted',
                           x=dates,
                           y=df['predicted'],
                           xaxis='x1',
                           yaxis='y1',
                           mode='lines',
                           marker=dict(size=12,
                                       line=dict(width=1),
                                       color='silver'))
    
    anomalies_map_level_1 = go.Scatter(name='Anomaly Sales Level 1',
                               x=dates,
                               y=anomaly_points_level_1,
                               mode='markers',
                               xaxis='x1',
                               yaxis='y1',
                               marker=dict(color='yellow',
                                           size=11,
                                           line=dict(color='yellow',
                                                     width=1)))
    
    anomalies_map_level_2 = go.Scatter(name='Anomaly Sales Level 2',
                               x=dates,
                               y=anomaly_points_level_2,
                               mode='markers',
                               xaxis='x1',
                               yaxis='y1',
                               marker=dict(color='orange',
                                           size=11,
                                           line=dict(color='orange',
                                                     width=1)))
    
    anomalies_map_level_3 = go.Scatter(name='Anomaly Sales Level 3',
                               x=dates,
                               y=anomaly_points_level_3,
                               mode='markers',
                               xaxis='x1',
                               yaxis='y1',
                               marker=dict(color='red',
                                           size=11,
                                           line=dict(color='red',
                                                     width=1)))
    
    anomalies_map_level_4 = go.Scatter(name='Anomaly Sales Level 4',
                               x=dates,
                               y=anomaly_points_level_4,
                               mode='markers',
                               xaxis='x1',
                               yaxis='y1',
                               marker=dict(color='darkred',
                                           size=11,
                                           line=dict(color='darkred',
                                                     width=1)))
    
    axis = dict(showline=True,
                zeroline=False,
                showgrid=True,
                mirror=True,
                ticklen=4,
                gridcolor='#ffffff',
                tickfont=dict(size=6))
    
    layout = dict(width=950,
                  height=950,
                  autosize=False,
                  title=metric_name,
                  margin=dict(l=0, r=0, t=50, b=10),
                  showlegend=True,
                  legend=dict(yanchor="top",
                              y=0.99,
                              xanchor="left",
                              x=0.01,
                              font=dict(size=10)),
                  xaxis1=dict(axis,
                              **dict(domain=[0, 1],
                                     anchor='y1',
                                     showticklabels=True)),
                  yaxis1=dict(axis,
                              **dict(domain=[0.5, 1],
                                     anchor='x1',
                                     hoverformat='.2f')))
    
    fig = go.Figure(data=[table, sales, predicted,
                          anomalies_map_level_1, anomalies_map_level_2, anomalies_map_level_3, anomalies_map_level_4],
                    layout=layout)
    
    return plot(fig, filename='./output/anomaly_detection_percentages.html')


# In[243]:


plot_anomaly_percentages(classify_df, "Daily Sales Anomaly Detection (by percentages)")


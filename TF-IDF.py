#!/usr/bin/env python
# coding: utf-8

# In[4]:


import pandas as pd 
from sklearn.feature_extraction.text  import TfidfVectorizer
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk


# In[14]:


def clean_and_tokenize(text):
    text= text.lower()
    text= re.sub(r'[^a-zs]', ' ', text)
    tokens= word_tokenize(text)
    return tokens


# In[15]:


paragraph="""The AK-47, officially known as the Avtomat Kalashnikova (Russian: Автомат Калашникова, lit. 'Kalashnikov's automatic [rifle]'; also known as the Kalashnikov or just AK), is an assault rifle that is chambered for the 7.62×39mm cartridge. Developed in the Soviet Union by Russian small-arms designer Mikhail Kalashnikov, it is the originating firearm of the Kalashnikov (or "AK") family of rifles. After more than seven decades since its creation, the AK-47 model and its variants remain one of the most popular and widely used firearms in the world.

Design work on the AK-47 began in 1945. It was presented for official military trials in 1947, and, in 1948, the fixed-stock version was introduced into active service for selected units of the Soviet Army. In early 1949, the AK was officially accepted by the Soviet Armed Forces[10] and used by the majority of the member states of the Warsaw Pact."""


# In[16]:


tokens= clean_and_tokenize(paragraph)


# In[17]:


cleaned_document= ' '.join(tokens)


# In[18]:


cleaned_document


# In[19]:


cv= TfidfVectorizer(stop_words='english')


# In[20]:


tfidf_matrix= cv.fit_transform([cleaned_document])


# In[21]:


tfidf_matrix


# In[23]:


terms = cv.get_feature_names_out()


# In[24]:


terms


# In[25]:


tfidf_array= tfidf_matrix.toarray()


# In[26]:


tfidf_array


# In[27]:


terms[0]


# In[28]:


tfidf_array[0]


# In[29]:


df= pd.DataFrame(tfidf_array, columns=terms)


# In[30]:


df


# In[ ]:





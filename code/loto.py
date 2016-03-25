
# coding: utf-8

# # Loto code

# In[1]:

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import datetime
get_ipython().magic('matplotlib inline')

from scipy.stats import chisquare
from scipy.stats import anderson


# #### Monte Carlo computation of the chi2 test on a dice

# In[2]:

def role_dice(n):
    out = np.random.randint(1,7,n)
    exp = float(n)/6 # we expect having each nmumber n/6 times
    obs = [(out==i).sum() for i in np.arange(1,7)] # true observations
    T = [(obs[i]-exp)**2/exp for i in np.arange(6)]
    T = np.sum(T)
    return T

n_exp = 10000
n_simul = 1000
T = []
for n in np.arange(n_exp):
    T.extend([role_dice(n_simul)])

print(np.percentile(T,95))


# ## Let's play with the lottery data

# In[3]:

data = pd.read_csv('/home/romain/Documents/Github/loto/data/nouveau_loto.csv',sep=';',index_col=False)
del data['Unnamed: 25'] # the file end each line with a semi-colon which causes an empty column
print(data.shape)


# In[4]:

data.head()


# ### Application of the chi2 test on the lottery data - Does some numbers get out more often ?

# #### Classic numbers

# In[5]:

boule = pd.concat([data['boule_1'],data['boule_2'],data['boule_3'],data['boule_4'],data['boule_5']])
freq = boule.value_counts()
plt.figure(figsize=(12,6))
plt.bar(list(freq.index),freq)


# ** Is this odd ? **
# > To check it, we will use the chi-2 test with 48 degree of freedom that the observation come from a multionomial law

# In[6]:

chisquare(freq)


# > The p-value is really high, so we do not reject H0 and thus there is nothing abnormal there

# #### The 'numero chance'

# In[7]:

freq_chance = data['numero_chance'].value_counts()
plt.figure(figsize=(12,6))
plt.bar(list(freq_chance.index),freq_chance)
plt.xticks(np.arange(1,11)+0.5,
                   ["numero chance "+str(i+1) for i in np.arange(10)],rotation='vertical')


# In[8]:

chisquare(freq_chance)


# > This one is a little more litigous ... Most of the time we want a p-value greater than 0.05 ... It may be a good idea to choose 1 as a 'numero chance'

# #### What about pairs ?

# In[9]:

def search(x,i):
    a = re.search(r"(^|-)"+str(i)+"(-|\+)",x)
    if a:
        out = True
    else:
        out = False
    return out

pairs = np.zeros((49,49))
for i in np.arange(1,50):
    ind_i = list(map(lambda x: search(x,i),data['combinaison_gagnante_en_ordre_croissant']))
    subdata_i = data.ix[ind_i]
    for j in np.arange(i+1,50):
        ind_ij = list(map(lambda x: search(x,j),subdata_i['combinaison_gagnante_en_ordre_croissant']))
        pairs[i-1,j-1] = np.sum(ind_ij)
        pairs[j-1,i-1] = np.sum(ind_ij)


# In[10]:

pairs = pd.DataFrame(pairs)
plt.style.use('ggplot')
plt.figure(figsize=(15,10))
plt.title('Pairs heatmap')
plt.pcolor(pairs,cmap=plt.cm.Reds)
width = .5
height = .5
plt.xticks(np.arange(pairs.shape[0])+width,
           ["boule "+str(i+1) for i in np.arange(pairs.shape[1])],rotation='vertical')
plt.yticks(np.arange(pairs.shape[0])+width,
           ["boule "+str(i+1) for i in np.arange(pairs.shape[1])])


# In[11]:

ind = np.triu_indices(49,1)
chisquare(np.array(pairs)[ind])


# > Again, the p-value is really high, so we do not reject H0 and thus there is nothing abnormal there. Too bad for us :)

# ### On the winning probability

# #### Over the years

# In[12]:

data['date'] = list(map(lambda x: datetime.datetime.strptime(x,'%d/%m/%Y').date(),data['date_de_tirage']))


# In[13]:

f, axarr = plt.subplots(3, 2,figsize=(15,15))
for i in np.arange(6):
    r = int(i/2)
    c = i%2
    axarr[r,c].plot(data['date'],data['nombre_de_gagnant_au_rang'+str(i+1)])
    axarr[r,c].set_title('number of winner with '+str(6-i)+' correct numbers')
plt.show()


# > There does not seem to be a winning trend over the years. Now is not better than before !

# #### The day of the week ?

# In[14]:

f, axarr = plt.subplots(3, 2,figsize=(15,15))
days = list(set(data['jour_de_tirage']))
for i in np.arange(6):
    r = int(i/2)
    c = i%2
    tab = [np.mean(data['nombre_de_gagnant_au_rang'+str(i+1)][data['jour_de_tirage']==days[k]]) 
           for k in np.arange(len(days))]
    axarr[r,c].scatter(np.arange(len(days)),tab)
    axarr[r,c].set_title('number of winner with '+str(6-i)+' correct numbers')
    axarr[r,c].set_xticklabels(['','Monday','','Wednesday','','Saturday',''])
plt.show()


# > Saturday seems to be a good day to play ! Or not ... because if there is more winner, you will win less. What would be more interesting is the proportion of winner, unfortunately we do not have the number of opponents. But with the randomness hypothesis tested so far, we can infer that the game is really random, so your winning probability won't change with the day of the week, but the money you will earn may be bigger on Mondays than on Saturdays !

# In[15]:

days = list(set(data['jour_de_tirage']))
uplift = []
for i in np.arange(6):
    monday = np.mean(data['nombre_de_gagnant_au_rang'+str(i+1)][data['jour_de_tirage']==days[0]]) 
    restOfWeek = np.mean(data['nombre_de_gagnant_au_rang'+str(i+1)][data['jour_de_tirage']!=days[0]]) 
    uplift.extend([(restOfWeek-monday)/restOfWeek*100])
print(np.mean(uplift))


# #### The month

# In[16]:

data['month'] = list(map(lambda dt: dt.month,data['date']))
f, axarr = plt.subplots(3, 2,figsize=(15,15))
for i in np.arange(6):
    r = int(i/2)
    c = i%2
    tab = [np.mean(data['nombre_de_gagnant_au_rang'+str(i+1)][data['month']==k]) 
           for k in np.arange(1,13)]
    axarr[r,c].scatter(np.arange(1,13),tab)
    axarr[r,c].set_title('number of winner with '+str(6-i)+' correct numbers')
    axarr[r,c].set_xlim([0.5,12.5])
plt.show()


# > It seems that there is less opponents during summer. Just sayin' ...

# In[17]:

uplift = []
for i in np.arange(6):
    june = np.mean(data['nombre_de_gagnant_au_rang'+str(i+1)][(data['month']==6)]) 
    restOfYear = np.mean(data['nombre_de_gagnant_au_rang'+str(i+1)][(data['month']!=6)]) 
    uplift.extend([(restOfYear-june)/restOfYear*100])
print(np.mean(uplift))


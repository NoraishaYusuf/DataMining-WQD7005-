# =============================================================================
# Milestone 6: CRAWL DATASET CODES 
# =============================================================================

# run first
import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import requests
from bs4 import BeautifulSoup
import string
import time
import re
import requests
from bs4 import BeautifulSoup
from twitterscraper import query_tweets
import datetime as dt
import numpy as np

# =============================================================================
# Part 1:
# Crawling all the stockname from the star (only once)
# =============================================================================

##### MUST DOWNLOAD WEBDRIVER fIRST 
##### http://chromedriver.chromium.org/ for Google chrome driver
##### https://github.com/mozilla/geckodriver/releases for Mozilla Firefox driver

urlTheStar='https://www.thestar.com.my/business/marketwatch/stock-list/?alphabet='
alpha = []
for letter in string.ascii_uppercase:
    alpha.append(letter)     
alpha.append('0-9')
print("!!!  Array of chars")
print(alpha)

stockname = []
for i in alpha:
    print("!!!  Now char "+ i)
#   Make sure to change the path to where you store your webdriver!!!!
    browser = webdriver.Firefox(executable_path=r'C:\Users\shash\Anaconda3\geckodriver.exe')
    browser.implicitly_wait(40)
    browser.get(urlTheStar + i)
    WebDriverWait(browser,40).until(EC.visibility_of_element_located((By.ID,'marketwatchtable')))
    innerHTML = browser.find_element_by_id("marketwatchtable").get_attribute("innerHTML")
    soup = BeautifulSoup(innerHTML, 'lxml') 
    links = soup.findAll('a')
    for link in links:
        splitlink = link['href'].split('=')
        stock = splitlink[1]
        stockname.append(stock)
        print(stock)
    browser.close()

dict = {'name':stockname}
df_stockname = pd.DataFrame(dict)
df_stockname.to_csv('stockname.csv')

# =============================================================================
# Part 2:
# Crawling historical prices (only once)
# =============================================================================

#using the stockname crawled and saved in csv. Then transform dataframe into list
df1 = pd.read_csv('stockname.csv',usecols=[1])
datanames = df1['name'].tolist()

sl=[];cl=[];ol=[];hl=[];ll=[];dl=[];vl=[];stocknames2=[]  

#set timeframe to crawl e.g. 3 months
startdate=str(1546343431) #date = Tuesday, January 1, 2019 7:50:31 PM
enddate=str(1554205831) #date = Tuesday, April 2, 2019 7:50:31 PM 

for name in datanames:
    url = 'https://charts.thestar.com.my/datafeed-udf/history?symbol='+name+'&resolution=D&from='+startdate+'&to='+enddate
    r = requests.get(url).json() 
    if r["s"] == "ok":
        stocknames2.append(name)
        for t in r["t"]:
            day=time.strftime("%Y-%m-%d",time.localtime(int(t)))
            dl.append(day)
            sl.append(name)
        for o in r["o"]:ol.append(o) #open price
        for c in r["c"]:cl.append(c) #closing price
        for h in r["h"]:hl.append(h) #high price
        for l in r["l"]:ll.append(l) #low price
        for v in r["v"]:vl.append(v) #volume
    print("Done for "+ name)
    #break       
    
df = pd.DataFrame({'name':sl,'day':dl,'close':cl,'open':ol,'high':hl,'low':ll,'volume':vl})
df.to_csv('price_df.csv')


# =============================================================================
# Part 3: 
# Crawl financial information)
# =============================================================================
print("!!! START CRAWLING FINANCIAL DATA")

# Update path to web driver first !!!!!
browser = webdriver.Firefox(executable_path=r'C:\Users\shash\Anaconda3\geckodriver.exe')
browser.implicitly_wait(40)
Financialurl = 'https://klse.i3investor.com/financial/quarter/latest.jsp'
browser.get(Financialurl)
WebElementexpanded = browser.find_element_by_xpath("//*[@id='ui-accordion-financialResultTableColumnsDiv-header-0']/span")
WebElementexpanded.click()


# !!!!! BEFORE RUNNING NEXT CODES, MANUALLY CLOSE COOKIE POPUPS when the website appears!!!!
# =============================================================================

allLinks = browser.find_elements_by_xpath('//input[@type="checkbox"]')
for link in allLinks:
    if link.is_selected():
        print('Checkbox already selected');
    else:
        link.click();
        print('Checkbox selected'); 

df = pd.DataFrame(columns=['no', 'stockname', 'annDate', 'fy', 
                           'quarter', 'h', 'price', 'ch', 'percentage', 
                           'revenue', 'pbt', 'np', 'NptoSh', 'dividend', 
                           'networth', 'divpayout', 'npmargin', 'roe', 
                           'rps', 'eps', 'dps', 'naps', 'QoY', 'YoY'])

elm = browser.find_element_by_class_name('next')
tbl = browser.find_element_by_xpath('//*[@id="tablebody"]')
while True:
    element = WebDriverWait(browser, 100).until(lambda x: x.find_element_by_id('tablebody'))
    for row in tbl.find_elements_by_tag_name('tr'):
        data = row.find_elements_by_tag_name('td')
        file_row = []
        for datum in data:
            datum_text = datum.text
            file_row.append(datum_text)
        print(file_row)
        s = pd.Series(file_row, index = df.columns)
        df = df.append(s,ignore_index=True)
    elm = browser.find_element_by_class_name('next')
    if 'ui-state-disabled' in elm.get_attribute('class'):
        break;
    elm.click()      
        
df = df.drop(columns='no')
df.to_csv('financial_df.csv',index=False)


# =============================================================================
# Part 4: 
# Crawl News from the STAR online
# =============================================================================
url1 = 'https://www.thestar.com.my/search/?q='
url2 = '&qsort=newest&qrec=10&qstockcode=&pgno='

# Download potentialstockdetails.csv file from github. 
#this data comprises keywords search which were manually key-in due to required customization
dfstock = pd.read_csv('potentialstockdetails.csv')
stockkeywords = dfstock['News key'].tolist()
print(stockkeywords)

name = [];timestamp = [];titles = [];intro=[];
numbers = list(range(1,8))

# # Update path to web driver first !!!!!
browser = webdriver.Firefox(executable_path=r'C:\Users\shash\Anaconda3\geckodriver.exe')
for word in stockkeywords:
    for i in numbers:
        newurl = url1+str(word)+url2+str(i)
#       print(newurl)
        browser.get(newurl)
        innerHTML = browser.execute_script('return document.body.innerHTML')
        soup = BeautifulSoup(innerHTML, 'lxml')
        datetimes = soup.findAll("label",{"class":"timestamp"})
        title = soup.findAll("h2",{"class":"f18"})
        shorttext = soup.findAll(id=re.compile("summary_result$"))
        print(word)
        for d in datetimes:
            print(d.getText())
            dtext = d.getText()
            timestamp.append(dtext)
            name.append(word)    
        for t in title:
#            print(t.getText())
            ttext = t.getText()
            titles.append(ttext)        
        for c in shorttext:
#            print(c.getText())
            ctext = c.getText()
            intro.append(ctext)
#    break;

# need to convert the dates    
df = pd.DataFrame({'name':name, 'dates':timestamp,'newstitle':titles,'newsintro':intro})
new = df["dates"].str.split("|", n = 1, expand = True) 
df["dates"] = new[0]
df["dates"] = pd.to_datetime(df['dates'])
df.head(6)

newdf = df[(df['dates']>dt.date(2019,1,1)) & (df['dates']<dt.date(2019,4,1))]  
newdf.rename(columns={'name':'News key'}, inplace=True)
newsdf = pd.merge(newdf,dfstock,how='inner',on='News key')

newsdf.to_csv('q1newsstockpot.csv',index=False)
newsdf = pd.read_csv('q1newsstockpot.csv')

#extract full text for word cloud (word cloud visualize on Tableau)
newsdf['fulltext'] = newsdf['newstitle'].str.cat(newsdf['newsintro'],sep=" ")
words = newsdf['fulltext'].str.split()
newsword_counts = pd.value_counts(words.apply(pd.Series).stack())
newsword_counts.to_csv('q1newstext.csv',index=True)

# =============================================================================
# Part 5: 
# Crawl tweets from Twitter
# =============================================================================

begin_date = dt.datetime.strptime("Jan 01 2019", "%b %d %Y").date()
end_date = dt.datetime.strptime("Apr 01 2019", "%b %d %Y").date()

# Download potentialstockdetails.csv file from github. 
#this data comprises keywords search which were manually key-in due to required customization
df = pd.read_csv('potentialstockdetails.csv')
stockfilter = df['Twitter key'].tolist();print(stockfilter)

#crawl loop function
tweets = []
filters = []
jsonlist=[]

for i in stockfilter:
    t = query_tweets([i],begindate=begin_date,enddate=end_date,lang='en',limit=400)
    for j in t:
        jsonlist.append(vars(j)) 
    size = len(t)
    if size >0:
        for m in range(size):
            filters.append(i)
    print("done for "+i)
#    break #remove break to crawl full tweet

df1 = pd.DataFrame({'filters':filters})
df2 = pd.DataFrame(jsonlist)
df3 = pd.concat([df1,df2],axis=1) 

#update date format
df3['timestamp'] = df3.timestamp.apply(lambda x: x.strftime('%Y-%m-%d'))
df3.rename(columns={'filters':'Twitter key'}, inplace=True)
tweetsdf = pd.merge(df,df3,how='inner',on='Twitter key')
tweetsdf.to_csv('q1tweetstockpot.csv',index=False)

#extract full text for word cloud (word cloud visualize on Tableau)
tweetsdf = pd.read_csv('q1tweetstockpot.csv')
words = tweetsdf['text'].str.split()
tweetsword_counts = pd.value_counts(words.apply(pd.Series).stack())
tweetsword_counts.to_csv('q1tweetstext.csv',index=True)

# =============================================================================
# Part 6: 
# For 'interest' data, crawling is not required. 
# Go to Google Trends, type terms to compared and download csv data
# =============================================================================
from selenium import webdriver
from bs4 import BeautifulSoup
import pandas as pd

# Setting the driver

#Pour Firefox Linux
#driver = webdriver.Firefox(executable_path="./geckodriver")

#Pour Google Chrome Windows. Il faut installer cependant chromedrive: https://sites.google.com/a/chromium.org/chromedriver/downloads
driver = webdriver.Chrome(executable_path="C:\Program Files (x86)\Google\Chrome\Application\chromedriver_win32\chromedriver.exe")

# get the content of the html page
driver.get("https://basketballnoise.com/nba-players-height-2019-2020/")

def_ratings = {}

content = driver.page_source
soup = BeautifulSoup(content,features="lxml")
div = soup.find("div", attrs={'class':'entry-content'})
name = ""
value = 0

for tbody in div.findAll('tbody'):
    for tr in tbody.findAll("tr"):
        tds = tr.findAll("td")
        name = tds[0].text.replace("\n", "")
        value= tds[1].text
                 
        '''if (len(tds) == 3):
                value = str(tds[0].text)'''
        def_ratings[name] = value
    
def_ratings.pop('Name', None)
def_ratings.pop(str('Name '), None)
def_ratings.pop('', None)

df = pd.DataFrame(list(def_ratings.items()),columns = ['Name','Height (ft)'])
df['Height (ft)'] = df['Height (ft)'].str.replace('’','.')
df['Height (cm)'] = pd.to_numeric(df['Height (ft)'].str.split('.').str[0])*30.48+pd.to_numeric(df['Height (ft)'].str.split('.').str[1])*2.54

path="../csv/players_height.csv"
df.to_csv(path)
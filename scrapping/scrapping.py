from selenium import webdriver
from bs4 import BeautifulSoup
import pandas as pd

# Setting the driver
driver = webdriver.Firefox(executable_path="./geckodriver")

# get the content of the html page
driver.get("https://www.basketball-reference.com/leaders/def_rtg_active.html")

def_ratings = {}

content = driver.page_source
soup = BeautifulSoup(content)
div = soup.find("div", attrs={'class':'table_container'})
name = ""
value = 0
for tr in div.findAll("tr"):
	for strong in tr.findAll("strong"):
		name = strong.text
		name = name.replace("\n", "")
	tds = tr.findAll("td")
	if (len(tds) == 3):
		value = float(tds[2].text)
	def_ratings[name] = value

for item in def_ratings.items():
	print(item)
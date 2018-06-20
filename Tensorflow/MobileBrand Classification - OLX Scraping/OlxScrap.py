from selenium import webdriver 
from selenium.webdriver.common.by import By 
from selenium.webdriver.support.ui import WebDriverWait 
from selenium.webdriver.support import expected_conditions as EC 
from selenium.common.exceptions import TimeoutException
import os
import time
from selenium.webdriver.common.keys import Keys
import requests
from bs4 import BeautifulSoup
import urllib.request
import uuid
import sys

## url exmaple: https://www.olx.com.pk/apple/
def create_links(urll, page_count=15):
	url = urll+"?page=1"
	try:
		driver = webdriver.Chrome(executable_path='chromedriver')
		driver.implicitly_wait(30)
		driver.get(url)
		i = 1
		list_link = []
		while i < page_count:
			lisst = driver.find_elements_by_xpath("//*[@id=\"body-container\"]/div[1]/div/div[1]/li/a")
			for k in lisst:
				album_url = k.get_attribute('href')
				list_link.append(album_url)
				print(album_url)
			driver.get(urll+"?page="+str(i+1))
			print('-'*100)
			i = i + 1
			time.sleep(3)
			
		with open('linkss.txt', 'a+') as f:
			f.write('\n'.join(list_link))
	except:
		print("Unexpected error:", sys.exc_info()[0])

def creat_details():
	file_list = []
	file = open("linkss.txt", 'r') 
	for line in file: 
		file_list.append(line)

	for f in file_list:
		try:
			url = f.strip()
			headers = {
				'user-agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/53.0.2785.143 Safari/537.36'}

			r = requests.get(url, headers=headers)
			title_text = '-'
			location_text = '-'
			price_text = '-'
			title_text = '-'
			img = '-'
			description_text = '-'
			
			if r.status_code == 200:
				html = r.text
				soup = BeautifulSoup(html, 'lxml')
				title = soup.find('h1')
				if title is not None:
					title_text = title.text.strip()
					print(title_text)

				location = soup.find('strong', {'class': 'c2b small'})
				if location is not None:
					location_text = location.text.strip()
					
				try:
					price = soup.select('div > .xxxx-large')
					if price is not None:
						price_text = price[0].text.strip('Rs').replace(',', '')
				except:
					print("Error: Price ")
				images = soup.select('#bigGallery > li > a')
				img = [image['href'].strip() for image in images]

				description = soup.select('#textContent > p')
				if description is not None:
					description_text = description[0].text.strip()

			# Creating a dictionary Object
			item = {}
			item['title'] = title_text
			item['description'] = description_text
			item['location'] = location_text
			item['price'] = price_text
			## to dwonload image
			for im in img:
				urllib.request.urlretrieve(im, "test/"+str(uuid.uuid4()) + '.png')
			item['images'] = img
			return item
		except:
			print("Unexpected error:", sys.exc_info()[0])
		print(item)

create_links("https://www.olx.com.pk/apple/",page_count=2)
items  = creat_details()
print(items)

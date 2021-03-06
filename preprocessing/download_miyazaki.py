# -*- coding: utf-8 -*-

#Import libraries
from GoogleImageScrapper import GoogleImageScraper
import os

"""
Created on Sun Jul 12 11:02:06 2020
@author: OHyic
@modified / debugged: Minna + Katherine
"""
def scrape_google():
    #Define file path
    # webdriver_path = os.path.normpath(os.getcwd()+"\\webdriver\\chromedriver.exe")
    webdriver_path = "/usr/local/bin/chromedriver"
    image_path = "/Users/ksang/Documents/cs_2021/cs1430/ganilla_test/night"
    print(webdriver_path)
    print(image_path)

    #Add new search key into array ["cat","t-shirt","apple","orange","pear","fish"]
    search_keys= ["ghibli rainy wallpaper"]

    #Parameters
    number_of_images = 50
    headless = False
    min_resolution=(0,0)
    max_resolution=(3000,3000)

    #Main program
    for search_key in search_keys:
        image_scrapper = GoogleImageScraper(webdriver_path,image_path,search_key,number_of_images,headless,min_resolution,max_resolution)
        image_urls = image_scrapper.find_image_urls()
        image_scrapper.save_images(image_urls)


"""
@author: Minna + Katherine
"""

def rename_files():
    count = 1
    print(os.listdir("photos"))
    for filename in os.listdir("photos"):
        os.rename("photos/" + filename, "miyazaki_" + str(count) + ".jpg")
        count +=1


""""
Execute desired functions here
""""
# rename_files()
# scrape_google()
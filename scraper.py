import concurrent.futures
from bs4 import BeautifulSoup
import requests
import pandas as pd
import csv

sevdalinka_names = []
sevdalinka_info = []

def getUrl(url):
    response = getResponse(url)

    soup = BeautifulSoup(response.text, "html.parser")

    list_titles = soup.find_all(class_="list-title")
    links_list = []
    for element in list_titles:
        soup2 = BeautifulSoup(str(element), "html.parser")
        links_list.append("https://bascarsija.ba" + soup2.find("a")["href"])

    return links_list

def getSevdalinkaInfo(url):
    response = getResponse(url)

    soup = BeautifulSoup(response.text, "html.parser")

    title = soup.find_all(class_="article-title")[0]
    soup2 = BeautifulSoup(str(title), "html.parser")
    sevdalinkaTitle = soup2.find("a").get_text().replace("\t","").replace("\n","").replace(",",";")
    full_lyrics = ""
    lyrics = soup.find_all(class_="MsoNormal")

    for line in lyrics:
        soup3 = BeautifulSoup(str(line), "html.parser")
        try:
            full_lyrics += soup3.find("span").get_text().replace("\t","").replace("\n","").replace(",",";")
        except:
            full_lyrics += ""
        full_lyrics += "\n"
    dict1 = {sevdalinkaTitle: full_lyrics}
    print(dict1)
    return dict1

def getResponse(url, headers=None):
    if headers is None:
        headers = {
            "User-Agent": "Mozilla/5.0 (Linux; Android 10; K) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Mobile Safari/537.36,gzip(gfe)"
        }
    response = requests.get(url, headers=headers)

    if response.status_code != 200:
        print(f"Failed to retrieve the web page. Status code: {response.status_code}")
        return
    
    return response

def main():
    base_url = "https://bascarsija.ba/bascarsija/kultura/tekstovi-sevdalinki"
    urls = [f"{base_url}?start={i * 10}" for i in range(0, 49)]

    with concurrent.futures.ThreadPoolExecutor() as executor:
        results = list(executor.map(getUrl, urls))

    for result in results:
        if result is not None:
            sevdalinka_names.extend(result)
    print("main gotov")

def sevdalinka():
    with concurrent.futures.ThreadPoolExecutor() as executor:
        results = list(executor.map(getSevdalinkaInfo, sevdalinka_names))

    for result in results:
        if result is not None:
            sevdalinka_info.append(result)

if __name__ == "__main__":
    main()
    sevdalinka()
    df = pd.DataFrame(sevdalinka_info)
    df.to_csv("sevdalinke.csv")
    print("Done")

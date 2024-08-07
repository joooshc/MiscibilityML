import os
import json
import requests
from tqdm import tqdm
from bs4 import BeautifulSoup

class DDBScraper:

    def __init__(self, base_url, index_url):
        self.base_url = base_url
        self.index_url = index_url

    def get_urls(self):
        data = requests.get(self.index_url)
        new_urls = []
        soup = BeautifulSoup(data.text, "html.parser")

        for link in soup.find_all('a'):
            new_urls.append(link.get('href'))

        return new_urls

    def url_cleaner(self, new_urls):
        while None in new_urls:
            new_urls.remove(None)
        list2 = [x for x in new_urls if not "#" in x]
        list3 = [sub.replace(" ", "%20") for sub in list2]
        list4 = [x for x in list3 if not "ddb" in x]
        return list4

    def all_data_fetcher(self, new_urls):
        output_dict = {}
        urls_indexed = 0

        for i in tqdm(range(len(new_urls))):
            url = self.base_url + new_urls[i]
            page_data = requests.get(url)
            soup = BeautifulSoup(page_data.text, "html.parser")
            text = soup.get_text(separator='\n')  # Collect all text from the page
            output_dict.update({url: text})
            urls_indexed += 1

        return output_dict, urls_indexed

    def save_to_json(self, data, relative_filepath):
        script_dir = os.path.dirname(os.path.realpath(__file__))
        absolute_filepath = os.path.join(script_dir, relative_filepath)
        os.makedirs(os.path.dirname(absolute_filepath), exist_ok=True)

        with open(absolute_filepath, 'w') as fp:
            json.dump(data, fp)

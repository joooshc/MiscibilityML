from bs4 import BeautifulSoup
import requests
import pandas as pd
import json
from tqdm import tqdm

def get_urls(): #Uses the website index to fetch a list of urls
    url = "http://www.ddbst.com/en/EED/VLE/VLEindex.php"
    data = requests.get(url)
    new_urls = []
    soup = BeautifulSoup(data.text, "html.parser")

    for link in soup.find_all('a'):
        new_urls.append(link.get('href'))
    return new_urls

def url_cleaner(new_urls): #Removes the urls that are not needed, and cleans the rest
    while None in new_urls:
        new_urls.remove(None)
    list2 = [x for x in new_urls if not "#" in x]
    list3 = [sub.replace(" ", "%20") for sub in list2]
    list4 = [x for x in list3 if not "ddb" in x]
    del list4[0]
    return list4

def table_fetcher(url): #Fetches the data tables from the urls
    fetched_tables = pd.read_html(url, match = "[K]")

    for j in range(0, (len(fetched_tables))):

        if len(fetched_tables[j].index) < 4 or len(fetched_tables[j].columns) < 2:
            fetched_tables[j] = pd.DataFrame()
        else:
            continue
    
    if len(fetched_tables) == 0:
        combined_tables = fetched_tables[0]
    else:
        combined_tables = pd.concat(fetched_tables, axis=0, ignore_index=True)
            
    return combined_tables

def cas_fetcher(url): #Fetches CAS data
    fetched_cas = pd.read_html(url, match = "CAS")
    fetched_cas = fetched_cas[0]

    return fetched_cas

def source_fetcher(url): #Fetches source data
    fetched_source = pd.read_html(url, match = "Source")
    fetched_source = fetched_source[0]

    return fetched_source

def table_to_dict(table): #Converts the tables to dictionaries
    dict = {}
    cols = table.columns.tolist()

    for i in range(len(cols)):
        data = table[cols[i]].values.tolist()
        dict.update({cols[i]: data})

    return dict

def main():
    full_output = {}; source_output = {}

    new_urls = get_urls()
    clean_urls = url_cleaner(new_urls)
    for i in tqdm(range(0, len(clean_urls))):
        url = f"http://www.ddbst.com/en/EED/VLE/{clean_urls[i]}"
        fetched_table = table_fetcher(url)

        if fetched_table.empty == True:
            continue
        else:
            data_dict = table_to_dict(fetched_table)
            cas_dict = table_to_dict(cas_fetcher(url))
            source_dict = table_to_dict(source_fetcher(url))

            full_output.update({url : (data_dict|cas_dict)})
            source_output.update({url : source_dict})
            continue

    with open("Datasets/DDBscrapeddata/full_output.json", "w") as outfile:
        json.dump(full_output, outfile)

    with open("Datasets/DDBscrapeddata/sources.json", "w") as outfile:
        json.dump(source_output, outfile)

main()
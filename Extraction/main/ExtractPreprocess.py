import os, sys

# Directory adjustments to import other classes
current_directory = os.path.dirname(os.path.abspath(__file__))
root_directory = os.path.dirname(current_directory)
while os.path.basename(root_directory) != "PSDI-miscibility2":
    root_directory = os.path.dirname(root_directory)
scripts_directory = os.path.join(root_directory, "Scripts")
sys.path.append(scripts_directory)
sys.path.append(root_directory)

import pandas as pd
from DDBScraper import DDBScraper
from Preprocess import Preprocessing
from CASManager import CASManager

class ExtractPreprocess:
    def __init__(self):
        self.current_directory = os.path.dirname(os.path.abspath(__file__))
        self.root_directory = self.find_parent_directory(self.current_directory, "PSDI-miscibility2")

    def find_parent_directory(self, current_path, target_folder_name):
        """Searches for a folder name starting from a given path and goes upwards."""
        while os.path.basename(current_path) != target_folder_name:
            new_path = os.path.dirname(current_path)
            if new_path == current_path:
                raise Exception(f"Folder {target_folder_name} not found.")
            current_path = new_path
        return current_path

    def extract_preprocess(self, iupac_dataset_path, diff_dataset_raw_path, diff_dataset_log_scaled_path):
        PLOT_HIST = False
        PLOT_SCALED_HIST = False

        cas_manager = CASManager(self.root_directory)
        cas_dict, df = cas_manager.smiles_gen(iupac_dataset_path)
        cas_pubchem_props_dict = cas_manager.pubchem_props_gen(cas_dict)
        cas_rdkit_descriptors_dict = cas_manager.rdkit_descriptors_gen(cas_dict)
        _, _, _ = cas_manager.merge_dicts_to_df(df, cas_pubchem_props_dict, cas_rdkit_descriptors_dict)

        if not os.path.exists(diff_dataset_log_scaled_path):
            dataset_log = pd.read_csv(diff_dataset_raw_path).iloc[:, 4:]
            dataset = Preprocessing.process_dataset(dataset_log, 20, PLOT_HIST, PLOT_SCALED_HIST, self.root_directory)
            dataset.to_csv("diff_dataset_log_scaled.csv", index=False)
        else:
            dataset_log = pd.read_csv(diff_dataset_raw_path).iloc[:, 4:]
            dataset = Preprocessing.process_dataset(dataset_log, 20, PLOT_HIST, PLOT_SCALED_HIST, self.root_directory)
        
    def scrape_data(self):
        base_url = "http://www.ddbst.com/en/EED/VLE"
        index_url = base_url + "/VLEindex.php"
        scraper = DDBScraper(base_url, index_url)

        new_urls = scraper.get_urls()
        new_urls = scraper.url_cleaner(new_urls)
        output, urls_indexed = scraper.all_data_fetcher(new_urls)

        print(f"Indexed {urls_indexed} URLs.")
        filename = 'Datasets/DDBscrapeddata/alldata.json'
        scraper.save_to_json(output, filename)
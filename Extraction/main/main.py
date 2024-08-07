import warnings, os
from ExtractPreprocess import ExtractPreprocess

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    ep = ExtractPreprocess()

    iupac_dataset_path = os.path.join(ep.root_directory, "Datasets", "IUPAC", "IUPAC_dataset_combined.json")
    diff_dataset_log_scaled_path = os.path.join(ep.root_directory, "Datasets", "IUPAC", "diff_dataset_log_scaled.csv")
    diff_dataset_raw_path = os.path.join(ep.root_directory, "Datasets", "IUPAC", "diff_dataset_raw.csv")

    PREPROCESS = 1
    SCRAPE = 0

    if PREPROCESS:
        ep.extract_preprocess(iupac_dataset_path, diff_dataset_raw_path, diff_dataset_log_scaled_path)
    if SCRAPE:
        ep.scrape_data()

import zipfile
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import datetime

# Unzipping the provided dataset
with zipfile.ZipFile("incident+management+process+enriched+event+log.zip", 'r') as z:
    # Listing files in the zip archive
    file_names = z.namelist()
    # Loading the dataset (assuming the first file in the archive is the desired dataset)
    dataset_path = z.extract(file_names[0])

def custom_date_parser(date_str):
    try:
        # First, try the format "X-X-X H:M"
        return datetime.datetime.strptime(date_str, "%d-%m-%Y %H:%M")
    except ValueError:
        # If the above fails, try the format "X/X/X H:M"
        return datetime.datetime.strptime(date_str, "%d/%m/%Y %H:%M")


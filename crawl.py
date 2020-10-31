#%%
from google_images_download import google_images_download 
import os
os.getcwd()
import pandas as pd

#%%
"""crawl"""
response = google_images_download.googleimagesdownload()

face_shape = ["round face", "oval face", "square face", "heart shaped face", "triangular face"]

for shape in face_shape:
    #creating list of arguments
    arguments = {
        "keywords":shape, "limit":1000,"print_urls":True, "offset":0, "delay":10, "save_source":"urls"}   
    # download
    paths = response.download(arguments)   

#%%
""" merge """
os.chdir("/home/0756728/crawl_gogole_img/")

menu = {
    "oval_menu":['long face.txt','oblong face men.txt','oval face.txt'],
    "square_menu":['face shape square.txt','square face men.txt','square face.txt'],
    "heart_menu":['heart shaped face.txt'],
    "round_menu":['round face men.txt','round face.txt'],
    "tri_menu":['triangular face.txt']}
file_name = dict()
for face_type, files in menu.items():
    df = pd.DataFrame()
    ## merge
    for file in files:
        temp_df = pd.read_csv("downloads/"+file, sep='\t', header=None)
        df = df.append(temp_df, ignore_index=True)
    df.columns = ["file_path", "url"]
    ## check duplicated in a face type
    dup = df.index[df.duplicated(subset="url", keep="first")]
    print("all",df.shape[0], ", duplicate", dup.shape[0])
    df = df.drop(dup)
    df["type"] = face_type
    file_name[face_type] = df
## check duplicated with different face types
df = pd.concat(list(file_name.values()), ignore_index=True)
dup = df.index[df.duplicated(subset="url", keep=False)]
df = df.drop(dup)
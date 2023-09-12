import os
import zipfile
import urllib.request

import dropbox

def download_and_extract(url, target_dir, tmp_name):
    print("> Downloading segmentations..", flush=True)
    try:
        # Remove the zip file
        os.remove(tmp_name)
    except: 
        pass
    try:
        # Download the segmentations
        urllib.request.urlretrieve(url, tmp_name)
        print("> Downloaded ZIP file, extracting..", flush=True)

        # Extract to the folder with manual segmentations
        with zipfile.ZipFile(tmp_name, 'r') as zip_ref:
            zip_ref.extractall(target_dir)
        print("> Extracted segmentations to", target_dir, flush=True)
        
    except Exception as err: 
        print(err, flush=True)

def upload_to_dropbox(filename, dbx_filename, dbx_token):
    # Upload them to dropbox
    dbx = dropbox.Dropbox(dbx_token)
    with open(filename, 'rb') as f:
        dbx.files_upload(f.read(), dbx_filename, 
            mode=dropbox.files.WriteMode.overwrite)
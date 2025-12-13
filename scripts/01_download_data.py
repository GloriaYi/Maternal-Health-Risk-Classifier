# download_data.py
# Downloads and extracts a zip file from a specified URL to a local directory.
# Creates the target directory if it doesn't exist.
# Saves the downloaded zip file and extracts its contents.

import click
import requests
import zipfile
import os

@click.command()
@click.option('--url', type=str, help="URL of dataset to be downloaded")
@click.option('--write-to', type=str, help="Path to directory where raw data will be written to")

def main(url, write_to):
    """
    Download and extract a zip file from a given URL.

    Parameters
    ----------
    url : str
        The URL of the zip file to download.
    write_to : str
        The path to the directory where the zip file will be downloaded
        and extracted. The directory will be created if it does not exist.

    Returns
    -------
    None
        The function downloads the file and extracts its contents to the 
        specified directory.
    """

    # create directory if it does not already exist
    os.makedirs(write_to, exist_ok=True)
    
    # extract filename from URL
    file_name = url.split('/')[-1]
    zip_path = os.path.join(write_to, file_name)

    try:
        # existing download + save zip code
        request = requests.get(url)
        # download the zip file
        with open(zip_path, 'wb') as f:
            f.write(request.content)

        # extract the zip file
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(write_to)

        click.echo("Download successful. Using downloaded raw data.")

    except Exception:
        click.echo("Download failed. Falling back to existing local raw data.")
        
if __name__ == '__main__':
    main()
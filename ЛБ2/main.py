import filecmp
import unittest
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
import subprocess
import re
import os
import shutil
from pathlib import Path
from tkinter import filedialog
import requests

def launch_get_links_get_audio(link):
    launch_options = Options()
    launch_options.add_argument("--headless")
    launch_options.add_argument("--no-gpu")
    driver = webdriver.Chrome(options=launch_options)
    driver.get(link)
    log = driver.execute_script("var performance = window.performance || window.mozPerformance || window.msPerformance || window.webkitPerformance || {}; var network = performance.getEntries() || {}; return network;")
    filename = driver.find_element(by='class name', value='soundTitle__title').find_element(by="tag name",value='span').text
    link = None
    for item in log:
        if "playlist.m3u8" in item['name']:
            link = item['name']
            break
    elements = driver.find_elements(by='class name', value='sc-artwork')
    art = None
    for element in elements:
        if element.get_attribute("aria-role") != "img":
            continue

        style = element.get_attribute('style')
        match = re.search(r'background-image: url\((.*)\)', style)
        if match and "500x500" in match.group(1):
            art = match.group(1).replace("\"", "")
            break
    driver.quit()
    subprocess.call("ffmpeg.exe -protocol_whitelist file,http,https,tcp,tls -i \"{}\" -b:a 320k \"file.mp3\"".format(link), shell=True)
    return art, filename

def get_art(art):
    with requests.get(art, stream=True) as r:
        r.raise_for_status()
        with open("art.jpg", "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)

def embed_art(art):
    ext = subprocess.call(
        "ffmpeg -i file.mp3 -i art.jpg -map 0:a -map 1:0 -c:a copy -id3v2_version 3 \"file2.mp3\"".format(art), shell=True)
    os.remove("art.jpg")

def save(filename):
    target = filedialog.asksaveasfilename(defaultextension=".mp3", initialfile="{}.mp3".format(filename), title="Save as...", initialdir=Path.home() / "Desktop/")
    shutil.move("file2.mp3", target)

def cleanup():
    os.remove("file.mp3")

print("Enter Link")
link = input(">")
art, filename = launch_get_links_get_audio(link)
get_art(art)
embed_art(art)
save(filename)
cleanup()

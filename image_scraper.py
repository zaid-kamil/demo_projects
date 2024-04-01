import requests
from bs4 import BeautifulSoup

def get_soup(query='birds'):
    '''
    Returns a BeautifulSoup object containing the HTML source 
    of the Bing Images search page for the given query.
    '''
    url = f'https://www.bing.com/images/search?q={query}'
    page = requests.get(url)
    if page.status_code == 200:
        return BeautifulSoup(page.text, 'html.parser')
    else:
        print('Error: Could not get page.')
        print('Status code:', page.status_code)

def extract_images(soup):
    images = []
    for link in soup.find_all('img'):
        src = link.get('src2')
        if src:
            images.append(src)
    print('Found', len(images), 'images')
    return images

def download_image(url, folder, pos=1):
    w=256
    h=256
    url = url.replace('w=42',f'w={w}').replace('h=42',f'h={h}')
    page = requests.get(url)
    if page.status_code == 200:
        filename = f'image_{pos}.jpg'
        filepath = f'{folder}/{filename}'
        with open(filepath, 'wb') as f:
            f.write(page.content)
        print('Downloaded', filename)
    else:
        print('Error: Could not download image.')
        print('Status code:', page.status_code)

if __name__ == '__main__':
    soup = get_soup('birds')
    images = extract_images(soup)
    for idx,image in enumerate(images):
        if idx > 50: break
        download_image(image, 'images', idx+1)
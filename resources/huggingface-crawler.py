import re
import time
import json
import requests
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

def replace_multiple_newlines(text):
    return re.sub(r'\n+', '\n', text)

# Chrome WebDriver 설정
chrome_options = Options()
# chrome_options.add_argument("--headless")  # 백그라운드 실행
chrome_options.add_argument("--disable-gpu")
chrome_options.add_argument("--no-sandbox")

# HuggingFace 로그인 정보
username = ''
password = ''

# Selenium을 사용하여 HuggingFace에 로그인
driver = webdriver.Chrome(options=chrome_options)
driver.get('https://huggingface.co/login')

# 로그인 폼 입력
username_input = WebDriverWait(driver, 10).until(
    EC.presence_of_element_located((By.NAME, 'username'))
)
username_input.send_keys(username)
password_input = driver.find_element(By.NAME, 'password')
password_input.send_keys(password)
password_input.send_keys(Keys.RETURN)

# 로그인 후 안정성을 위해 잠시 대기
time.sleep(3)

driver.get('https://huggingface.co/BISPL-KAIST')

# BeautifulSoup을 사용하여 모델 정보 크롤링
soup = BeautifulSoup(driver.page_source, 'html.parser')

models = []
model_elements = soup.find_all('a', class_='block p-2')

for model_element in model_elements:
    model_link = model_element['href']
    model_page_url = f"https://huggingface.co{model_link}"
    print(model_page_url)
    # 각 모델의 상세 페이지로 이동하여 정보 크롤링
    driver.get(model_page_url)
    model_soup = BeautifulSoup(driver.page_source, 'html.parser')
    
    organization_name = model_soup.find('a', class_='text-gray-400 hover:text-blue-600').text.strip()
    model_id = model_link.split('/')[-1]
    try:
        model_card_content = replace_multiple_newlines(model_soup.find('div', class_='model-card-content').text.strip())
    except:
        model_card_content = ''
    
    models.append({
        'model_name': organization_name + '/' + model_id,
        'model_id': model_id,
        'model_card_content': model_card_content
    })

# 결과를 JSON Lines 파일로 저장
with open('BISPL-KAIST_models.jsonl', 'w', encoding='utf-8') as f:
    for model in models:
        json.dump(model, f, ensure_ascii=False)
        f.write('\n')

# 드라이버 종료
driver.quit()

import requests
import time
import os
from urllib.parse import urlparse
import hashlib
import json

# ===================== 基础配置（按需修改）=====================
# 1. 你的Cookie（从浏览器复制，关键！）
cookies = {
    'BIDUPSID': 'C4210F4FCE75E84D7CBA7F6D7D2659D3',
    'PSTM': '1761921363',
    'BAIDUID': 'B610A82293895758157FE5971ACA8A1E:FG=1',
    'H_PS_PSSID': '60272_63140_64004_64979_65250_65313_65361_65588_65604_65759_65778_65789_65843_65852_65942_65953_65960_65971_65999_66076_66099_66111_65636_65866',
    'BDUSS_BFESS': 'FNeE5JbmtMcTBQMDQxSXJpaDJJeEFDUzBCMlp-d0lhbUN1flhlS01mZkZZaXhwSUFBQUFBJCQAAAAAAQAAAAEAAAAdzH94wuTEu9PAsrvC5MS7AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAMXVBGnF1QRpZ0',
    'BAIDUID_BFESS': 'B610A82293895758157FE5971ACA8A1E:FG=1',
    'ZFY': 'P1g9sAm:AIqf43uLXGWYX3mblTC0WbgD95aBXu18:BOBM:C',
    'BDRCVFR[BIVAaPonX6T]': '-_EV5wtlMr0mh-8uz4WUvY',
    'BA_HECTOR': '8h2gal0ka18l04a5a52h8ha0alaha31kgga6h24',
    'PSINO': '3',
    'delPer': '0',
    'BDORZ': 'FFFB88E999055A3F8A630C64834BD6D0',
    'H_WISE_SIDS': '64979_65250_65313_65361_65604_65778_65789_65852_65942_65953_65999_66076_66099_66111_65636',
}

# 2. 请求头（模拟浏览器，无需修改）
headers = {
    'Accept': 'application/json, text/plain, */*',
    'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8,en-GB;q=0.7,en-US;q=0.6',
    'Connection': 'keep-alive',
    'Referer': 'https://image.baidu.com/search/index?tn=baiduimage&ps=1&ct=201326592&lm=-1&cl=2&nc=1&ie=utf-8&lid=b88e601d000412cf&dyTabStr=MTIsMCwzLDEsMiwxMyw3LDYsNSw5&word=%E8%8D%89%E4%B9%A6',
    'Sec-Fetch-Dest': 'empty',
    'Sec-Fetch-Mode': 'cors',
    'Sec-Fetch-Site': 'same-origin',
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/142.0.0.0 Safari/537.36 Edg/142.0.0.0',
    'sec-ch-ua': '"Chromium";v="142", "Microsoft Edge";v="142", "Not_A Brand";v="99"',
    'sec-ch-ua-mobile': '?0',
    'sec-ch-ua-platform': '"Windows"',
    # 'Cookie': 'BIDUPSID=C4210F4FCE75E84D7CBA7F6D7D2659D3; PSTM=1761921363; BAIDUID=B610A82293895758157FE5971ACA8A1E:FG=1; H_PS_PSSID=60272_63140_64004_64979_65250_65313_65361_65588_65604_65759_65778_65789_65843_65852_65942_65953_65960_65971_65999_66076_66099_66111_65636_65866; BDUSS_BFESS=FNeE5JbmtMcTBQMDQxSXJpaDJJeEFDUzBCMlp-d0lhbUN1flhlS01mZkZZaXhwSUFBQUFBJCQAAAAAAQAAAAEAAAAdzH94wuTEu9PAsrvC5MS7AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAMXVBGnF1QRpZ0; BAIDUID_BFESS=B610A82293895758157FE5971ACA8A1E:FG=1; ZFY=P1g9sAm:AIqf43uLXGWYX3mblTC0WbgD95aBXu18:BOBM:C; BDRCVFR[BIVAaPonX6T]=-_EV5wtlMr0mh-8uz4WUvY; BA_HECTOR=8h2gal0ka18l04a5a52h8ha0alaha31kgga6h24; PSINO=3; delPer=0; BDORZ=FFFB88E999055A3F8A630C64834BD6D0; H_WISE_SIDS=64979_65250_65313_65361_65604_65778_65789_65852_65942_65953_65999_66076_66099_66111_65636',
}
# 3. 爬取配置
KEYWORD = '草书'  # 搜索关键词（可修改为其他内容）
SAVE_DIR = 'caoshu'  # 保存图片的文件夹名
MAX_PAGES = 10  # 爬取页数（1页=30张，5页=150张，建议不超过10页）
DELAY = 2  # 每页请求间隔（秒，建议1-3秒，防反爬）
RECORD_FILE = 'last_page_cao.json'  # 保存上次进度的文件

# 读取上次爬取进度
if os.path.exists(RECORD_FILE):
    with open(RECORD_FILE, 'r', encoding='utf-8') as f:
        record = json.load(f)
        START_PAGE = record.get(KEYWORD, 0) + 1
else:
    record = {}
    START_PAGE = 1

END_PAGE = START_PAGE + MAX_PAGES - 1
print(f"📘 本次将爬取关键词「{KEYWORD}」的第 {START_PAGE} 页到第 {END_PAGE} 页。")

# 4. 接口基础参数（无需修改）
base_params = {
    'tn': 'resultjson_com',
    'word': KEYWORD,
    'ie': 'utf-8',
    'fp': 'result',
    'rn': '30',  # 每页固定30张（百度接口最大限制）
    'nojc': '0',
    'gsm': '3c',
    'newReq': '1',
}

# ===================== 工具函数 =====================
def create_save_dir():
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)
    print(f'📁 图片将保存到：{os.path.abspath(SAVE_DIR)}')

def download_img(img_url, save_path):
    """下载单张图片"""
    try:
        resp = requests.get(img_url, headers=headers, cookies=cookies, timeout=10, stream=True)
        resp.raise_for_status()
        with open(save_path, 'wb') as f:
            for chunk in resp.iter_content(1024):
                f.write(chunk)
        return True
    except Exception as e:
        print(f'❌ 下载失败 {img_url}：{str(e)[:50]}')
        return False

def get_img_filename(img_url):
    """生成唯一文件名"""
    url_hash = hashlib.md5(img_url.encode()).hexdigest()
    ext = os.path.splitext(urlparse(img_url).path)[1]
    ext = ext[:5] if ext else '.jpg'
    return f'{url_hash}{ext}'

# ===================== 主爬取逻辑 =====================
def main():
    create_save_dir()
    all_img_urls = set()
    success_count = fail_count = 0

    for page in range(START_PAGE, END_PAGE + 1):
        pn = (page - 1) * 30
        base_params['pn'] = pn

        try:
            time.sleep(DELAY)
            print(f'\n🔍 正在爬取第 {page} 页...')
            response = requests.get(
                'https://image.baidu.com/search/acjson',
                params=base_params, cookies=cookies, headers=headers, timeout=10
            )
            data = response.json()
            img_data_list = data['data']['images']

            current_urls = []
            for item in img_data_list:
                if isinstance(item, dict):
                    url = item.get('objurl')
                    if url and url not in all_img_urls:
                        current_urls.append(url)
                        all_img_urls.add(url)

            print(f'📸 共找到 {len(current_urls)} 张图片')
            for img_url in current_urls:
                filename = get_img_filename(img_url)
                save_path = os.path.join(SAVE_DIR, filename)
                if download_img(img_url, save_path):
                    success_count += 1
                else:
                    fail_count += 1

            # ✅ 每爬完一页就保存当前进度
            record[KEYWORD] = page
            with open(RECORD_FILE, 'w', encoding='utf-8') as f:
                json.dump(record, f, ensure_ascii=False, indent=2)

        except Exception as e:
            print(f'⚠️ 第 {page} 页出错：{str(e)}')
            continue

    print('\n' + '='*60)
    print(f'✅ 本次共爬取 {START_PAGE}-{END_PAGE} 页')
    print(f'📥 成功下载：{success_count} 张 | ❌ 失败：{fail_count} 张')
    print(f'📂 图片保存路径：{os.path.abspath(SAVE_DIR)}')

if __name__ == '__main__':
    main()
import requests
import re
import os

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 \
        (KHTML, like Gecko) Chrome/83.0.4103.97 Safari/537.36"
}


def download():
    r = requests.get(url=down_url, headers=headers)
    if not os.path.exists("./pdf"):
        os.mkdir("./pdf")
    # 直接下载pdf
    with open(f"./pdf/{os.path.basename(down_url)}", "wb") as f:
        f.write(r.content)


if __name__ == "__main__":

    url = "https://pubmed.ncbi.nlm.nih.gov/"
    term = input("Please input your keyword: ").split(" ")
    size = 200
    page = 1
    param = {
        "term": term,
        "size": size,
        "page": page
    }
    doi_list = []
    response = requests.get(url=url, params=param, headers=headers)
    page_text = response.text
    results_amount = int(re.search(r"""<span class="value">(\d+(?:,?\d+)?)</span>.*?results""", page_text,
                                   re.DOTALL).group(1).replace(",", ""))
    doi_list += re.findall(r"""doi: (10\..*?)\.[ <]""", page_text)
    if results_amount % 200 == 0:
        step_num = results_amount / 200 - 1
    else:
        step_num = results_amount // 200
    if step_num:
        for page in range(2, step_num + 2):
            size = 200
            page = page
            param = {
                "term": term,
                "size": size,
                "page": page
            }
            response = requests.get(url=url, params=param, headers=headers)
            page_text = response.text
            doi_list += re.findall(r"""doi: (10\..*?)\.[ <]""", page_text)

    # 从sci-hub下载
    for doi in doi_list:
        down_url = r"https://sci.bban.top/pdf/" + doi + ".pdf"
        # 将下载地址写入文件
        with open(r"./down_url.txt", "a") as u:
            u.write(down_url + "\n")

        download()

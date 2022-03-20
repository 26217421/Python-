import os
import requests

down_url = "https://sci.bban.top/pdf/10.1248/bpb.28.1731.pdf"
r = requests.get(url=down_url)
with open(f"./pdf/{os.path.basename(down_url)}", "wb") as f:
    f.write(r.content)
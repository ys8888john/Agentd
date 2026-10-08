# -*- coding: utf-8 -*-
"""验证智谱 web_search API（key 从 models.json 读取，不打印明文）。"""
import io
import json
import sys
import urllib.request

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

data = json.load(open(r"C:\Users\Administrator\.agentd\gui\models.json", encoding="utf-8"))
prof = next(p for p in data["profiles"] if p["id"] == "zhipu-glm")
key = prof["env"]["AGENTD_ZHIPU_API_KEY"]

def search(engine: str, query: str, count: int = 5) -> None:
    body = json.dumps({
        "search_engine": engine,
        "search_query": query,
        "count": count,
    }).encode()
    req = urllib.request.Request(
        "https://open.bigmodel.cn/api/paas/v4/web_search",
        data=body,
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"},
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            out = json.loads(resp.read().decode("utf-8"))
        items = out.get("search_result", [])
        print(f"--- {engine}: {len(items)} 条")
        for it in items[:3]:
            print(f"   [{it.get('media','')}] {it.get('title','')[:40]}")
            print(f"     {it.get('link','')[:70]}")
            print(f"     {it.get('content','')[:80]}")
    except urllib.error.HTTPError as e:
        print(f"--- {engine}: HTTP {e.code} {e.read().decode('utf-8', 'replace')[:200]}")
    except Exception as e:
        print(f"--- {engine}: {type(e).__name__}: {e}")


search("search_std", "成都到北京航班时刻表 航班号 起降时间")
search("search_pro", "成都到北京航班时刻表 航班号 起降时间")
search("search_pro_sogou", "成都到北京航班时刻表 航班号 起降时间")

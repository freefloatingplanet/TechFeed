import os, re, time, hashlib
from datetime import datetime, timezone
from dateutil import parser as dtp
import feedparser, yaml
from feedgen.feed import FeedGenerator
from bs4 import BeautifulSoup

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---------- config ----------
CFG = yaml.safe_load(open("config.yaml", "r", encoding="utf-8"))
TOP_K = CFG.get("top_k", 50)
HALF_LIFE = CFG.get("half_life_days", 7)
BLOCK_KW = set(CFG.get("block_keywords", []))
MIN_TITLE = CFG.get("min_title_len", 12)

# ---------- utils ----------
def strip_html(s):
    return BeautifulSoup(s or "", "html.parser").get_text(" ", strip=True)

def norm(s):
    s = strip_html(s or "")
    s = re.sub(r"\s+", " ", s).strip().lower()
    return s

def recency_weight(published_dt, now):
    # 半減期ベースの減衰: w = 0.5 ** (days / HALF_LIFE)
    days = max(0.0, (now - published_dt).total_seconds() / 86400.0) if published_dt else 999
    return 0.5 ** (days / max(1e-6, HALF_LIFE))

def entry_dt(e):
    for key in ("published_parsed","updated_parsed"):
        if getattr(e, key, None):
            t = getattr(e, key)
            try:
                return datetime(*t[:6], tzinfo=timezone.utc)
            except Exception:
                pass
    # fallback: try parsed date field
    for key in ("published","updated"):
        v = e.get(key)
        if v:
            try:
                return dtp.parse(v).astimezone(timezone.utc)
            except Exception:
                pass
    return None

def dedup_key(title, link):
    base = (norm(title)[:80] + "|" + (link or "")).encode("utf-8")
    return hashlib.md5(base).hexdigest()

# ---------- load OPML ----------
def parse_opml(path):
    urls = []
    with open(path, "r", encoding="utf-8") as f:
        txt = f.read()
    for m in re.finditer(r'xmlUrl="([^"]+)"', txt):
        urls.append(m.group(1))
    return sorted(set(urls))

# ---------- load likes ----------
def load_likes_csv(path):
    likes = []
    if not os.path.exists(path):
        return likes
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i == 0 and line.lower().startswith("url,"):
                continue
            parts = line.strip().split(",", 1)
            if parts:
                url = parts[0].strip()
                title = parts[1].strip() if len(parts) > 1 else ""
                likes.append({"url": url, "title": title})
    return likes

# ---------- fetch candidates ----------
def fetch_candidates(feed_urls, max_per_feed=100):
    cand = []
    for u in feed_urls:
        try:
            d = feedparser.parse(u)
            for e in d.entries[:max_per_feed]:
                title = e.get("title", "")
                summary = e.get("summary", "")
                link = e.get("link", "")
                if len(title.strip()) < MIN_TITLE:
                    continue
                # NGワード
                text = (title + " " + strip_html(summary)).lower()
                if any(kw.lower() in text for kw in BLOCK_KW):
                    continue
                cand.append({
                    "title": title,
                    "summary": summary,
                    "link": link,
                    "published": entry_dt(e),
                    "source": d.feed.get("title", u)
                })
        except Exception:
            continue
    return cand

# ---------- scoring ----------
def score_candidates(candidates, likes):
    if not likes:
        # ライクが無い場合は新しさ順
        now = datetime.now(timezone.utc)
        for c in candidates:
            c["score"] = recency_weight(c["published"], now)
        return candidates

    liked_texts = [norm(x.get("title","")) for x in likes if x.get("title")]
    if not liked_texts:
        liked_texts = [x.get("url","") for x in likes]

    cand_texts = [norm(c["title"] + " " + c["summary"]) for c in candidates]

    vec = TfidfVectorizer(max_features=8000, ngram_range=(1,2))
    X_liked = vec.fit_transform(liked_texts)
    X_cand  = vec.transform(cand_texts)

    sims = cosine_similarity(X_cand, X_liked).mean(axis=1)

    now = datetime.now(timezone.utc)
    out = []
    seen = set()
    for c, s in zip(candidates, sims):
        w = recency_weight(c["published"], now)
        score = 0.7 * s + 0.3 * w  # 類似度7:新しさ3 の重み
        k = dedup_key(c["title"], c["link"])
        if k in seen:
            continue
        seen.add(k)
        c2 = dict(c)
        c2["score"] = float(score)
        out.append(c2)
    out.sort(key=lambda x: x["score"], reverse=True)
    return out

# ---------- generate RSS ----------
def write_rss(items, path_xml):
    fg = FeedGenerator()
    fg.title(CFG.get("title","My Reco Feed"))
    fg.link(href=CFG.get("site_url","https://example.com"), rel='alternate')
    fg.description(CFG.get("description","Recommended tech news"))
    fg.language('en')

    for it in items:
        fe = fg.add_entry()
        fe.title(it["title"])
        fe.link(href=it["link"])
        desc = strip_html(it["summary"])[:500]
        fe.description(f"{desc} (source: {it.get('source','?')})")
        pub = it["published"] or datetime.now(timezone.utc)
        fe.pubDate(pub)

        # 安定したGUID（リンクがあればそれ、なければハッシュ）
        guid = it["link"] or hashlib.md5((it["title"]+str(pub)).encode("utf-8")).hexdigest()
        fe.guid(guid, permalink=bool(it["link"]))

    os.makedirs(os.path.dirname(path_xml), exist_ok=True)
    fg.rss_file(path_xml, pretty=True)
    print(f"Wrote {path_xml} ({len(items)} items)")

def main():
    feeds = parse_opml("data/feeds.opml")
    likes = load_likes_csv("data/likes.csv")
    cands = fetch_candidates(feeds)
    ranked = score_candidates(cands, likes)[:TOP_K]
    write_rss(ranked, "docs/recommended.xml")

if __name__ == "__main__":
    main()

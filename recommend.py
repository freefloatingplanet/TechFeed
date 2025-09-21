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
            # split into three parts at most: url,title,summary
            parts = line.strip().split(",", 2)
            if parts:
                url = parts[0].strip()
                title = parts[1].strip() if len(parts) > 1 else ""
                summary = parts[2].strip() if len(parts) > 2 else ""
                # If summary exists, merge it into title for matching convenience
                combined = title
                if summary:
                    combined = (title + " - " + summary).strip()
                likes.append({"url": url, "title": title, "combined": combined})
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

    # prefer combined (title + summary) if available
    liked_texts = [norm(x.get("combined") or x.get("title") or "") for x in likes if (x.get("combined") or x.get("title"))]
    if not liked_texts:
        liked_texts = [x.get("url","") for x in likes]

    cand_texts = [norm(c["title"] + " " + c["summary"]) for c in candidates]

    vec = TfidfVectorizer(max_features=8000, ngram_range=(1,2))
    X_liked = vec.fit_transform(liked_texts)
    X_cand  = vec.transform(cand_texts)
    sims = cosine_similarity(X_cand, X_liked).mean(axis=1)

    now = datetime.now(timezone.utc)
    out, seen = [], set()
    for c, s in zip(candidates, sims):
        w = recency_weight(c["published"], now)
        final = 0.995 * float(s) + 0.005 * float(w)  # 類似度,新しさの重み付け和
        k = dedup_key(c["title"], c["link"])
        if k in seen:
            continue
        seen.add(k)
        it = dict(c)
        it["sim"] = float(s)       # ← 類似度を保持
        it["score"] = final        # ← 最終スコアを保持
        print(f"SIM {s:.3f}  W {w:.3f}  => {final:.3f}  | {c['title']}")
        out.append(it)

    out.sort(key=lambda x: x["score"], reverse=True)
    return post_filter_with_threshold(out)

def post_filter_with_threshold(sorted_items):
    """
    1) しきい値でフィルタ
    2) 件数が足りなければ min_items までスコア順で埋める
    3) 最後に top_k で切る
    """
    mode = CFG.get("apply_threshold_on", "similarity")  # "similarity" or "final_score"
    use_pct = CFG.get("use_percentile_threshold", True)
    p = CFG.get("percentile_p", 0.8)
    top_k = CFG.get("top_k", 50)
    min_items = CFG.get("min_items", top_k)

    if mode == "final_score":
        base_vals = [it["score"] for it in sorted_items]
        fixed_thr = CFG.get("score_threshold", None)
    else:
        base_vals = [it["sim"] for it in sorted_items]
        fixed_thr = CFG.get("min_similarity", None)

    dyn_thr = percentile(base_vals, p) if (use_pct and base_vals) else None

    # 有効なしきい値（固定と動的の“高い方”）
    thr_list = [x for x in (fixed_thr, dyn_thr) if isinstance(x, (int, float))]
    effective_thr = max(thr_list) if thr_list else None

    if effective_thr is not None:
        passed = [it for it in sorted_items
                  if (it["score"] if mode == "final_score" else it["sim"]) >= effective_thr]
    else:
        passed = list(sorted_items)

    # 最低件数まで埋める
    if len(passed) < min_items:
        used_ids = {id(x) for x in passed}
        for it in sorted_items:
            if id(it) not in used_ids:
                passed.append(it)
                if len(passed) >= min_items:
                    break

    # 上限でカット
    return passed[:top_k]

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

def percentile(values, p):
    if not values:
        return None
    vs = sorted(values)
    k = int(round((len(vs)-1) * float(p)))
    k = max(0, min(k, len(vs)-1))
    return vs[k]

def main():
    feeds = parse_opml("data/feeds.opml")
    likes = load_likes_csv("data/likes.csv")
    cands = fetch_candidates(feeds)
    ranked = score_candidates(cands, likes)[:TOP_K]
    write_rss(ranked, "docs/recommended.xml")

if __name__ == "__main__":
    main()

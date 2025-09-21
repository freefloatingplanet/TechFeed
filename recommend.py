import argparse
import hashlib
import os
import re
from datetime import datetime, timezone

import feedparser
import yaml
from bs4 import BeautifulSoup
from dateutil import parser as dtp
from feedgen.feed import FeedGenerator
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def strip_html(s):
    return BeautifulSoup(s or "", "html.parser").get_text(" ", strip=True)


def norm(s):
    s = strip_html(s or "")
    s = re.sub(r"\s+", " ", s).strip().lower()
    return s


def recency_weight(published_dt, now, half_life_days):
    hl = half_life_days if half_life_days is not None else 7
    hl = max(hl, 1e-6)
    if not published_dt:
        days = 999
    else:
        days = max(0.0, (now - published_dt).total_seconds() / 86400.0)
    return 0.5 ** (days / hl)


def entry_dt(e):
    for key in ("published_parsed", "updated_parsed"):
        if getattr(e, key, None):
            t = getattr(e, key)
            try:
                return datetime(*t[:6], tzinfo=timezone.utc)
            except Exception:
                pass
    for key in ("published", "updated"):
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


def parse_opml(path):
    urls = []
    with open(path, "r", encoding="utf-8") as f:
        txt = f.read()
    for m in re.finditer(r'xmlUrl="([^"]+)"', txt):
        urls.append(m.group(1))
    return sorted(set(urls))


def load_likes_csv(path):
    likes = []
    if not os.path.exists(path):
        return likes
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i == 0 and line.lower().startswith("url,"):
                continue
            parts = line.strip().split(",", 2)
            if parts:
                url = parts[0].strip()
                title = parts[1].strip() if len(parts) > 1 else ""
                summary = parts[2].strip() if len(parts) > 2 else ""
                combined = title
                if summary:
                    combined = (title + " - " + summary).strip()
                likes.append({
                    "url": url,
                    "title": title,
                    "combined": combined,
                })
    return likes


def fetch_candidates(feed_urls, cfg, max_per_feed=None):
    candidates = []
    limit = max_per_feed
    if limit is None:
        limit = cfg.get("max_entries_per_feed", cfg.get("max_per_feed", 100))
    try:
        limit = int(limit)
    except (TypeError, ValueError):
        limit = 100
    min_title = cfg.get("min_title_len", 12)
    block_kw = {str(kw).lower() for kw in cfg.get("block_keywords", [])}

    for u in feed_urls:
        try:
            parsed = feedparser.parse(u)
        except Exception:
            continue
        entries = parsed.entries
        if limit:
            entries = entries[:limit]
        for e in entries:
            title = e.get("title", "")
            summary = e.get("summary", "")
            link = e.get("link", "")
            if len(title.strip()) < min_title:
                continue
            text = (title + " " + strip_html(summary)).lower()
            if block_kw and any(kw in text for kw in block_kw):
                continue
            candidates.append({
                "title": title,
                "summary": summary,
                "link": link,
                "published": entry_dt(e),
                "source": parsed.feed.get("title", u),
            })
    return candidates


def score_candidates(candidates, likes, cfg):
    if not candidates:
        return []

    half_life = cfg.get("half_life_days", 7)
    label = cfg.get("name", "default")
    log_similarity = cfg.get("log_similarity", True)
    now = datetime.now(timezone.utc)

    if not likes:
        for c in candidates:
            c["score"] = recency_weight(c.get("published"), now, half_life)
        candidates.sort(key=lambda x: x.get("score", 0.0), reverse=True)
        return candidates

    liked_texts = [
        norm(x.get("combined") or x.get("title") or "")
        for x in likes
        if (x.get("combined") or x.get("title"))
    ]
    if not liked_texts:
        liked_texts = [x.get("url", "") for x in likes]

    cand_texts = [norm((c.get("title") or "") + " " + (c.get("summary") or "")) for c in candidates]
    if not cand_texts:
        return []

    vec = TfidfVectorizer(max_features=cfg.get("tfidf_max_features", 8000), ngram_range=(1, 2))
    X_liked = vec.fit_transform(liked_texts)
    X_cand = vec.transform(cand_texts)
    sims = cosine_similarity(X_cand, X_liked).mean(axis=1)

    out, seen = [], set()
    for c, s in zip(candidates, sims):
        w = recency_weight(c.get("published"), now, half_life)
        final = 0.995 * float(s) + 0.005 * float(w)
        k = dedup_key(c.get("title"), c.get("link"))
        if k in seen:
            continue
        seen.add(k)
        item = dict(c)
        item["sim"] = float(s)
        item["score"] = final
        if log_similarity:
            print(f"[{label}] SIM {s:.3f}  W {w:.3f}  => {final:.3f}  | {c.get('title', '')}")
        out.append(item)

    out.sort(key=lambda x: x["score"], reverse=True)
    return post_filter_with_threshold(out, cfg)


def post_filter_with_threshold(sorted_items, cfg):
    mode = cfg.get("apply_threshold_on", "similarity")
    use_pct = cfg.get("use_percentile_threshold", True)
    p = cfg.get("percentile_p", 0.8)
    top_k = cfg.get("top_k", 50)
    min_items = cfg.get("min_items", top_k)

    if mode == "final_score":
        base_vals = [it.get("score", 0.0) for it in sorted_items]
        fixed_thr = cfg.get("score_threshold")
    else:
        base_vals = [it.get("sim", 0.0) for it in sorted_items]
        fixed_thr = cfg.get("min_similarity")

    dyn_thr = percentile(base_vals, p) if (use_pct and base_vals) else None
    thr_list = [x for x in (fixed_thr, dyn_thr) if isinstance(x, (int, float))]
    effective_thr = max(thr_list) if thr_list else None

    if effective_thr is not None:
        passed = [
            it for it in sorted_items
            if (it.get("score") if mode == "final_score" else it.get("sim")) >= effective_thr
        ]
    else:
        passed = list(sorted_items)

    if len(passed) < min_items:
        used_ids = {id(x) for x in passed}
        for it in sorted_items:
            if id(it) not in used_ids:
                passed.append(it)
                if len(passed) >= min_items:
                    break

    return passed[:top_k]


def percentile(values, p):
    if not values:
        return None
    vs = sorted(values)
    k = int(round((len(vs) - 1) * float(p)))
    k = max(0, min(k, len(vs) - 1))
    return vs[k]


def write_rss(items, path_xml, cfg):
    fg = FeedGenerator()
    fg.title(cfg.get("title", "My Reco Feed"))
    fg.link(href=cfg.get("site_url", "https://example.com"), rel="alternate")
    fg.description(cfg.get("description", "Recommended tech news"))
    fg.language(cfg.get("language", "en"))

    for it in items:
        fe = fg.add_entry()
        fe.title(it.get("title", ""))
        if it.get("link"):
            fe.link(href=it.get("link"))
        desc = strip_html(it.get("summary"))[:500]
        fe.description(f"{desc} (source: {it.get('source', '?')})")
        pub = it.get("published") or datetime.now(timezone.utc)
        fe.pubDate(pub)
        guid_src = it.get("link") or hashlib.md5((it.get("title", "") + str(pub)).encode("utf-8")).hexdigest()
        fe.guid(guid_src, permalink=bool(it.get("link")))

    dir_path = os.path.dirname(path_xml)
    if dir_path:
        os.makedirs(dir_path, exist_ok=True)
    fg.rss_file(path_xml, pretty=True)
    print(f"[{cfg.get('name', 'default')}] Wrote {path_xml} ({len(items)} items)")


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError("config.yaml はマッピング形式で記述してください")
    return data


def build_profiles(raw_cfg):
    base = dict(raw_cfg or {})
    profiles = base.pop("profiles", None)
    if not profiles:
        cfg = dict(base)
        cfg.setdefault("name", cfg.get("profile_name") or "default")
        return [(cfg["name"], cfg)]

    result = []
    for idx, prof in enumerate(profiles):
        if not isinstance(prof, dict):
            continue
        merged = dict(base)
        merged.update(prof)
        merged.setdefault("name", prof.get("name") or prof.get("profile_name") or f"profile_{idx + 1}")
        result.append((merged["name"], merged))
    return result


def run_for_cfg(cfg):
    label = cfg.get("name", "default")
    feeds_path = cfg.get("feeds_opml") or cfg.get("feeds") or cfg.get("feeds_path") or "data/feeds.opml"
    likes_path = cfg.get("likes_csv") or cfg.get("likes") or cfg.get("likes_path") or "data/likes.csv"
    output_path = cfg.get("output_rss") or cfg.get("output") or cfg.get("output_path") or "docs/recommended.xml"

    if not os.path.exists(feeds_path):
        raise FileNotFoundError(f"[{label}] OPML ファイルが見つかりません: {feeds_path}")

    feed_urls = parse_opml(feeds_path)
    likes = load_likes_csv(likes_path)
    candidates = fetch_candidates(feed_urls, cfg)
    print(f"[{label}] feeds={len(feed_urls)} likes={len(likes)} candidates={len(candidates)}")

    ranked = score_candidates(candidates, likes, cfg)
    top_k = cfg.get("top_k", 50)
    ranked = ranked[:top_k]
    write_rss(ranked, output_path, cfg)


def parse_args():
    parser = argparse.ArgumentParser(description="Generate personalised RSS feeds")
    parser.add_argument("--config", default="config.yaml", help="読み込む設定ファイルのパス")
    parser.add_argument("--profile", action="append", help="実行するプロファイル名（複数指定可）")
    return parser.parse_args()


def main():
    args = parse_args()
    base_cfg = load_config(args.config)
    profiles = build_profiles(base_cfg)
    if not profiles:
        raise SystemExit("config に有効なプロファイルが定義されていません")

    selected = set(args.profile) if args.profile else None
    if selected:
        available = {name for name, _ in profiles}
        missing = selected - available
        if missing:
            raise SystemExit(f"存在しないプロファイルが指定されました: {', '.join(sorted(missing))}")
        profiles = [(name, cfg) for name, cfg in profiles if name in selected]

    for name, cfg in profiles:
        print(f"=== {name} ===")
        run_for_cfg(cfg)


if __name__ == "__main__":
    main()

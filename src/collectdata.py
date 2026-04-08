"""
STEP 1: Data Collection
========================
يجمع صور من Google و Bing لـ 3 classes:
  - no_resin   : شجرة بدون راتنج
  - not_ready  : راتنج طازج مش ناضج
  - ready      : راتنج جاهز للحصاد
"""

from icrawler.builtin import GoogleImageCrawler, BingImageCrawler
import os

# ===================== CONFIG =====================
DATASET_DIR = "dataset/raw"
MAX_PER_KEYWORD = 150  # per source (Google + Bing) per keyword

KEYWORDS = {
    "no_resin": [
        "Boswellia tree bark no resin",
        "Boswellia sacra trunk bark",
        "frankincense tree without resin",
        "Boswellia neglecta trunk",
    ],
    "not_ready": [
        "Boswellia fresh resin tapping",
        "frankincense tree bleeding resin fresh",
        "Boswellia resin liquid flowing",
        "frankincense tree wound fresh sap",
    ],
    "ready": [
        "frankincense resin harvest ready",
        "Boswellia hardened resin chunks bark",
        "frankincense resin crystals tree",
        "dried frankincense resin on tree",
    ],
}
# ==================================================


def crawl_images(label: str, keyword: str, source: str, max_num: int):
    save_dir = os.path.join(DATASET_DIR, label)
    os.makedirs(save_dir, exist_ok=True)

    print(f"  [{source}] '{keyword}' → {save_dir}")

    if source == "google":
        crawler = GoogleImageCrawler(
            storage={"root_dir": save_dir},
            feeder_threads=2,
            parser_threads=2,
            downloader_threads=4,
        )
    else:
        crawler = BingImageCrawler(
            storage={"root_dir": save_dir},
            downloader_threads=4,
        )

    crawler.crawl(keyword=keyword, max_num=max_num)


def main():
    print("=" * 60)
    print("  STEP 1: Collecting Images")
    print("=" * 60)

    for label, keyword_list in KEYWORDS.items():
        print(f"\n▶ Class: {label}")
        for keyword in keyword_list:
            crawl_images(label, keyword, "google", MAX_PER_KEYWORD)
            crawl_images(label, keyword, "bing", MAX_PER_KEYWORD)

    print("\n✅ Collection done! Check dataset/raw/")
    print("   Now run: python step2_clean_data.py")


if __name__ == "__main__":
    main()
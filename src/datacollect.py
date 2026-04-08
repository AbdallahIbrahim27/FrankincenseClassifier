from icrawler.builtin import GoogleImageCrawler, BingImageCrawler

keywords = {
    "no_resin": [
        "Boswellia tree bark no resin",
        "Boswellia sacra trunk",
        "frankincense tree without resin",
    ],
    "not_ready": [
        "Boswellia fresh resin tapping",
        "frankincense tree bleeding resin",
        "Boswellia resin not mature",
    ],
    "ready": [
        "frankincense resin harvest ready",
        "Boswellia hardened resin chunks",
        "frankincense resin crystals on tree",
    ],
}

for label, keyword_list in keywords.items():
    for i, keyword in enumerate(keyword_list):
        # Google
        g_crawler = GoogleImageCrawler(
            storage={'root_dir': f'dataset/{label}'},
            feeder_threads=2,
            parser_threads=2,
            downloader_threads=4,
        )
        g_crawler.crawl(keyword=keyword, max_num=150)

        # Bing كمان علشان تتنوع الصور
        b_crawler = BingImageCrawler(
            storage={'root_dir': f'dataset/{label}'},
            downloader_threads=4,
        )
        b_crawler.crawl(keyword=keyword, max_num=100)
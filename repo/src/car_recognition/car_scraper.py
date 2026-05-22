import pandas as pd
import requests
import re
import time
import random
import argparse
from pathlib import Path
from urllib.parse import unquote, quote
from concurrent.futures import ThreadPoolExecutor, as_completed
from scrapling.fetchers import StealthyFetcher
from exterior_filterer import ExteriorFilterer

USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
MAX_IMAGES = 100
BETWEEN_DELAY_MIN = 3
BETWEEN_DELAY_MAX = 6
DOWNLOAD_TIMEOUT = 10


class CarImageScraper:
    def __init__(self, output_dir=None):
        self.output_dir = output_dir or Path(__file__).parent.parent.parent / "datasets/car_images_scrapler/"
        self.session = requests.Session()
        self.session.headers.update({'User-Agent': USER_AGENT})
        self.exterior_filterer = ExteriorFilterer()

    def download_car_images(self, csv_path, start=0, end=None):
        self.output_dir.mkdir(parents=True, exist_ok=True)
        df = pd.read_csv(csv_path)

        titles = df['title'].tolist()[start:end]

        for index, title in enumerate(titles):
            print(f"[{index + 1}/{len(titles)}] Processing: {title}")
            try:
                car_folder = self._make_car_folder(title)
                existing = list(car_folder.glob("*.jpg"))
                if len(existing) >= MAX_IMAGES:
                    print(f"  Already have {len(existing)} images, skipping.")
                    continue

                img_urls = self._search_google_images(title)
                if img_urls:
                    count = self._download_images(img_urls, car_folder)
                    print(f"  Downloaded {count} images for: {title}")
                    print(f"  Running exterior filter on: {car_folder}")
                    self.exterior_filterer.filter_images(str(car_folder))
                    self._reindex_images(car_folder)
                else:
                    print(f"  No images found for: {title}")
            except Exception as e:
                print(f"  Error processing {title}: {e}")

            if index < len(titles) - 1:
                delay = random.uniform(BETWEEN_DELAY_MIN, BETWEEN_DELAY_MAX)
                print(f"  Waiting {delay:.1f}s before next search...")
                time.sleep(delay)

    def _search_google_images(self, query):
        search_query = f"{query} car exterior"

        # Step 1: Regular Google search to find the Images tab link
        search_url = f"https://www.google.com/search?q={quote(search_query)}&hl=en"
        print(f"    Step 1: Fetching regular search page...")
        try:
            page = StealthyFetcher.fetch(search_url, headless=True)
        except Exception as e:
            print(f"    StealthyFetcher failed on search page: {e}")
            return []

        # Find the Images tab link from the navigation
        images_url = None
        links = page.css('a')
        for link in links:
            href = link.attrib.get('href', '')
            text = link.text or ''
            if 'Images' in text or 'images' in text:
                if href.startswith('/'):
                    images_url = f"https://www.google.com{href}"
                elif href.startswith('http'):
                    images_url = href
                print(f"    Found Images tab link: {images_url[:80]}...")
                break

        if not images_url:
            # Fallback: use udm=2 which is Google's images mode
            images_url = f"https://www.google.com/search?q={quote(search_query)}&udm=2&hl=en"
            print(f"    Images tab not found in nav, using udm=2 fallback")

        # Step 2: Fetch the Images results page
        print(f"    Step 2: Fetching images page...")
        time.sleep(random.uniform(1, 2))
        try:
            images_page = StealthyFetcher.fetch(images_url, headless=True)
            html = images_page.html_content if hasattr(images_page, 'html_content') else str(images_page)
        except Exception as e:
            print(f"    StealthyFetcher failed on images page: {e}")
            return []

        all_urls = self._extract_image_urls_from_html(html)

        # Deduplicate while preserving order
        seen = set()
        unique_urls = []
        for u in all_urls:
            if u not in seen and len(u) > 20:
                seen.add(u)
                unique_urls.append(u)

        print(f"    Found {len(unique_urls)} unique image URLs from Google")
        return unique_urls

    def _extract_image_urls_from_html(self, html):
        all_urls = []

        # Pattern 1: Full-res image URLs in script tags ["url", width, height]
        matches = re.findall(
            r'\["(https?://[^"]+\.(?:jpg|jpeg|png|webp)(?:\?[^"]*)?)",\s*\d+,\s*\d+\]',
            html
        )
        for m in matches:
            cleaned = m.replace('\\u003d', '=').replace('\\u0026', '&')
            if not self._is_google_domain(cleaned):
                all_urls.append(cleaned)

        # Pattern 2: data-ou attributes (direct image URLs)
        ou_matches = re.findall(r'data-ou="(https?://[^"]+)"', html)
        for m in ou_matches:
            all_urls.append(unquote(m))

        # Pattern 3: imgurl= in href links
        imgurl_matches = re.findall(r'imgurl=(https?://[^&"]+)', html)
        for m in imgurl_matches:
            all_urls.append(unquote(m))

        # Pattern 4: Broad image URL extraction from embedded script data
        script_matches = re.findall(
            r'(https?://[^\s"\\,\]\)]+\.(?:jpg|jpeg|png|webp))',
            html
        )
        for m in script_matches:
            cleaned = m.replace('\\u003d', '=').replace('\\u0026', '&')
            if not self._is_google_domain(cleaned):
                all_urls.append(cleaned)

        return all_urls

    def _is_google_domain(self, url):
        blocked = ['gstatic.com', 'google.com', 'googleapis.com', 'googleusercontent.com', 'ggpht.com']
        return any(domain in url for domain in blocked)

    def _make_car_folder(self, title):
        safe_title = "".join(c for c in title if c.isalnum() or c in (' ', '_', '-')).rstrip()
        car_folder = self.output_dir / safe_title.replace(' ', '_')
        car_folder.mkdir(parents=True, exist_ok=True)
        return car_folder

    def _download_single_image(self, img_url):
        """Download a single image, return bytes or None."""
        try:
            resp = self.session.get(img_url, timeout=DOWNLOAD_TIMEOUT)
            if resp.status_code != 200:
                return None
            content_type = resp.headers.get('Content-Type', '')
            if not content_type.startswith('image/'):
                return None
            if len(resp.content) < 5000:
                return None
            return resp.content
        except Exception as e:
            print(f"    Failed to download {img_url}: {e}")
            return None

    def _download_images(self, img_urls, car_folder, max_workers=20):
        # Filter out logo/icon URLs first
        filtered_urls = [u for u in img_urls if 'logo' not in u.lower() and 'icon' not in u.lower()]
        # Only submit up to MAX_IMAGES + buffer to account for failures
        candidate_urls = filtered_urls[:MAX_IMAGES + 50]

        count = 0
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_url = {
                executor.submit(self._download_single_image, url): url
                for url in candidate_urls
            }
            for future in as_completed(future_to_url):
                if count >= MAX_IMAGES:
                    break
                img_data = future.result()
                if img_data:
                    with open(car_folder / f"{car_folder.name}_{count}.jpg", 'wb') as f:
                        f.write(img_data)
                    count += 1
        return count

    def _reindex_images(self, car_folder):
        images = sorted(car_folder.glob("*.jpg"), key=lambda f: f.stat().st_mtime)
        for new_index, img_path in enumerate(images):
            new_name = car_folder / f"{car_folder.name}_{new_index}.jpg"
            if img_path != new_name:
                img_path.rename(new_name)
        print(f"  Reindexed {len(images)} remaining images in: {car_folder.name}")

    def _extract_car_id(self, title):
        return title.replace(' ', '_')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Scrape car images from Google via Scrapling")
    parser.add_argument("--start", type=int, default=0, help="Start index in CSV (inclusive)")
    parser.add_argument("--end", type=int, default=None, help="End index in CSV (exclusive)")
    args = parser.parse_args()

    csv_path = Path(__file__).parent.parent.parent / "datasets/hatla2ee_cars_unique.csv"
    scraper = CarImageScraper()
    scraper.download_car_images(csv_path, start=args.start, end=args.end)

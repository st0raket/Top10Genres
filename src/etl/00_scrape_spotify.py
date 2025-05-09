#!/usr/bin/env python3
"""
Extract and save Spotify global weekly charts.
"""
import os
import csv
import time
import argparse
import logging
from datetime import datetime, timedelta
from pathlib import Path

from selenium import webdriver
from selenium.webdriver.edge.service import Service
from selenium.webdriver.edge.options import Options
from webdriver_manager.microsoft import EdgeChromiumDriverManager
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.keys import Keys
from bs4 import BeautifulSoup

import sys
from pathlib import Path

root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(root))


import config

def login_spotify(driver: webdriver.Edge, username: str, password: str) -> None:
    """Log into Spotify via Selenium."""
    login_url = (
        "https://accounts.spotify.com/en/login"
    )
    driver.get(login_url)
    wait = WebDriverWait(driver, 20)
    usr = wait.until(EC.presence_of_element_located((By.ID, "login-username")))
    usr.clear(); usr.send_keys(username)
    pwd = driver.find_element(By.ID, "login-password")
    pwd.clear(); pwd.send_keys(password)
    try:
        btn = wait.until(EC.element_to_be_clickable((By.ID, "login-button")))
        btn.click()
    except Exception:
        logging.warning("Login button not clickable, sending ENTER")
        pwd.send_keys(Keys.ENTER)
    time.sleep(3)
    logging.info(f"Logged in, current URL: {driver.current_url}")


def extract(
    start_date: datetime,
    end_date: datetime,
    username: str,
    password: str,
    output_file: Path,
) -> None:
    """Loop weekly from start_date back to end_date and write charts to CSV."""
    # Setup webdriver
    opts = Options()
    opts.add_argument(
        "user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
    )
    service = Service(EdgeChromiumDriverManager().install())
    driver = webdriver.Edge(service=service, options=opts)

    fieldnames = [
        "Position",
        "Song Title",
        "Artists",
        "Last Week",
        "Peak Position",
        "Weeks on Chart",
        "Streams",
        "Date",
    ]

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        current = start_date
        while current >= end_date:
            date_str = current.strftime("%Y-%m-%d")
            url = f"https://charts.spotify.com/charts/view/regional-global-weekly/{date_str}"
            logging.info(f"Scraping {url}")
            driver.get(url)

            wait = WebDriverWait(driver, 20)
            try:
                wait.until(
                    lambda d: len(d.find_elements(By.CSS_SELECTOR, "tr[data-encore-id='tableRow']")) > 0
                )
            except Exception:
                logging.warning(f"No rows for {date_str}")

            soup = BeautifulSoup(driver.page_source, "html.parser")
            rows = soup.select("tr[data-encore-id='tableRow']")
            for idx, row in enumerate(rows, start=1):
                cells = row.select("td[data-encore-id='tableCell']")
                if len(cells) < 7:
                    continue
                title = cells[2].select_one(".styled__StyledTruncatedTitle-sc-135veyd-22")
                artists = cells[2].select(".styled__StyledArtistsTruncatedDiv-sc-135veyd-28 a")

                writer.writerow({
                    "Position": idx,
                    "Song Title": title.get_text(strip=True) if title else "",
                    "Artists": ", ".join(a.get_text(strip=True) for a in artists),
                    "Last Week": cells[3].get_text(strip=True),
                    "Peak Position": cells[4].get_text(strip=True),
                    "Weeks on Chart": cells[5].get_text(strip=True),
                    "Streams": int(cells[6].get_text(strip=True).replace(",", "") or 0),
                    "Date": date_str,
                })

            current -= timedelta(weeks=1)

    driver.quit()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Spotify charts pipeline")
    parser.add_argument(
        "--start", type=lambda s: datetime.strptime(s, "%Y-%m-%d"),
        default=datetime.now(), help="Start date YYYY-MM-DD"
    )
    parser.add_argument(
        "--end", type=lambda s: datetime.strptime(s, "%Y-%m-%d"),
        default=datetime(2016, 12, 29), help="End date YYYY-MM-DD"
    )
    parser.add_argument(
        "--user", default=os.getenv("SPOTIFY_USER"), help="Spotify username"
    )
    parser.add_argument(
        "--pass", dest="password", default=os.getenv("SPOTIFY_PASS"), help="Spotify password"
    )
    parser.add_argument(
        "--out", type=Path,
        default=config.PROCESSED_DIR / "spotify_charts.csv",
        help="Output CSV path"
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    args = parse_args()
    extract(args.start, args.end, args.user, args.password, args.out)


if __name__ == "__main__":
    main()

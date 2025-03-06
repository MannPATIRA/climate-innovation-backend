from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import Select, WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from typing import Generator, Dict, Any
import os
import time
import csv
from .fetchers import ReportFetcher
from .utils.selenium_utils import wait_for_page_load
from .utils.download_handlers import (
    try_generic_pdf_download,
    extract_sciencedirect_preview
)

class WebReportFetcher(ReportFetcher):
    def __init__(self, base_url: str, download_dir: str = None):
        self.base_url = base_url
        self.download_dir = download_dir or os.path.join(os.getcwd(), "downloads")
        self.driver = None
        if not os.path.exists(self.download_dir):
            os.makedirs(self.download_dir)

    def __enter__(self):
        self.driver = self._setup_driver()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.driver:
            self.driver.quit()

    def fetch(self) -> Generator[str, None, None]:
        """
        Yields downloaded report paths one at a time.
        """
        try:
            print("Starting the evidence collection process...")
            self.driver.get(self.base_url)
            goals = self._get_goals()
            
            for index, goal in enumerate(goals, 1):
                print(f"\nProcessing goal {index} of {len(goals)}")
                print(f"Goal: {goal['name']}")
                for report_path in self._process_goal(goal['url'], goal['name']):
                    yield report_path

        except Exception as e:
            print(f"Error in fetch: {e}")
            self.driver.save_screenshot("error.png")

    def _setup_driver(self):
        """Sets up Chrome browser with defaults"""
        options = webdriver.ChromeOptions()
        # Add headless mode options
        options.add_argument('--headless')  # Run in headless mode
        options.add_argument('--disable-gpu')  # Disable GPU hardware acceleration
        options.add_argument('--no-sandbox')  # Bypass OS security model
        options.add_argument('--disable-dev-shm-usage')  # Overcome limited resource problems
        options.add_argument('--window-size=1920,1080')  # Set a standard window size

        

        options.add_experimental_option("prefs", {
            "download.default_directory": self.download_dir,
            "download.prompt_for_download": False,
            "download.directory_upgrade": True,
            "safebrowsing.enabled": True,
            "plugins.always_open_pdf_externally": True  # Auto-download PDFs instead of opening them
        })
        return webdriver.Chrome(options=options)

    def _get_goals(self) -> list:
        tab_menu = WebDriverWait(self.driver, 10).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, "div.tab-menu"))
        )
        goal_elements = tab_menu.find_elements(By.CSS_SELECTOR, "li.tab-item a")
        return [{
            'url': element.get_attribute('href'),
            'name': element.find_element(By.CSS_SELECTOR, "span").text
        } for element in goal_elements]

    def _get_outcomes(self) -> list:
        try:
            dropdown = WebDriverWait(self.driver, 10).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "form.highlight-nav select.menu-item"))
            )
            select = Select(dropdown)
            outcomes = [option.text for option in select.options if option.text != "Select an Outcome"]
            print(f"Found {len(outcomes)} outcomes to process")
            return outcomes
        except Exception as e:
            print(f"Couldn't find the outcomes dropdown: {e}")
            return []

    def _get_evidence_links(self) -> list:
        try:
            WebDriverWait(self.driver, 10).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "div.data_grid .row"))
            )
            evidence_items = self.driver.find_elements(By.CSS_SELECTOR, "div.data_grid .row.displayed")
            
            links = []
            for item in evidence_items:
                link_element = item.find_element(By.CSS_SELECTOR, "span.source a")
                citation_element = item.find_element(By.CSS_SELECTOR, "span.citation")
                links.append({
                    'url': link_element.get_attribute('href'),
                    'text': link_element.text,
                    'citation': citation_element.text
                })
            return links
        except Exception as e:
            print(f"Had trouble getting the evidence links: {e}")
            return []

    def _process_goal(self, goal_url: str, goal_name: str) -> Generator[str, None, None]:
        try:
            print(f"\n=== Starting to process goal: {goal_name} ===")
            self.driver.get(goal_url)
            
            WebDriverWait(self.driver, 10).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "div.panel-content"))
            )

            if 'panel/evidence' not in self.driver.current_url:
                evidence_tab = WebDriverWait(self.driver, 10).until(
                    EC.element_to_be_clickable((By.CSS_SELECTOR, "a.menu-link[href*='panel/evidence']"))
                )
                evidence_tab.click()
                time.sleep(2)

            outcomes = self._get_outcomes()
            
            for outcome in outcomes:
                print(f"\n-> Looking at outcome: {outcome}")
                
                dropdown = WebDriverWait(self.driver, 10).until(
                    EC.presence_of_element_located((By.CSS_SELECTOR, "form.highlight-nav select.menu-item"))
                )
                select = Select(dropdown)
                select.select_by_visible_text(outcome)
                time.sleep(2)
                
                evidence_links = self._get_evidence_links()
                
                for link in evidence_links:
                    report_path = self._download_report(link['url'], goal_name, outcome, link['text'])
                    if report_path:
                        yield report_path

        except Exception as e:
            print(f"Something went wrong while processing {goal_name}: {e}")
            self.driver.save_screenshot(f"reportErrors/error_{goal_name.replace(' ', '_')}.png")

    def _download_report(self, url: str, goal_name: str, outcome: str, evidence_text: str) -> str:
        try:
            print(f"Attempting to download report from: {url}")
            self.driver.get(url)
            wait_for_page_load(self.driver)
            downloaded_path = None
            success = None
            
            # Site-specific handling
            if "sciencedirect.com" in url:
                success = extract_sciencedirect_preview(self.driver, url, self.download_dir)
                if success:
                    downloaded_path = os.path.join(self.download_dir, f"preview_{int(time.time())}.txt")
                else:
                  print("Science direct download failed")
            else:
                success = try_generic_pdf_download(self.driver, url)
                if success:
                    # Wait for download and get the path
                    time.sleep(5)  # Adjust based on typical download times
                    # You'll need to implement a way to get the actual downloaded file path
                    downloaded_path = self._get_latest_download()
                else:
                  print(f"No PDF link found on page: {url}")

            # Record the download attempt
            self._save_download_record({
                'goal': goal_name,
                'outcome': outcome,
                'evidence_text': evidence_text,
                'source_url': url,
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'status': 'completed' if success else 'failed'
            })

            return downloaded_path

        except Exception as e:
            print(f"Error while trying to download report: {e}")
            self.driver.save_screenshot(f"download_error_{time.strftime('%Y%m%d_%H%M%S')}.png")
            return None

    def _save_download_record(self, record: Dict[str, Any]):
        """Save the results to a CSV file"""
        filename = os.path.join(self.download_dir, 'download_records.csv')
        file_exists = os.path.exists(filename)
        
        with open(filename, 'a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=['goal', 'outcome', 'evidence_text', 'source_url', 'timestamp', 'status'])
            if not file_exists:
                writer.writeheader()
            writer.writerow(record)

    def _get_latest_download(self) -> str:
        # Implementation to get the most recent downloaded file
        files = [os.path.join(self.download_dir, f) for f in os.listdir(self.download_dir)]
        if files:
            return max(files, key=os.path.getctime)
        return None
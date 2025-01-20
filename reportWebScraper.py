from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import Select
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time
import os
import csv

def setup_driver():
    """Sets up our Chrome browser with some nice defaults"""
    # Let's configure Chrome to automatically download files to a downloads folder
    options = webdriver.ChromeOptions()
    options.add_experimental_option("prefs", {
        "download.default_directory": os.path.join(os.getcwd(), "downloads"),
        "download.prompt_for_download": False,  # Don't ask where to save files
    })
    driver = webdriver.Chrome(options=options)
    return driver

def get_outcomes(driver):
    """Finds all the possible outcomes in the dropdown menu
    
    This is a bit tricky because we need to:
    1. Wait for the dropdown to actually appear
    2. Skip the first option which just says "Select an Outcome"
    3. Handle any errors gracefully
    """
    try:
        # First, let's wait for the dropdown to show up - giving it up to 10 seconds
        print("Looking for the outcomes dropdown...")
        dropdown = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, "form.highlight-nav select.menu-item"))
        )
        select = Select(dropdown)
        
        # Get all options except the first one (which is just the placeholder)
        outcomes = [option.text for option in select.options if option.text != "Select an Outcome"]
        print(f"Found {len(outcomes)} outcomes to process")
        return outcomes
    except Exception as e:
        print(f"Uh oh! Couldn't find the outcomes dropdown: {e}")
        return []

def get_evidence_links(driver):
    """Grabs all the evidence links that are currently shown on the page
    
    We need to be careful here because:
    1. Links only show up after selecting an outcome
    2. We only want the 'displayed' items
    3. Each evidence item has a specific structure we need to parse
    """
    try:
        print("Waiting for evidence items to load...")
        # Give the page time to show the evidence
        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, "div.data_grid .row"))
        )
        
        # Only grab the evidence items that are actually showing
        evidence_items = driver.find_elements(By.CSS_SELECTOR, "div.data_grid .row.displayed")
        
        links = []
        for item in evidence_items:
            # Each evidence item should have a source link
            link_element = item.find_element(By.CSS_SELECTOR, "span.source a")
            citation_element = item.find_element(By.CSS_SELECTOR, "span.citation")
            links.append({
                'url': link_element.get_attribute('href'),
                'text': link_element.text,
                'citation': citation_element.text
            })
        
        print(f"Found {len(links)} pieces of evidence")
        return links
    except Exception as e:
        print(f"Had trouble getting the evidence links: {e}")
        return []

def process_goal(driver, goal_url, goal_name):
    """Works through a single goal, getting all its outcomes and evidence
    
    This is the main workhorse function that:
    1. Goes to the goal's page
    2. Makes sure we're on the evidence tab
    3. Goes through each outcome
    4. Collects all the evidence
    """
    results = []
    try:
        print(f"\n=== Starting to process goal: {goal_name} ===")
        
        # Navigate to the goal URL with explicit wait
        driver.get(goal_url)
        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, "div.panel-content"))
        )
        
        # Make sure we're on the Evidence tab
        if 'panel/evidence' not in driver.current_url:
            print("Looking for the Evidence tab...")
            evidence_tab = WebDriverWait(driver, 10).until(
                EC.element_to_be_clickable((By.CSS_SELECTOR, "a.menu-link[href*='panel/evidence']"))
            )
            evidence_tab.click()
            time.sleep(2)
        
        # Rest of the processing...
        outcomes = get_outcomes(driver)
        
        for outcome in outcomes:
            print(f"\n-> Looking at outcome: {outcome}")
            
            dropdown = WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "form.highlight-nav select.menu-item"))
            )
            select = Select(dropdown)
            print(f"Selecting {outcome} from dropdown...")
            select.select_by_visible_text(outcome)
            time.sleep(2)
            
            evidence_links = get_evidence_links(driver)
            
            # Save the results
            for link in evidence_links:
                results.append({
                    'goal': goal_name,
                    'outcome': outcome,
                    'evidence_url': link['url'],
                    'evidence_text': link['text']
                })
                print(f"Found evidence: {link['text']}")
                # TODO: Here's where we'd actually download or save the evidence
                download_report(driver, link['url'], goal_name, outcome, link['text'])
    
    # save_to_csv(results)            
    except Exception as e:
        print(f"Something went wrong while processing {goal_name}: {e}")
        driver.save_screenshot(f"reportErrors/error_{goal_name.replace(' ', '_')}.png")


def save_to_csv(results):
    """Save the results to a CSV file"""
    import csv
    import os
    
    filename = 'evidence_results.csv'
    file_exists = os.path.isfile(filename)
    
    with open(filename, 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['goal', 'outcome', 'evidence_url', 'evidence_text'])
        
        if not file_exists:
            writer.writeheader()
            
        writer.writerows(results)

def wait_for_page_load(driver, timeout=10):
    """More comprehensive page load check"""
    try:
        old_page = driver.find_element(By.TAG_NAME, 'html')
        
        # Wait for staleness of old page
        WebDriverWait(driver, timeout).until(
            EC.staleness_of(old_page)
        )
        
        # Wait for presence of body
        WebDriverWait(driver, timeout).until(
            EC.presence_of_element_located((By.TAG_NAME, "body"))
        )
        
        # Wait for any animations to complete
        WebDriverWait(driver, timeout).until(
            lambda d: d.execute_script("""
                return (typeof jQuery != 'undefined') ? 
                    jQuery.active == 0 : true
            """)
        )
        
        # Wait for any AJAX requests to complete
        WebDriverWait(driver, timeout).until(
            lambda d: d.execute_script("""
                return (typeof jQuery != 'undefined') ? 
                    jQuery.active == 0 : true
            """)
        )
        
        # Wait for any Angular requests to complete (if Angular is present)
        WebDriverWait(driver, timeout).until(
            lambda d: d.execute_script("""
                if (window.angular) {
                    var injector = window.angular.element('body').injector();
                    var $http = injector.get('$http');
                    return $http.pendingRequests.length == 0;
                }
                return true;
            """)
        )
        
    except Exception as e:
        print(f"Page load wait timed out: {e}")

def try_generic_pdf_download(driver, url):
    """
    Attempts to find and click PDF download links using various methods
    Returns True if successful, False otherwise
    """
    print(f"Trying generic PDF download approach for: {url}")
    
    # Define selector patterns
    pdf_selectors = [
        "a[href$='.pdf']",  # Links ending in .pdf
        "a[href*='pdf']",   # Links containing 'pdf'
        "a[download]",      # Links with download attribute
        "a:contains('PDF')", # Links containing text 'PDF'
        "a:contains('Download')", # Links containing text 'Download'
        ".pdf-download",    # Common class names
        ".download-pdf",
        "#download-button",
        "button:contains('Download')",
        "button:contains('PDF')"
    ]
    
    xpath_patterns = [
        "//a[contains(translate(text(), 'PDF', 'pdf'), 'pdf')]",
        "//a[contains(translate(text(), 'DOWNLOAD', 'download'), 'download')]",
        "//button[contains(translate(text(), 'PDF', 'pdf'), 'pdf')]",
        "//button[contains(translate(text(), 'DOWNLOAD', 'download'), 'download')]"
    ]
    
    try:
        # Try CSS selectors first
        for selector in pdf_selectors:
            print(f"Trying selector: {selector}")
            try:
                elements = driver.find_elements(By.CSS_SELECTOR, selector)
                if elements:
                    print(f"Found {len(elements)} elements with selector {selector}")
                    for elem in elements:
                        print(f"Element text: {elem.text}")
                        print(f"Element href: {elem.get_attribute('href')}")
                        try:
                            elem.click()
                            print("Click successful")
                            return True
                        except Exception as click_error:
                            print(f"Click attempt failed: {click_error}")
            except Exception as selector_error:
                print(f"Selector {selector} failed: {selector_error}")
        
        # Try XPath patterns
        for xpath in xpath_patterns:
            print(f"Trying XPath: {xpath}")
            try:
                elements = driver.find_elements(By.XPATH, xpath)
                if elements:
                    print(f"Found {len(elements)} elements with XPath {xpath}")
                    for elem in elements:
                        print(f"Element text: {elem.text}")
                        print(f"Element href: {elem.get_attribute('href')}")
                        try:
                            elem.click()
                            print("Click successful")
                            return True
                        except Exception as click_error:
                            print(f"Click attempt failed: {click_error}")
            except Exception as xpath_error:
                print(f"XPath {xpath} failed: {xpath_error}")
        
        # Try JavaScript approach as last resort
        print("Trying JavaScript click approach")
        try:
            success = driver.execute_script("""
                var links = document.getElementsByTagName('a');
                for(var i = 0; i < links.length; i++) {
                    if(links[i].href.includes('pdf') || 
                       links[i].textContent.toLowerCase().includes('pdf') ||
                       links[i].textContent.toLowerCase().includes('download')) {
                        links[i].click();
                        return true;
                    }
                }
                return false;
            """)
            if success:
                print("JavaScript click successful")
                return True
        except Exception as js_error:
            print(f"JavaScript click attempt failed: {js_error}")
        
        print("No PDF download link found")
        return False
        
    except Exception as e:
        print(f"All generic PDF download approaches failed: {e}")
        driver.save_screenshot(f"pdf_error_{time.strftime('%Y%m%d_%H%M%S')}.png")
        return False


def download_report(driver, url, goal_name, outcome, evidence_text):
    """Attempts to find and download the report from various academic websites"""
    try:
        print(f"Attempting to download report from: {url}")
        driver.get(url)
        # time.sleep(5)  # Give the page time to load
        wait_for_page_load(driver)
        status = 'attempted'
        
        # Different patterns for different sites
        if "sciencedirect.com" in url:
            success = extract_sciencedirect_preview(driver, url)
            status = 'completed' if success else 'failed'
                
        elif "nature.com" in url:
            try:
                # Try to find PDF download button on Nature
                pdf_button = WebDriverWait(driver, 10).until(
                    EC.element_to_be_clickable((By.CSS_SELECTOR, "a.c-pdf-download__link, a[data-track-action='download pdf']"))
                )
                pdf_button.click()
            except Exception as e:
                print(f"Couldn't find PDF button on Nature: {e}")
                
        elif "onlinelibrary.wiley.com" in url:
            try:
                # Try to find PDF download button on Wiley
                pdf_button = WebDriverWait(driver, 10).until(
                    EC.element_to_be_clickable((By.CSS_SELECTOR, "a.pdf-download, a[title*='PDF']"))
                )
                pdf_button.click()
            except Exception as e:
                print(f"Couldn't find PDF button on Wiley: {e}")
                
        elif "academic.oup.com" in url:
            try:
                # Try to find PDF download button on Oxford Academic
                pdf_button = WebDriverWait(driver, 10).until(
                    EC.element_to_be_clickable((By.CSS_SELECTOR, "a.article-pdfLink, a[data-article-type='pdf']"))
                )
                pdf_button.click()
            except Exception as e:
                print(f"Couldn't find PDF button on Oxford Academic: {e}")
                
        elif "springer.com" in url:
            try:
                # Try to find PDF download button on Springer
                pdf_button = WebDriverWait(driver, 10).until(
                    EC.element_to_be_clickable((By.CSS_SELECTOR, "a.c-pdf-download__link, a[data-track-action='download pdf']"))
                )
                pdf_button.click()
            except Exception as e:
                print(f"Couldn't find PDF button on Springer: {e}")
        
        else:
            # Generic approach for other sites
            success = try_generic_pdf_download(driver, url)
            if not success:
                print(f"No PDF link found on page: {url}")

        
        # Wait for download to complete
        time.sleep(5)  # Adjust this based on typical download times
        
        # Create a record of the download attempt
        download_record = {
            'goal': goal_name,
            'outcome': outcome,
            'evidence_text': evidence_text,
            'source_url': url,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'status': status
        }
        
        # Save download record to CSV
        save_download_record(download_record)
        
    except Exception as e:
        print(f"Error while trying to download report: {e}")
        driver.save_screenshot(f"download_error_{time.strftime('%Y%m%d_%H%M%S')}.png")

def save_download_record(record):
    """Save download attempt record to CSV"""
    filename = 'download_records.csv'
    file_exists = os.path.isfile(filename)
    
    with open(filename, 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['goal', 'outcome', 'evidence_text', 'source_url', 'timestamp', 'status'])
        if not file_exists:
            writer.writeheader()
        writer.writerow(record)

def extract_sciencedirect_preview(driver, url):
    """Extract preview text content from ScienceDirect articles"""
    try:
        print(f"Extracting preview content from: {url}")
        
        # Create a dictionary to store the sections
        preview_content = {
            'title': '',
            'abstract': '',
            'introduction': '',
            'section_snippets': [],
            'url': url
        }
        
        # Find the title
        try:
            title_element = driver.find_element(By.CLASS_NAME, "title-text")
            preview_content['title'] = title_element.text.strip()
        except Exception as e:
            print(f"Could not find title: {e}")
        
        # Find the abstract
        try:
            abstract_section = driver.find_element(By.ID, "preview-section-abstract")
            abstract_paragraphs = abstract_section.find_elements(By.CSS_SELECTOR, "div.u-margin-s-bottom")
            preview_content['abstract'] = "\n".join([p.text.strip() for p in abstract_paragraphs])
        except Exception as e:
            print(f"Could not find abstract: {e}")
            
        # Find the introduction
        try:
            intro_section = driver.find_element(By.ID, "preview-section-introduction")
            intro_paragraphs = intro_section.find_elements(By.CSS_SELECTOR, "div.u-margin-s-bottom")
            preview_content['introduction'] = "\n".join([p.text.strip() for p in intro_paragraphs])
        except Exception as e:
            print(f"Could not find introduction: {e}")
            
        # Find section snippets
        try:
            snippets_section = driver.find_element(By.ID, "preview-section-snippets")
            snippet_titles = snippets_section.find_elements(By.CSS_SELECTOR, "h2.section-title")
            snippet_contents = snippets_section.find_elements(By.CSS_SELECTOR, "div.u-margin-s-bottom")
            
            for title, content in zip(snippet_titles, snippet_contents):
                preview_content['section_snippets'].append({
                    'title': title.text.strip(),
                    'content': content.text.strip()
                })
        except Exception as e:
            print(f"Could not find section snippets: {e}")
            
        # Save to file
        filename = f"preview_{int(time.time())}.txt"
        with open(os.path.join("downloads", filename), 'w', encoding='utf-8') as f:
            f.write(f"URL: {url}\n\n")
            f.write(f"TITLE: {preview_content['title']}\n\n")
            f.write(f"ABSTRACT:\n{preview_content['abstract']}\n\n")
            f.write(f"INTRODUCTION:\n{preview_content['introduction']}\n\n")
            f.write("SECTION SNIPPETS:\n")
            for snippet in preview_content['section_snippets']:
                f.write(f"\n{snippet['title']}:\n{snippet['content']}\n")
                
        print(f"Preview content saved to: {filename}")
        return True
        
    except Exception as e:
        print(f"Error extracting preview content: {e}")
        return False

def main():
    """This is where everything starts!
    
    We:
    1. Set up our browser
    2. Go to the main dashboard
    3. Find all the goals
    4. Process each goal one by one
    5. Clean up when we're done
    """
    print("Starting the evidence collection process...")
    driver = setup_driver()
    
    try:
        # Go to the main dashboard
        print("Going to the main dashboard...")
        driver.get("https://iris.thegiin.org/share/id/47226x678e3dca05e43/")
        
        # Wait for the tab menu to be present and find all goal links
        tab_menu = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, "div.tab-menu"))
        )
        
        # Find all goal links within the tab menu
        goal_elements = tab_menu.find_elements(By.CSS_SELECTOR, "li.tab-item a")
        
        # Create a list of goals with their URLs and names
        goals = []
        for goal_element in goal_elements:
            goals.append({
                'url': goal_element.get_attribute('href'),
                'name': goal_element.find_element(By.CSS_SELECTOR, "span").text
            })
        
        print(f"Found {len(goals)} goals to process")
        
        # Process each goal one at a time
        for index, goal in enumerate(goals, 1):
            print(f"\nProcessing goal {index} of {len(goals)}")
            print(f"Goal: {goal['name']}")
            process_goal(driver, goal['url'], goal['name'])
            
    except Exception as e:
        print(f"Oops! Something went wrong in the main process: {e}")
        # Optionally capture a screenshot of the error
        driver.save_screenshot("error.png")
    finally:
        print("\nAll done! Closing the browser...")
        driver.quit()

if __name__ == "__main__":
    main()
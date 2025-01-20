from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import Select
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time
import os

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
            links.append({
                'url': link_element.get_attribute('href'),
                'text': link_element.text
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
    
    # save_to_csv(results)            
    except Exception as e:
        print(f"Something went wrong while processing {goal_name}: {e}")
        driver.save_screenshot(f"error_{goal_name.replace(' ', '_')}.png")


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
import os
import time
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

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
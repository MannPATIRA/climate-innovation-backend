from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC

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
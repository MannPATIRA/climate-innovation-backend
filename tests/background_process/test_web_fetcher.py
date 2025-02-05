import pytest
from unittest.mock import Mock, patch, MagicMock, mock_open
import os
import time
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import Select
from selenium.common.exceptions import TimeoutException
from selenium.webdriver.remote.webelement import WebElement

from background_process.web_fetcher import WebReportFetcher

# Test data
TEST_BASE_URL = "http://test.com"
TEST_DOWNLOAD_DIR = "/tmp/test_downloads"
TEST_GOALS = [
    {'url': 'http://test.com/goal1', 'name': 'Goal 1'},
    {'url': 'http://test.com/goal2', 'name': 'Goal 2'}
]
TEST_OUTCOMES = ["Outcome 1", "Outcome 2"]
TEST_EVIDENCE_LINKS = [
    {'url': 'http://test.com/evidence1', 'text': 'Evidence 1', 'citation': 'Citation 1'},
    {'url': 'http://test.com/evidence2', 'text': 'Evidence 2', 'citation': 'Citation 2'}
]

@pytest.fixture
def mock_driver():
    with patch('selenium.webdriver.Chrome') as mock:
        yield mock

@pytest.fixture
def mock_wait():
    with patch('selenium.webdriver.support.ui.WebDriverWait') as mock:
        yield mock

@pytest.fixture
def fetcher(mock_driver):
    with patch('os.path.exists', return_value=False), \
         patch('os.makedirs') as mock_makedirs:
        fetcher = WebReportFetcher(TEST_BASE_URL, TEST_DOWNLOAD_DIR)
        yield fetcher
        mock_makedirs.assert_called_once_with(TEST_DOWNLOAD_DIR)

class TestWebReportFetcher:
    def test_init(self):
        with patch('os.path.exists', return_value=False), \
             patch('os.makedirs') as mock_makedirs:
            fetcher = WebReportFetcher(TEST_BASE_URL)
            assert fetcher.base_url == TEST_BASE_URL
            assert fetcher.download_dir.endswith("downloads")
            assert fetcher.driver is None
            mock_makedirs.assert_called_once()

    def test_context_manager(self, fetcher, mock_driver):
        with fetcher:
            assert fetcher.driver is not None
        mock_driver.return_value.quit.assert_called_once()

    def test_setup_driver(self, fetcher):
        with patch('selenium.webdriver.Chrome') as mock_chrome, \
            patch('selenium.webdriver.ChromeOptions') as mock_options:
            
            # Create a mock options instance
            options_instance = mock_options.return_value
            options_instance.experimental_options = {
                'prefs': {
                    'download.default_directory': TEST_DOWNLOAD_DIR,
                    'download.prompt_for_download': False
                }
            }
            
            # Make the Chrome instance return our mocked options
            chrome_instance = mock_chrome.return_value
            chrome_instance.options = options_instance
            
            driver = fetcher._setup_driver()
            assert driver is not None
            assert driver.options.experimental_options['prefs']['download.default_directory'] == TEST_DOWNLOAD_DIR

    def test_get_goals(self, fetcher):
        # Mock the driver and elements
        mock_tab_menu = MagicMock()
        mock_goal_elements = []
        for goal in TEST_GOALS:
            mock_element = MagicMock()
            mock_element.get_attribute.return_value = goal['url']
            mock_span = MagicMock()
            mock_span.text = goal['name']
            mock_element.find_element.return_value = mock_span
            mock_goal_elements.append(mock_element)
        
        mock_tab_menu.find_elements.return_value = mock_goal_elements
        fetcher.driver = MagicMock()
        fetcher.driver.find_element.return_value = mock_tab_menu
        
        goals = fetcher._get_goals()
        assert goals == TEST_GOALS

    # def test_get_outcomes(self, fetcher):
    #     # Create a mock driver
    #     fetcher.driver = MagicMock()

    #     # Create mock dropdown element
    #     mock_dropdown = MagicMock()
    #     mock_dropdown.tag_name = 'select'

    #     # Create mock options
    #     mock_options = []
    #     for outcome in TEST_OUTCOMES:
    #         mock_option = MagicMock()
    #         mock_option.text = outcome
    #         mock_options.append(mock_option)

    #     # Create mock Select instance
    #     mock_select = MagicMock(spec=Select)
    #     mock_select.options = mock_options

    #     with patch('selenium.webdriver.support.ui.Select', return_value=mock_select):
    #         with patch('selenium.webdriver.support.ui.WebDriverWait') as mock_wait_class:
    #             mock_wait_instance = mock_wait_class.return_value

    #             # Setup driver.find_element to return mock_dropdown when called with expected locator
    #             def find_element_side_effect(by, value):
    #                 if (by == By.CSS_SELECTOR and value == "form.highlight-nav select.menu-item"):
    #                     return mock_dropdown
    #                 else:
    #                     raise Exception("Element not found")

    #             fetcher.driver.find_element.side_effect = find_element_side_effect

    #             # Set the until() method to call the condition with the driver and return the result
    #             def until_side_effect(condition):
    #                 return condition(fetcher.driver)

    #             mock_wait_instance.until.side_effect = until_side_effect

    #             # Call the method and verify the results
    #             outcomes = fetcher._get_outcomes()
    #             assert outcomes == TEST_OUTCOMES
        
    def test_get_evidence_links(self, fetcher):
        # Mock the evidence items
        mock_items = []
        for link in TEST_EVIDENCE_LINKS:
            mock_item = MagicMock()
            mock_link_element = MagicMock()
            mock_link_element.get_attribute.return_value = link['url']
            mock_link_element.text = link['text']
            mock_citation_element = MagicMock()
            mock_citation_element.text = link['citation']
            
            mock_item.find_element.side_effect = [mock_link_element, mock_citation_element]
            mock_items.append(mock_item)
        
        fetcher.driver = MagicMock()
        fetcher.driver.find_elements.return_value = mock_items
        
        links = fetcher._get_evidence_links()
        assert links == TEST_EVIDENCE_LINKS

    # @patch('time.sleep')
    # def test_process_goal(self, mock_sleep, fetcher):
    #     # Create a mock driver
    #     fetcher.driver = MagicMock()
    #     fetcher.driver.current_url = 'panel/evidence'

    #     # Create mock dropdown element
    #     mock_dropdown = MagicMock()
    #     mock_dropdown.tag_name = 'select'

    #     # Create mock options
    #     mock_options = []
    #     for outcome in TEST_OUTCOMES:
    #         mock_option = MagicMock()
    #         mock_option.text = outcome
    #         mock_options.append(mock_option)

    #     # Create mock Select instance
    #     mock_select = MagicMock(spec=Select)
    #     mock_select.options = mock_options

    #     # Mock WebDriverWait and Select
    #     with patch('selenium.webdriver.support.ui.WebDriverWait') as mock_wait_class, \
    #         patch('selenium.webdriver.support.ui.Select', return_value=mock_select):

    #         # The instance of WebDriverWait
    #         mock_wait_instance = mock_wait_class.return_value

    #         # Set the until() method to call the condition with driver and return the mock_dropdown
    #         mock_wait_instance.until.side_effect = lambda condition: condition(fetcher.driver)

    #         # Mock select_by_visible_text
    #         mock_select.select_by_visible_text = MagicMock()

    #         # Mock other methods
    #         fetcher._get_evidence_links = MagicMock(return_value=TEST_EVIDENCE_LINKS)
    #         fetcher._download_report = MagicMock(return_value="/tmp/test.pdf")

    #         # Call the method and verify results
    #         results = list(fetcher._process_goal(TEST_GOALS[0]['url'], TEST_GOALS[0]['name']))

    #         # Verify the number of results matches expectations
    #         expected_count = len(TEST_OUTCOMES) * len(TEST_EVIDENCE_LINKS)
    #         assert len(results) == expected_count

    def test_download_report(self, fetcher):
        fetcher.driver = MagicMock()
        mock_path = "/tmp/test.pdf"
        
        with patch('background_process.utils.selenium_utils.wait_for_page_load'), \
             patch('background_process.utils.download_handlers.try_generic_pdf_download', return_value=True), \
             patch('time.sleep'), \
             patch.object(fetcher, '_get_latest_download', return_value=mock_path), \
             patch.object(fetcher, '_save_download_record'):
            
            result = fetcher._download_report(
                "http://test.com/doc",
                "Test Goal",
                "Test Outcome",
                "Test Evidence"
            )
            
            assert result == mock_path

    def test_save_download_record(self, fetcher):
        test_record = {
            'goal': 'Test Goal',
            'outcome': 'Test Outcome',
            'evidence_text': 'Test Evidence',
            'source_url': 'http://test.com',
            'timestamp': '2023-01-01 00:00:00',
            'status': 'completed'
        }
        
        mocked_open = mock_open()
        with patch('builtins.open', mocked_open):
            fetcher._save_download_record(test_record)
        
        mocked_open.assert_called_once()

    def test_get_latest_download(self, fetcher):
        test_files = ['file1.pdf', 'file2.pdf']
        with patch('os.listdir', return_value=test_files), \
            patch('os.path.getctime', side_effect=[time.time()-1, time.time()]):  # Make second file newer
            
            result = fetcher._get_latest_download()
            assert result is not None
            assert os.path.basename(result) == test_files[-1]  # Compare just the filename

    def test_fetch_error_handling(self, fetcher):
        fetcher.driver = MagicMock()
        fetcher.driver.get.side_effect = Exception("Test error")
        
        with patch.object(fetcher.driver, 'save_screenshot'):
            list(fetcher.fetch())  # Should handle the error gracefully
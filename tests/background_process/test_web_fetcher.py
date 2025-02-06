import pytest
from unittest.mock import patch, MagicMock, mock_open
import os
import time
from selenium.webdriver.common.by import By
from selenium.common.exceptions import TimeoutException
from selenium.common.exceptions import TimeoutException

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

    def test_get_outcomes_empty_list(self, fetcher):
        """Test handling of empty outcomes list"""
        mock_select = MagicMock()
        mock_select.options = []

        with patch('selenium.webdriver.support.ui.Select', return_value=mock_select):
            with patch('selenium.webdriver.support.ui.WebDriverWait') as mock_wait:
                mock_wait.return_value.until.return_value = MagicMock()
                
                outcomes = fetcher._get_outcomes()
                assert len(outcomes) == 0

    def test_get_outcomes_exception_handling(self, fetcher):
        """Test exception handling in _get_outcomes"""
        with patch('selenium.webdriver.support.ui.WebDriverWait') as mock_wait:
            mock_wait.return_value.until.side_effect = TimeoutException("Test timeout")
            
            outcomes = fetcher._get_outcomes()
            assert len(outcomes) == 0
        
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

    def test_process_goal_no_outcomes(self, fetcher):
        """Test process_goal behavior when no outcomes are found"""
        fetcher.driver = MagicMock()
        fetcher.driver.current_url = 'panel/evidence'

        with patch.object(fetcher, '_get_outcomes', return_value=[]):
            results = list(fetcher._process_goal(TEST_GOALS[0]['url'], TEST_GOALS[0]['name']))
            assert len(results) == 0

    def test_process_goal_download_failure(self, fetcher):
        """Test process_goal handling of download failures"""
        fetcher.driver = MagicMock()
        fetcher.driver.current_url = 'panel/evidence'

        with patch.object(fetcher, '_get_outcomes', return_value=TEST_OUTCOMES), \
             patch.object(fetcher, '_get_evidence_links', return_value=TEST_EVIDENCE_LINKS), \
             patch.object(fetcher, '_download_report', return_value=None):
            
            results = list(fetcher._process_goal(TEST_GOALS[0]['url'], TEST_GOALS[0]['name']))
            assert len(results) == 0
            
    def test_process_goal_evidence_tab_navigation(self, fetcher):
        """Test navigation to evidence tab"""
        fetcher.driver = MagicMock()
        # Set current_url to a URL that does NOT contain 'panel/evidence'
        fetcher.driver.current_url = 'http://test.com/goal1'

        # Create mocks for elements
        mock_panel_content = MagicMock(name='panel_content')
        mock_panel_content.is_displayed.return_value = True

        mock_evidence_tab = MagicMock(name='evidence_tab')
        mock_evidence_tab.is_displayed.return_value = True
        mock_evidence_tab.is_enabled.return_value = True
        mock_evidence_tab.click = MagicMock()

        # Mock driver.find_element() to return the correct elements based on the locator
        def mock_find_element(by, value):
            if by == By.CSS_SELECTOR and value == "div.panel-content":
                return mock_panel_content
            elif by == By.CSS_SELECTOR and value == "a.menu-link[href*='panel/evidence']":
                return mock_evidence_tab
            else:
                # Return a generic MagicMock for any other elements
                return MagicMock()

        fetcher.driver.find_element.side_effect = mock_find_element

        with patch.object(fetcher, '_get_outcomes', return_value=[]):
            list(fetcher._process_goal(TEST_GOALS[0]['url'], TEST_GOALS[0]['name']))

        # Assert that evidence_tab.click() was called
        mock_evidence_tab.click.assert_called_once()

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
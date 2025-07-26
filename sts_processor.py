import os
import re
import csv
import time
import statistics
from datetime import datetime, timedelta
from collections import defaultdict, Counter
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.common.exceptions import NoSuchElementException, TimeoutException
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager

class STSClaimsProcessor:
    """STS Claims Data Collection and Processing Class"""
    
    def __init__(self, relief_rate=320.47, export_path=None, headless=False):
        self.relief_rate = relief_rate
        self.export_path = export_path or r"C:\Users\311741\OneDrive - Delta Air Lines\Documents\STS ANALYTICS"
        self.headless = headless
        os.makedirs(self.export_path, exist_ok=True)
        
        # Initialize logging
        self.script_start_time = datetime.now()
        self.timestamp = self.script_start_time.strftime("%Y%m%d_%H%M%S")
        self.logfile = os.path.join(self.export_path, f"sts_claims_outputlog_{self.timestamp}.txt")
        
        self.all_tickets = []
        self.processed_tickets = set()
        self.claims_data = []
        
    def logprint(self, *args, **kwargs):
        """Print and log messages"""
        msg = " ".join(str(a) for a in args)
        print(msg, **kwargs)
        with open(self.logfile, "a", encoding="utf-8") as f:
            f.write(msg + "\n")
    
    def hhmm_to_minutes(self, hhmm):
        """Convert HH:MM format to minutes"""
        try:
            if not hhmm:
                return 0
            hhmm = hhmm.replace(';', ':')
            hours, minutes = map(int, hhmm.strip().split(':'))
            return hours * 60 + minutes
        except:
            return 0

    def relief_dollars(self, minutes):
        """Calculate relief dollars from minutes"""
        return (minutes / 60) * self.relief_rate if minutes else 0

    def group_subject_key(self, violation):
        """Group violations into categories"""
        v = violation.lower()
        if 'rest' in v:
            return 'Rest'
        if any(x in v for x in ['12.t', 'yellow slip', '12t', 'ys']):
            return 'Yellow Slip / 12.T'
        if any(x in v for x in ['11.f', '11f']):
            return '11.F'
        if any(x in v for x in ['payback day', '23.s.11', '23s11']):
            return 'Payback Day / 23.S.11'
        if any(x in v for x in ['23.q', '23q', 'green slip', 'gs']):
            return 'Green Slip / 23.Q'
        if 'green slip' in v:
            return 'Green Slip / 23.Q'
        if 'sc' in v or 'short call' in v:
            return 'Short Call'
        if 'long call' in v or 'lc' in v:
            return 'Long Call'
        if '23.o' in v or '23o' in v:
            return '23.O'
        if 'rotation coverage sequence' in v:
            return 'Rotation Coverage Sequence'
        if 'inverse assignment' in v or '23.r' in v:
            return 'Inverse Assignment / 23.R'
        if any(x in v for x in ['deadhead', '8.d', '8.d.3', '8d3']):
            return 'Deadhead / 8.D'
        if 'swap with the pot' in v:
            return 'Swap With The Pot'
        if '23.j' in v:
            return '23.J'
        if '4.f' in v:
            return '4.F'
        if 'arcos' in v or '23.z' in v:
            return 'ARCOS / 23.Z'
        if any(x in v for x in ['white slip', '23.p']):
            return 'White Slip / 23.P'
        if any(x in v for x in ['reroute', '23.l', '23l']):
            return 'Reroute / 23.L'
        if 'illegal rotation' in v:
            return 'Illegal Rotation'
        if '23.k' in v:
            return '23.K'
        if 'mou 24-01' in v:
            return 'MOU 24-01'
        return violation.strip() or "Unknown"

    def extract_emp_number(self, empcat):
        """Extract employee number from text"""
        m = re.search(r'\b(\d{6})\b', empcat)
        if m:
            return m.group(1)
        return empcat.strip()

    def status_canonical(self, status):
        """Normalize status values"""
        s = status.strip().lower()
        if s == "submitted to company":
            return "open"
        if s == "impasse":
            return "impasse"
        if s == "closed without payment":
            return "denied"
        if s == "paid":
            return "approved"
        if s in ("in review", "archived", "contested"):
            return s
        if s in ("open", "approved", "denied", "impasse"):
            return s
        return s

    def extract_month(self, date_str):
        """Extract month from date string"""
        date_str = date_str.split()[0]
        try:
            dt = datetime.strptime(date_str, "%Y-%m-%d")
            return dt.strftime("%Y-%m")
        except:
            try:
                dt = datetime.strptime(date_str, "%m/%d/%Y")
                return dt.strftime("%Y-%m")
            except:
                return "Unknown"

    def setup_driver(self):
        """Initialize Chrome WebDriver"""
        options = Options()
        if self.headless:
            options.add_argument("--headless")
        service = Service(ChromeDriverManager().install())
        self.driver = webdriver.Chrome(service=service, options=options)
        self.wait = WebDriverWait(self.driver, 10)

    def login_and_navigate(self, username, password):
        """Login to STS portal and navigate to claims"""
        self.logprint(f"Script initiated at: {self.script_start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        self.driver.get("https://sts2.alpa.org/adfs/ls/?wa=wsignin1.0&wtrealm=https%3a%2f%2fdal.alpa.org&wctx=rm%3d0%26id%3d65788f3a-90df-4c13-b375-f2e8ad524a11%26ru%3d%252fsts-admin&wct=2025-07-11T11%3a52%3a42Z&whr=https%3a%2f%2fdal.alpa.org&cfg=6")
        time.sleep(3)
        
        self.driver.find_element(By.ID, "userNameInput").send_keys(username)
        self.driver.find_element(By.ID, "passwordInput").send_keys(password)
        self.driver.find_element(By.ID, "submitButton").click()
        time.sleep(5)
        
        self.driver.get("https://dal.alpa.org/sts-admin/claims")

    def configure_filters(self):
        """Configure page filters and settings"""
        # Click sort & filter button
        sort_filter_button = self.wait.until(EC.element_to_be_clickable((By.XPATH, "/html/body/form/div[3]/div/div/div/div/div/div/main/div[1]/div[1]/div[2]/div[1]/div/div[2]/button[2]")))
        self.driver.execute_script("arguments[0].scrollIntoView(true);", sort_filter_button)
        sort_filter_button.click()
        self.logprint("✅ 'Sort & Filter' panel opened.")
        time.sleep(2)

        # Click Add All
        try:
            add_all_button = self.wait.until(EC.element_to_be_clickable((By.XPATH, "/html/body/form/div[3]/div/div/div/div/div/div/main/div[1]/div[1]/div[2]/div[6]/div[1]/div[1]/div/div[3]/div/div/div[1]/fieldset/div[4]/button[1]")))
            self.driver.execute_script("arguments[0].scrollIntoView(true);", add_all_button)
            add_all_button.click()
            self.logprint("✅ 'Add All' clicked.")
            time.sleep(1)
        except TimeoutException:
            self.logprint("❌ Failed to click 'Add All': Timeout")

        # Apply filters
        try:
            apply_button = self.wait.until(EC.element_to_be_clickable((By.XPATH, "/html/body/form/div[3]/div/div/div/div/div/div/main/div[1]/div[1]/div[2]/div[6]/div[1]/div[2]/div[2]/button[2]")))
            self.driver.execute_script("arguments[0].scrollIntoView(true);", apply_button)
            apply_button.click()
            self.logprint("✅ 'Apply' clicked.")
            time.sleep(2)
        except TimeoutException:
            self.logprint("❌ Failed to click 'Apply': Timeout")

        time.sleep(5)

        # Set results per page to 50
        try:
            per_page_dropdown = self.wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "select.form-select")))
            self.driver.execute_script("arguments[0].scrollIntoView(true);", per_page_dropdown)
            if per_page_dropdown.get_attribute("value") != "50":
                per_page_dropdown.click()
                option_50 = per_page_dropdown.find_element(By.XPATH, ".//option[@value='50']")
                option_50.click()
                time.sleep(3)
        except Exception as e:
            self.logprint("Could not set results per page to 50:", e)

    def collect_ticket_urls(self, max_pages=0):
        """Collect all ticket URLs from the claims list"""
        def get_total_pages():
            try:
                last_btn = self.driver.find_elements(By.CSS_SELECTOR, "div[data-testid='pagination'] button.dt-paging-button")
                if not last_btn:
                    last_btn = self.driver.find_elements(By.CSS_SELECTOR, "button.dt-paging-button")
                page_nums = [int(btn.text.strip()) for btn in last_btn if btn.text.strip().isdigit()]
                if page_nums:
                    return max(page_nums)
                return 1
            except Exception as e:
                self.logprint("Could not determine total pages, defaulting to 1:", e)
                return 1

        total_pages = get_total_pages()
        if max_pages > 0:
            total_pages = min(total_pages, max_pages)
            
        self.logprint(f"Detected {total_pages} total pages.")

        for page_number in range(1, total_pages+1):
            self.logprint(f"Scraping ticket numbers from page {page_number}...")
            
            try:
                WebDriverWait(self.driver, 10).until(
                    EC.presence_of_element_located((By.CSS_SELECTOR, "button.dt-paging-button"))
                )
                page_buttons = self.driver.find_elements(By.CSS_SELECTOR, "button.dt-paging-button")
                found = False
                for btn in page_buttons:
                    if btn.text.strip() == str(page_number):
                        self.driver.execute_script("arguments[0].click();", btn)
                        found = True
                        break
                        
                if not found and page_number != 1:
                    self.logprint(f"Could not find page button for {page_number}, skipping.")
                    continue
                    
                time.sleep(2)
                WebDriverWait(self.driver, 10).until(
                    EC.presence_of_element_located((By.CSS_SELECTOR, "table tbody tr"))
                )
            except Exception as e:
                self.logprint(f"Could not navigate to page {page_number}: {e}")
                continue

            rows = self.driver.find_elements(By.CSS_SELECTOR, "table tbody tr")
            for row_idx in range(len(rows)):
                try:
                    rows = self.driver.find_elements(By.CSS_SELECTOR, "table tbody tr")
                    row = rows[row_idx]
                    cells = row.find_elements(By.TAG_NAME, "td")
                    if len(cells) >= 1:
                        ticket_number = cells[0].text.strip()
                        if ticket_number.isdigit() and ticket_number not in self.processed_tickets:
                            ticket_url = f"https://dal.alpa.org/sts-admin/report/{ticket_number}"
                            self.all_tickets.append({
                                "number": ticket_number, 
                                "url": ticket_url,
                                "row_cells": [c.text.strip() for c in cells]
                            })
                            self.processed_tickets.add(ticket_number)
                except Exception as e:
                    self.logprint("Error extracting ticket from row:", e)

        self.logprint(f"Collected {len(self.all_tickets)} ticket URLs.")

    def scrape_ticket_details(self):
        """Scrape detailed information from each ticket"""
        start_time = datetime.now()
        
        for i, ticket in enumerate(self.all_tickets):
            self.driver.get(ticket["url"])
            time.sleep(2)
            
            try:
                cells = ticket["row_cells"]
                claim = {
                    "Ticket #": cells[0] if len(cells) > 0 else ticket["number"],
                    "Status": cells[1] if len(cells) > 1 else "",
                    "Last Interaction": cells[2] if len(cells) > 2 else "",
                    "Assignee": cells[4] if len(cells) > 4 else "",
                    "Pilot Emp # Category": cells[5] if len(cells) > 5 else "",
                    "Emp #": self.extract_emp_number(cells[5] if len(cells) > 5 else ""),
                    "Subject Violations": cells[6] if len(cells) > 6 else "",
                    "Dispute #": cells[7] if len(cells) > 7 else "",
                    "Incident Date Rot #": f"{cells[9] if len(cells) > 9 else ''} {cells[0] if len(cells) > 0 else ticket['number']}"
                }
                
                # Extract relief requested
                self.wait.until(EC.presence_of_element_located((By.XPATH, "//*[contains(text(), 'Relief Requested')]")))
                relief_element = None
                xpaths = [
                    "//*[contains(text(), 'Relief Requested')]/following::span[1]",
                    "//*[contains(text(), 'Relief Requested')]/following::div[1]",
                    "//*[contains(text(), 'Relief Requested')]/following::p[1]",
                    "//*[contains(text(), 'Relief Requested')]/following::*[1]"
                ]
                
                for xpath in xpaths:
                    try:
                        relief_candidate = self.driver.find_element(By.XPATH, xpath)
                        if relief_candidate.text.strip():
                            relief_element = relief_candidate
                            break
                    except NoSuchElementException:
                        continue

                if relief_element and relief_element.text.strip():
                    relief_text = relief_element.text.strip()
                    minutes = self.hhmm_to_minutes(relief_text)
                    claim["Relief Requested"] = relief_text
                    claim["Relief Minutes"] = minutes
                else:
                    claim["Relief Requested"] = ""
                    claim["Relief Minutes"] = 0

                # Only include claims with relief > 0
                if claim["Relief Minutes"] > 0:
                    self.claims_data.append(claim)
                    self.logprint(f"[{i+1}/{len(self.all_tickets)}] Ticket #{claim['Ticket #']}, Relief: {claim['Relief Requested']}, Emp #: {claim['Emp #']}, Subject: {claim['Subject Violations']}")

                # Progress reporting
                if (i+1) % 50 == 0 or (i+1) == len(self.all_tickets):
                    elapsed = (datetime.now() - start_time).total_seconds()
                    rate = (i+1)/elapsed if elapsed > 0 else 0
                    remaining = (len(self.all_tickets)-(i+1))/rate if rate > 0 else 0
                    self.logprint(f"Processed {i+1}/{len(self.all_tickets)} in {elapsed/60:.1f} min, est {remaining/60:.1f} min remaining")

            except Exception as e:
                self.logprint(f"Error scraping ticket {ticket['number']}: {e}")

    def generate_analytics(self):
        """Generate analytics and export to CSV"""
        outfile = os.path.join(self.export_path, f"sts_claims_analytics_{self.timestamp}.csv")
        
        # Get all statuses
        all_statuses = set()
        for c in self.claims_data:
            canon = self.status_canonical(c.get("Status", ""))
            all_statuses.add(canon)
        all_statuses = sorted(all_statuses)

        # Subject grouped stats
        subject_grouped_stats = defaultdict(lambda: {"count": 0})
        for c in self.claims_data:
            subject = self.group_subject_key(c["Subject Violations"])
            status = self.status_canonical(c["Status"])
            mins = c["Relief Minutes"]
            dollars = self.relief_dollars(mins)
            
            subject_grouped_stats[subject]["count"] += 1
            subject_grouped_stats[subject][status] = subject_grouped_stats[subject].get(status, 0) + 1
            subject_grouped_stats[subject][f"{status}_minutes"] = subject_grouped_stats[subject].get(f"{status}_minutes", 0) + mins
            subject_grouped_stats[subject][f"{status}_dollars"] = subject_grouped_stats[subject].get(f"{status}_dollars", 0) + dollars

        # Export to CSV
        header = ["Subject", "Total Cases"]
        for status in all_statuses:
            header.extend([f"{status.title()} Count", f"{status.title()} Dollars", f"{status.title()} % of Total"])
            
        with open(outfile, "w", newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(header)
            
            for subject, stats in subject_grouped_stats.items():
                row = [subject, stats["count"]]
                total = stats["count"]
                for status in all_statuses:
                    count = stats.get(status, 0)
                    dollars = stats.get(f"{status}_dollars", 0)
                    pct = (count / total * 100) if total else 0
                    row.extend([count, f"${dollars:.2f}", f"{pct:.2f}%"])
                writer.writerow(row)

        self.logprint(f"Analytics exported to {outfile}")
        return outfile

    def run_full_process(self, username, password, max_pages=0):
        """Run the complete data collection and analysis process"""
        try:
            self.setup_driver()
            self.login_and_navigate(username, password)
            self.configure_filters()
            self.collect_ticket_urls(max_pages)
            self.scrape_ticket_details()
            analytics_file = self.generate_analytics()
            
            script_end_time = datetime.now()
            runtime_delta = script_end_time - self.script_start_time
            self.logprint(f"\nScript completed at: {script_end_time.strftime('%Y-%m-%d %H:%M:%S')}")
            self.logprint(f"Total run time: {runtime_delta}")
            
            return {
                'success': True,
                'analytics_file': analytics_file,
                'claims_count': len(self.claims_data),
                'runtime': runtime_delta
            }
            
        except Exception as e:
            self.logprint(f"Error in process: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }
        finally:
            if hasattr(self, 'driver'):
                self.driver.quit()

# For direct script execution
if __name__ == "__main__":
    processor = STSClaimsProcessor()
    result = processor.run_full_process("N0000937", "STSD@L!42AlPa14")
    print("Process result:", result)

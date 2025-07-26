
import sys
import os
sys.path.append(r"C:\Users\311741\sts")

from sts_processor import STSClaimsProcessor

if __name__ == "__main__":
    processor = STSClaimsProcessor(
        relief_rate=320.47,
        export_path=r"C:\Users\311741\OneDrive - Delta Air Lines\Documents\STS ANALYTICS",
        headless=False
    )
    result = processor.run_full_process("N0000937", "STSD@L!42AlPa14", max_pages=0)
    print("Process completed:", result)

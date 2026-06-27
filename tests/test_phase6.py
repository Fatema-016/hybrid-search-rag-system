import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.storage.s3_service import upload_pdf_bytes, download_pdf_bytes, list_pdfs, pdf_exists


test_pdf_path = "data/raw_pdfs/2606.27287v1.pdf"  
test_key = "papers/test_2606.27287v1.pdf"

with open(test_pdf_path, "rb") as f:
    original_bytes = f.read()

print(f"Uploading {len(original_bytes)} bytes to S3...")
upload_pdf_bytes(original_bytes, test_key)
print("Upload succeeded.")

print("Checking existence...")
assert pdf_exists(test_key), "File should exist in S3 but doesn't"
print("Exists check passed.")

print("Downloading back...")
downloaded_bytes = download_pdf_bytes(test_key)
assert downloaded_bytes == original_bytes, "Downloaded bytes don't match original!"
print(f"Round-trip verified: {len(downloaded_bytes)} bytes match exactly.")

print("\nAll PDFs currently in bucket under 'papers/':")
for key in list_pdfs():
    print(" -", key)
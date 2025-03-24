from paddleocr import PaddleOCR
import re

# ✅ Initialize PaddleOCR (English Only for Faster Processing)
ocr = PaddleOCR(lang="en")

# ✅ Function to Extract Financial Transactions
def extract_text_paddleocr(image_path):
    try:
        # Perform OCR with PaddleOCR
        results = ocr.ocr(image_path, cls=True)

        # ✅ Extract and format text
        extracted_text = "\n".join([" ".join([word_info[1][0] for word_info in line]) for line in results])
        print("\n🔍 Extracted Text:\n", extracted_text)

        # ✅ Extract structured financial transactions
        transactions = {}
        pattern = r"([A-Za-z\s]+)\s+\$?([\d,]+\.\d{2})"
        matches = re.findall(pattern, extracted_text)

        for vendor, amount in matches:
            try:
                transactions[vendor.strip()] = f"${float(amount.replace(',', '')):,.2f}"
            except ValueError:
                transactions[vendor.strip()] = amount  # Keep raw value if parsing fails

        return extracted_text, transactions
    except Exception as e:
        print(f"⚠️ OCR Error: {e}")
        return "Error extracting text", {}

# ✅ Load and Process Your Test Image
image_path = r"C:\Users\vivek mani\Downloads\test2.webp"

# ✅ Run OCR Extraction
extracted_text, transactions = extract_text_paddleocr(image_path)

# ✅ Print Extracted Transactions
print("\n🔹 Extracted Transactions:", transactions)

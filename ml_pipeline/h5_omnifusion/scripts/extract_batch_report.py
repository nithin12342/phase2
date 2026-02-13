import fitz

doc = fitz.open(r'c:\Users\thela\OneDrive\Desktop\phase 2\BATCH_18_REPORT.pdf')
text = ""
for page in doc:
    text += page.get_text()

with open('batch_report_text.txt', 'w', encoding='utf-8') as f:
    f.write(text)

print("Extracted", len(text), "characters")
print("=" * 50)
print(text[:8000])

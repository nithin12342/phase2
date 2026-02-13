"""Convert folder_structure_report.md to PDF using xhtml2pdf."""
import markdown2
from xhtml2pdf import pisa

md_file = r"c:\Users\thela\OneDrive\Desktop\phase 2\implementation of the final year project\h5_omnifusion\plan to improve.md"
pdf_file = r"c:\Users\thela\OneDrive\Desktop\phase 2\implementation of the final year project\h5_omnifusion\docs\plan_to_improve.pdf"

with open(md_file, 'r', encoding='utf-8') as f:
    md_content = f.read()

html_content = markdown2.markdown(md_content, extras=['fenced-code-blocks', 'tables'])

styled_html = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <style>
        @page {{
            size: A4;
            margin: 1.5cm;
        }}
        body {{
            font-family: 'Courier New', Courier, monospace;
            font-size: 9px;
            line-height: 1.3;
        }}
        h1 {{
            color: #2c3e50;
            font-size: 18px;
            border-bottom: 2px solid #3498db;
            margin-top: 20px;
        }}
        h2 {{
            color: #34495e;
            font-size: 14px;
            border-bottom: 1px solid #bdc3c7;
        }}
        h3 {{
            color: #7f8c8d;
            font-size: 12px;
        }}
        code {{
            background: #f4f4f4;
            padding: 2px 4px;
            border-radius: 2px;
            font-size: 8px;
        }}
        pre {{
            background: #f5f5f5;
            padding: 10px;
            border-radius: 5px;
            overflow-x: auto;
            font-size: 8px;
            white-space: pre-wrap;
            word-wrap: break-word;
        }}
        table {{
            border-collapse: collapse;
            width: 100%;
            margin: 10px 0;
        }}
        th, td {{
            border: 1px solid #ddd;
            padding: 4px;
            text-align: left;
            font-size: 8px;
        }}
        th {{
            background: #3498db;
            color: white;
        }}
        tr:nth-child(even) {{
            background: #f9f9f9;
        }}
    </style>
</head>
<body>
{html_content}
</body>
</html>
"""

with open(pdf_file, 'wb') as pdf:
    pisa_status = pisa.CreatePDF(styled_html, dest=pdf, encoding='utf-8')

if pisa_status.err:
    print(f"Error creating PDF: {pisa_status.err}")
else:
    print(f"✅ PDF saved successfully to: {pdf_file}")

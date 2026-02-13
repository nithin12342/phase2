"""Convert multiple Markdown files to PDF using markdown2pdf or fpdf2."""
import os
import sys

MD_FILES = [
    r"c:\Users\thela\OneDrive\Desktop\phase 2\implementation of the final year project\h5_omnifusion\docs\H5_TRAINING_STRATEGY.md",
    r"c:\Users\thela\OneDrive\Desktop\phase 2\implementation of the final year project\h5_omnifusion\docs\preprocessing_guide.md",
    r"c:\Users\thela\OneDrive\Desktop\phase 2\implementation of the final year project\h5_omnifusion\docs\ARCHITECTURE.md",
    r"c:\Users\thela\OneDrive\Desktop\phase 2\implementation of the final year project\h5_omnifusion\docs\PREPROCESSING_AND_FEATURE_EXTRACTION.md",
]


def try_md2pdf(md_file: str, pdf_file: str) -> bool:
    """Try using md2pdf library."""
    try:
        from md2pdf.core import md2pdf
        
        md2pdf(
            pdf_file_path=pdf_file,
            md_file_path=md_file,
            css_file_path=None,
            base_url=os.path.dirname(md_file)
        )
        return True
    except Exception as e:
        print(f"   md2pdf failed: {e}")
        return False


def try_markdown_pdf(md_file: str, pdf_file: str) -> bool:
    """Try using markdown-pdf library."""
    try:
        import markdown
        from fpdf import FPDF
        
        with open(md_file, 'r', encoding='utf-8') as f:
            md_content = f.read()
        
        html_content = markdown.markdown(md_content, extensions=['tables', 'fenced_code'])
        
        pdf = FPDF()
        pdf.add_page()
        pdf.set_auto_page_break(auto=True, margin=15)
        
        pdf.add_font('DejaVu', '', 'DejaVuSans.ttf', uni=True) if os.path.exists('DejaVuSans.ttf') else None
        
        try:
            pdf.set_font('DejaVu', size=10)
        except:
            pdf.set_font('Helvetica', size=10)
        
        pdf.write_html(html_content)
        pdf.output(pdf_file)
        return True
        
    except Exception as e:
        print(f"   fpdf2 failed: {e}")
        return False


def try_pdfkit(md_file: str, pdf_file: str) -> bool:
    """Try using pdfkit with wkhtmltopdf."""
    try:
        import pdfkit
        import markdown
        
        with open(md_file, 'r', encoding='utf-8') as f:
            md_content = f.read()
        
        html_content = markdown.markdown(md_content, extensions=['tables', 'fenced_code'])
        
        styled_html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; line-height: 1.6; font-size: 11pt; }}
                h1 {{ color: #1a365d; border-bottom: 2px solid #3182ce; padding-bottom: 10px; }}
                h2 {{ color: #2c5282; border-bottom: 1px solid #90cdf4; margin-top: 25px; }}
                h3 {{ color: #2b6cb0; }}
                code {{ background: #edf2f7; padding: 2px 6px; border-radius: 4px; }}
                pre {{ background: #2d2d2d; color: #f8f8f2; padding: 15px; border-radius: 5px; }}
                table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background: #3182ce; color: white; }}
                blockquote {{ border-left: 4px solid #3182ce; margin: 20px 0; padding-left: 20px; background: #ebf8ff; }}
            </style>
        </head>
        <body>
        {html_content}
        </body>
        </html>
        """
        
        pdfkit.from_string(styled_html, pdf_file)
        return True
        
    except Exception as e:
        print(f"   pdfkit failed: {e}")
        return False


def try_reportlab_simple(md_file: str, pdf_file: str) -> bool:
    """Create a simple PDF using ReportLab."""
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Preformatted
        from reportlab.lib.units import cm
        from reportlab.lib.enums import TA_LEFT
        import re
        
        with open(md_file, 'r', encoding='utf-8') as f:
            md_content = f.read()
        
        doc = SimpleDocTemplate(
            pdf_file,
            pagesize=A4,
            rightMargin=2*cm,
            leftMargin=2*cm,
            topMargin=2*cm,
            bottomMargin=2*cm
        )
        
        styles = getSampleStyleSheet()
        
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=18,
            spaceAfter=20,
            textColor='#1a365d'
        )
        
        h2_style = ParagraphStyle(
            'CustomH2',
            parent=styles['Heading2'],
            fontSize=14,
            spaceBefore=15,
            spaceAfter=10,
            textColor='#2c5282'
        )
        
        h3_style = ParagraphStyle(
            'CustomH3',
            parent=styles['Heading3'],
            fontSize=12,
            spaceBefore=10,
            spaceAfter=8,
            textColor='#2b6cb0'
        )
        
        body_style = ParagraphStyle(
            'CustomBody',
            parent=styles['Normal'],
            fontSize=10,
            spaceAfter=8,
            leading=14
        )
        
        code_style = ParagraphStyle(
            'CustomCode',
            parent=styles['Code'],
            fontSize=8,
            fontName='Courier',
            backColor='#f0f0f0',
            leftIndent=10,
            rightIndent=10,
            spaceBefore=5,
            spaceAfter=5
        )
        
        story = []
        lines = md_content.split('\n')
        in_code_block = False
        code_buffer = []
        
        for line in lines:
            if line.strip().startswith('```'):
                if in_code_block:
                    if code_buffer:
                        code_text = '\n'.join(code_buffer)
                        story.append(Preformatted(code_text, code_style))
                        story.append(Spacer(1, 10))
                    code_buffer = []
                in_code_block = not in_code_block
                continue
            
            if in_code_block:
                code_buffer.append(line)
                continue
            
            if not line.strip():
                story.append(Spacer(1, 6))
                continue
            
            if line.startswith('# '):
                text = line[2:].strip()
                text = text.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
                story.append(Paragraph(text, title_style))
            elif line.startswith('## '):
                text = line[3:].strip()
                text = text.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
                story.append(Paragraph(text, h2_style))
            elif line.startswith('### '):
                text = line[4:].strip()
                text = text.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
                story.append(Paragraph(text, h3_style))
            elif line.startswith('---'):
                story.append(Spacer(1, 20))
            elif line.startswith('|'):
                continue
            elif line.startswith('>'):
                text = line[1:].strip()
                text = text.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
                story.append(Paragraph(f"<i>{text}</i>", body_style))
            else:
                text = line.strip()
                text = text.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
                text = re.sub(r'`([^`]+)`', r'<font face="Courier">\1</font>', text)
                text = re.sub(r'\*\*([^*]+)\*\*', r'<b>\1</b>', text)
                text = re.sub(r'\*([^*]+)\*', r'<i>\1</i>', text)
                
                try:
                    story.append(Paragraph(text, body_style))
                except:
                    story.append(Paragraph(text.encode('ascii', 'ignore').decode(), body_style))
        
        doc.build(story)
        return True
        
    except Exception as e:
        print(f"   reportlab failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def convert_md_to_pdf(md_file: str) -> bool:
    """Try multiple methods to convert markdown to PDF."""
    pdf_file = md_file.replace('.md', '.pdf')
    
    print(f"\n📄 Converting: {os.path.basename(md_file)}")
    
    methods = [
        ("md2pdf", try_md2pdf),
        ("pdfkit", try_pdfkit),
        ("reportlab", try_reportlab_simple),
        ("fpdf2", try_markdown_pdf),
    ]
    
    for name, method in methods:
        print(f"   Trying {name}...")
        if method(md_file, pdf_file):
            print(f"   ✅ Success with {name}!")
            return True
    
    return False


def main():
    print("=" * 60)
    print("       MD to PDF Converter - H5-OmniFusion Docs")
    print("=" * 60)
    
    success_count = 0
    failed_files = []
    
    for md_file in MD_FILES:
        if not os.path.exists(md_file):
            print(f"\n⚠️ File not found: {md_file}")
            failed_files.append(md_file)
            continue
        
        if convert_md_to_pdf(md_file):
            success_count += 1
        else:
            failed_files.append(md_file)
    
    print("\n" + "=" * 60)
    print(f"       Results: {success_count}/{len(MD_FILES)} files converted")
    print("=" * 60)
    
    if failed_files:
        print("\n❌ Failed files:")
        for f in failed_files:
            print(f"   - {os.path.basename(f)}")
        print("\n💡 To install required libraries, run:")
        print("   pip install md2pdf reportlab fpdf2 markdown pdfkit")
    else:
        print("\n🎉 All files converted successfully!")
        print("\nPDF files saved in:")
        print(r"   c:\Users\thela\OneDrive\Desktop\phase 2\implementation of the final year project\h5_omnifusion\docs")


if __name__ == "__main__":
    main()

import os
import glob
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak, Preformatted
from reportlab.lib.enums import TA_LEFT, TA_CENTER
from reportlab.pdfgen import canvas
import re

def combine_markdown_files(directory_path, output_md_file):
    """
    Combine all markdown files in the specified directory into one file.
    """
    md_files = glob.glob(os.path.join(directory_path, "*.md"))
    
    md_files.sort()
    
    print(f"Found {len(md_files)} markdown files")
    
    combined_content = []
    
    for md_file in md_files:
        filename = os.path.basename(md_file)
        print(f"Processing: {filename}")
        
        combined_content.append(f"\n\n{'='*80}\n")
        combined_content.append(f"# FILE: {filename}\n")
        combined_content.append(f"{'='*80}\n\n")
        
        with open(md_file, 'r', encoding='utf-8') as f:
            content = f.read()
            combined_content.append(content)
    
    with open(output_md_file, 'w', encoding='utf-8') as f:
        f.write(''.join(combined_content))
    
    print(f"\nCombined markdown saved to: {output_md_file}")
    return output_md_file

def convert_markdown_to_pdf(md_file, output_pdf_file):
    """
    Convert markdown file to PDF using reportlab.
    """
    print(f"\nConverting to PDF...")
    
    with open(md_file, 'r', encoding='utf-8') as f:
        md_content = f.read()
    
    doc = SimpleDocTemplate(
        output_pdf_file,
        pagesize=A4,
        leftMargin=2*cm,
        rightMargin=2*cm,
        topMargin=2*cm,
        bottomMargin=2*cm
    )
    
    styles = getSampleStyleSheet()
    
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=18,
        textColor='#2c3e50',
        spaceAfter=12,
        spaceBefore=12,
        alignment=TA_LEFT
    )
    
    heading1_style = ParagraphStyle(
        'CustomHeading1',
        parent=styles['Heading1'],
        fontSize=16,
        textColor='#2c3e50',
        spaceAfter=10,
        spaceBefore=10
    )
    
    heading2_style = ParagraphStyle(
        'CustomHeading2',
        parent=styles['Heading2'],
        fontSize=14,
        textColor='#34495e',
        spaceAfter=8,
        spaceBefore=8
    )
    
    heading3_style = ParagraphStyle(
        'CustomHeading3',
        parent=styles['Heading3'],
        fontSize=12,
        textColor='#555555',
        spaceAfter=6,
        spaceBefore=6
    )
    
    body_style = ParagraphStyle(
        'CustomBody',
        parent=styles['Normal'],
        fontSize=10,
        leading=14,
        spaceAfter=6
    )
    
    code_style = ParagraphStyle(
        'Code',
        parent=styles['Code'],
        fontSize=8,
        fontName='Courier',
        leftIndent=20,
        rightIndent=20,
        spaceAfter=10,
        spaceBefore=10
    )
    
    story = []
    lines = md_content.split('\n')
    
    i = 0
    in_code_block = False
    code_buffer = []
    
    while i < len(lines):
        line = lines[i]
        
        if line.strip().startswith('```'):
            if not in_code_block:
                in_code_block = True
                code_buffer = []
            else:
                in_code_block = False
                if code_buffer:
                    code_text = '\n'.join(code_buffer)
                    code_text = code_text.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
                    try:
                        story.append(Preformatted(code_text, code_style))
                    except:
                        story.append(Paragraph(code_text, body_style))
                code_buffer = []
            i += 1
            continue
        
        if in_code_block:
            code_buffer.append(line)
            i += 1
            continue
        
        if line.startswith('# '):
            text = line[2:].strip()
            text = text.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
            story.append(Paragraph(text, heading1_style))
            story.append(Spacer(1, 0.2*cm))
        elif line.startswith('## '):
            text = line[3:].strip()
            text = text.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
            story.append(Paragraph(text, heading2_style))
            story.append(Spacer(1, 0.2*cm))
        elif line.startswith('### '):
            text = line[4:].strip()
            text = text.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
            story.append(Paragraph(text, heading3_style))
            story.append(Spacer(1, 0.1*cm))
        elif line.startswith('='*80):
            story.append(Spacer(1, 0.5*cm))
        elif line.strip():
            text = line.strip()
            text = text.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
            text = re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', text)  # Bold
            text = re.sub(r'\*(.+?)\*', r'<i>\1</i>', text)  # Italic
            text = re.sub(r'`(.+?)`', r'<font name="Courier">\1</font>', text)  # Inline code
            
            try:
                story.append(Paragraph(text, body_style))
            except Exception as e:
                story.append(Paragraph(line.strip(), body_style))
        else:
            story.append(Spacer(1, 0.3*cm))
        
        i += 1
    
    doc.build(story)
    
    print(f"PDF saved to: {output_pdf_file}")

if __name__ == "__main__":
    yuup_directory = r"c:\Users\thela\OneDrive\Desktop\phase 2\implementation of the final year project\yuup"
    output_md = r"c:\Users\thela\OneDrive\Desktop\phase 2\implementation of the final year project\yuup\combined_responses.md"
    output_pdf = r"c:\Users\thela\OneDrive\Desktop\phase 2\implementation of the final year project\yuup\combined_responses.pdf"
    
    print("="*80)
    print("COMBINING MARKDOWN FILES AND CONVERTING TO PDF")
    print("="*80)
    
    combined_md_file = combine_markdown_files(yuup_directory, output_md)
    
    convert_markdown_to_pdf(combined_md_file, output_pdf)
    
    print("\n" + "="*80)
    print("PROCESS COMPLETED SUCCESSFULLY!")
    print("="*80)
    print(f"\nCombined Markdown: {output_md}")
    print(f"PDF Output: {output_pdf}")

#!/usr/bin/env python3
"""
Markdown to PDF Converter Script

Usage:
    python md2pdf.py <input.md> [output.pdf]

If output.pdf is not specified, it will use the same name as input with .pdf extension.

Requirements:
    pip install markdown weasyprint==59.0 pydyf==0.8.0
"""

import sys
import os
import argparse
import markdown
from weasyprint import HTML


def convert_md_to_pdf(input_file: str, output_file: str = None, css_style: str = None):
    """
    Convert a Markdown file to PDF.
    
    Args:
        input_file: Path to the input Markdown file
        output_file: Path to the output PDF file (optional)
        css_style: Custom CSS style string (optional)
    """
    # Determine output filename
    if output_file is None:
        base_name = os.path.splitext(input_file)[0]
        output_file = f"{base_name}.pdf"
    
    # Read markdown file
    with open(input_file, "r", encoding="utf-8") as f:
        md_content = f.read()
    
    # Convert markdown to HTML with extensions
    html_content = markdown.markdown(
        md_content, 
        extensions=['tables', 'fenced_code', 'codehilite', 'toc']
    )
    
    # Default CSS style with Chinese font support
    if css_style is None:
        css_style = """
        @page {
            size: A4;
            margin: 2cm;
        }
        body {
            font-family: "Noto Sans CJK SC", "WenQuanYi Micro Hei", "Microsoft YaHei", "SimHei", "PingFang SC", sans-serif;
            font-size: 11pt;
            line-height: 1.6;
            color: #333;
        }
        h1 {
            font-size: 22pt;
            color: #222;
            border-bottom: 2px solid #333;
            padding-bottom: 10px;
            margin-top: 30px;
        }
        h2 {
            font-size: 16pt;
            color: #333;
            margin-top: 25px;
            border-bottom: 1px solid #ddd;
            padding-bottom: 5px;
        }
        h3 {
            font-size: 13pt;
            color: #444;
            margin-top: 20px;
        }
        h4 {
            font-size: 12pt;
            color: #555;
        }
        code {
            background-color: #f4f4f4;
            padding: 2px 6px;
            border-radius: 3px;
            font-family: "Courier New", "Consolas", monospace;
            font-size: 10pt;
        }
        pre {
            background-color: #f4f4f4;
            padding: 15px;
            border-radius: 5px;
            overflow-x: auto;
            border: 1px solid #ddd;
        }
        pre code {
            background: none;
            padding: 0;
        }
        img {
            max-width: 100%;
            height: auto;
            display: block;
            margin: 15px auto;
        }
        table {
            border-collapse: collapse;
            width: 100%;
            margin: 20px 0;
        }
        th, td {
            border: 1px solid #ddd;
            padding: 10px;
            text-align: left;
        }
        th {
            background-color: #f2f2f2;
            font-weight: bold;
        }
        tr:nth-child(even) {
            background-color: #fafafa;
        }
        ul, ol {
            padding-left: 25px;
        }
        li {
            margin-bottom: 5px;
        }
        blockquote {
            border-left: 4px solid #ccc;
            margin: 15px 0;
            padding: 10px 20px;
            background-color: #f9f9f9;
            color: #666;
        }
        a {
            color: #0066cc;
            text-decoration: none;
        }
        hr {
            border: none;
            border-top: 1px solid #ddd;
            margin: 30px 0;
        }
        strong {
            font-weight: bold;
        }
        em {
            font-style: italic;
        }
        """
    
    # Wrap with full HTML structure
    full_html = f"""
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <style>
    {css_style}
    </style>
</head>
<body>
{html_content}
</body>
</html>
"""
    
    # Get base directory for resolving relative paths (images, etc.)
    base_url = os.path.dirname(os.path.abspath(input_file))
    
    # Convert to PDF
    HTML(string=full_html, base_url=base_url).write_pdf(output_file)
    
    print(f"✓ PDF created successfully: {output_file}")
    return output_file


def main():
    parser = argparse.ArgumentParser(
        description="Convert Markdown file to PDF with Chinese support",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python md2pdf.py README.md
    python md2pdf.py document.md output.pdf
    python md2pdf.py "文档.md" "输出.pdf"
        """
    )
    parser.add_argument("input", help="Input Markdown file path")
    parser.add_argument("output", nargs="?", help="Output PDF file path (optional)")
    
    args = parser.parse_args()
    
    # Check if input file exists
    if not os.path.exists(args.input):
        print(f"Error: Input file '{args.input}' not found.")
        sys.exit(1)
    
    try:
        convert_md_to_pdf(args.input, args.output)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

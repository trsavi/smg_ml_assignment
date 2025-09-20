#!/usr/bin/env python3
"""
Simple script to convert the testing checklist markdown to PDF.
This creates a printable version of the testing checklist.
"""

import os
import sys
from pathlib import Path

def create_simple_pdf():
    """Create a simple PDF version of the testing checklist."""
    
    # Read the markdown file
    md_file = Path("TESTING_CHECKLIST.md")
    if not md_file.exists():
        print("Error: TESTING_CHECKLIST.md not found")
        return False
    
    with open(md_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Convert markdown to simple HTML for PDF
    html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Madrid Housing ML Pipeline - Testing Checklist</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
            color: #333;
        }}
        h1 {{
            color: #2c3e50;
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #34495e;
            border-bottom: 1px solid #bdc3c7;
            padding-bottom: 5px;
            margin-top: 30px;
        }}
        h3 {{
            color: #2980b9;
            margin-top: 25px;
        }}
        .checklist {{
            background: #f8f9fa;
            padding: 15px;
            border-left: 4px solid #3498db;
            margin: 10px 0;
        }}
        code {{
            background: #f1f2f6;
            padding: 2px 6px;
            border-radius: 3px;
            font-family: 'Courier New', monospace;
            color: #e74c3c;
        }}
        pre {{
            background: #2c3e50;
            color: #ecf0f1;
            padding: 15px;
            border-radius: 5px;
            overflow-x: auto;
        }}
        .test-section {{
            margin: 20px 0;
            padding: 15px;
            border: 1px solid #ddd;
            border-radius: 5px;
        }}
        .test-number {{
            font-weight: bold;
            color: #e74c3c;
        }}
        ul {{
            list-style-type: none;
            padding-left: 0;
        }}
        li {{
            margin: 8px 0;
            padding-left: 20px;
            position: relative;
        }}
        li:before {{
            content: "☐";
            position: absolute;
            left: 0;
            color: #3498db;
            font-weight: bold;
        }}
        .summary {{
            background: #e8f5e8;
            padding: 20px;
            border-radius: 5px;
            margin: 20px 0;
        }}
        .quick-commands {{
            background: #fff3cd;
            padding: 15px;
            border-radius: 5px;
            margin: 15px 0;
        }}
        @media print {{
            body {{
                font-size: 12px;
            }}
            .test-section {{
                page-break-inside: avoid;
            }}
        }}
    </style>
</head>
<body>
    <h1>🧪 Madrid Housing ML Pipeline - Testing Checklist</h1>
    
    <div class="summary">
        <h2>📋 Overview</h2>
        <p>This checklist covers comprehensive testing of all scripts, modes, and configurations in the Madrid Housing Market ML Pipeline.</p>
        <p><strong>Total Tests:</strong> 50+ individual test cases</p>
        <p><strong>Estimated Time:</strong> 2-3 hours for complete testing</p>
    </div>
"""

    # Convert markdown content to HTML (simple conversion)
    lines = content.split('\n')
    in_code_block = False
    
    for line in lines:
        if line.strip().startswith('```'):
            if not in_code_block:
                html_content += '<pre><code>'
                in_code_block = True
            else:
                html_content += '</code></pre>'
                in_code_block = False
        elif in_code_block:
            html_content += line + '\n'
        elif line.startswith('# '):
            html_content += f'<h1>{line[2:]}</h1>\n'
        elif line.startswith('## '):
            html_content += f'<h2>{line[3:]}</h2>\n'
        elif line.startswith('### '):
            html_content += f'<h3>{line[4:]}</h3>\n'
        elif line.startswith('- [ ]'):
            html_content += f'<li>{line[5:]}</li>\n'
        elif line.startswith('- [x]'):
            html_content += f'<li style="text-decoration: line-through; color: #7f8c8d;">{line[5:]}</li>\n'
        elif line.strip().startswith('```bash') or line.strip().startswith('```'):
            continue
        elif line.strip().startswith('**Test'):
            html_content += f'<div class="test-section"><p class="test-number">{line.strip()}</p>\n'
        elif line.strip() == '' and html_content.endswith('</div>\n'):
            continue
        elif line.strip() == '':
            html_content += '<br>\n'
        else:
            # Convert inline code
            line = line.replace('`', '<code>').replace('`', '</code>')
            html_content += f'<p>{line}</p>\n'
    
    html_content += """
    </body>
</html>
"""
    
    # Write HTML file
    html_file = Path("TESTING_CHECKLIST.html")
    with open(html_file, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"✅ Created HTML version: {html_file}")
    print(f"📄 You can open {html_file} in your browser and print to PDF")
    print(f"🌐 Or use online tools like: https://www.markdowntopdf.com/")
    
    return True

if __name__ == "__main__":
    print("Creating PDF version of testing checklist...")
    if create_simple_pdf():
        print("✅ HTML version created successfully!")
        print("\nTo create PDF:")
        print("1. Open TESTING_CHECKLIST.html in your browser")
        print("2. Press Ctrl+P (or Cmd+P on Mac)")
        print("3. Choose 'Save as PDF'")
        print("4. Save as TESTING_CHECKLIST.pdf")
    else:
        print("❌ Failed to create PDF version")
        sys.exit(1)


import nbformat
import re
import sys


def convert_notebook(filepath):
    print(f"Converting {filepath}...")
    with open(filepath, 'r', encoding='utf-8') as f:
        nb = nbformat.read(f, as_version=4)

    new_cells = []
    
    # Regex to capture fenced code blocks: ```language ... ```
    # Captures: (pre_content), (language), (code), (post_content)
    # This is a bit complex because there might be multiple blocks in one cell.
    # We will use a split approach.
    
    code_block_pattern = re.compile(r'```(rust|typescript|ts|python)\n(.*?)```', re.DOTALL)

    for cell in nb.cells:
        if cell.cell_type != 'markdown':
            new_cells.append(cell)
            continue

        content = cell.source
        last_pos = 0
        
        matches = list(code_block_pattern.finditer(content))
        
        if not matches:
            new_cells.append(cell)
            continue
            
        for match in matches:
            # Add markdown text before the code block
            pre_text = content[last_pos:match.start()].strip()
            if pre_text:
                new_cells.append(nbformat.v4.new_markdown_cell(pre_text))
            
            # Extract language and code
            lang = match.group(1)
            if lang == 'ts': lang = 'typescript'
            code = match.group(2).strip()
            
            # Create code cell
            code_cell = nbformat.v4.new_code_cell(code)
            # Add polyglot/kernel metadata if needed. 
            # For now, we set the language metadata which is often sufficient for polyglot extensions.
            code_cell.metadata['language'] = lang
            
            # If it's the main language (e.g. python), we might not need special metadata 
            # if the kernel is python, but explicit is better for polyglot.
            
            new_cells.append(code_cell)
            
            last_pos = match.end()
            
        # Add remaining markdown text
        post_text = content[last_pos:].strip()
        if post_text:
            new_cells.append(nbformat.v4.new_markdown_cell(post_text))

    nb.cells = new_cells
    
    # Update kernel spec to be polyglot-friendly if possible, or leave as is.
    # The user asked for "polyglot notebooks". 
    # Often this implies using the .NET Interactive kernel or similar.
    # We will leave the kernelspec as is for now, assuming the user will open it 
    # with a tool that respects the per-cell language metadata.
    
    with open(filepath, 'w', encoding='utf-8') as f:
        nbformat.write(nb, f)
    print(f"Converted {filepath} successfully.")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python conversion_script.py <notebook_path> [notebook_path ...]")
        sys.exit(1)
        
    for path in sys.argv[1:]:
        convert_notebook(path)

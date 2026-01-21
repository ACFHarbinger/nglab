import nbformat
import glob

def verify_notebooks(pattern):
    files = glob.glob(pattern)
    print(f"Verifying {len(files)} notebooks...")
    
    total_polyglot_cells = 0
    
    for filepath in files:
        if "convert_polyglot" in filepath or "verify_polyglot" in filepath:
            continue
            
        with open(filepath, 'r', encoding='utf-8') as f:
            nb = nbformat.read(f, as_version=4)
            
        polyglot_cells = 0
        details = []
        for i, cell in enumerate(nb.cells):
            if cell.cell_type == 'code':
                lang = cell.metadata.get('language')
                if lang and lang.lower() not in ['python', 'py']:
                    polyglot_cells += 1
                    details.append(f"Cell {i}: {lang}")
        
        if polyglot_cells > 0:
            print(f"{filepath}: {polyglot_cells} polyglot cells found.")
            # print(f"  Details: {', '.join(details)}")
            total_polyglot_cells += polyglot_cells
        else:
            print(f"{filepath}: No polyglot cells found (pure Python or no metadata).")

    print(f"\nTotal polyglot cells across all notebooks: {total_polyglot_cells}")

if __name__ == "__main__":
    verify_notebooks("/home/pkhunter/Repositories/nglab/notebooks/*.ipynb")

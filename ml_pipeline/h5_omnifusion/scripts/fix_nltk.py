
import json
import os

NOTEBOOK_PATH = r"c:\Users\thela\OneDrive\Desktop\phase 2\ml_pipeline\h5_omnifusion\notebooks\H5_OmniFusion_Colab_Runner.ipynb"

def fix_notebook():
    with open(NOTEBOOK_PATH, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    
    
    found = False
    for cell in nb['cells']:
        if cell['cell_type'] == 'code':
            source_joined = "".join(cell['source'])
            if "pandas tabulate" in source_joined:
                print("Found dependency cell.")
                new_source = []
                for line in cell['source']:
                    new_source.append(line)
                    if "pandas tabulate" in line:
                        new_source.append("\n")
                        new_source.append("import nltk\n")
                        new_source.append("try:\n")
                        new_source.append("    nltk.download('vader_lexicon')\n")
                        new_source.append("    nltk.download('punkt')\n")
                        new_source.append("except:\n")
                        new_source.append("    pass\n")
                        new_source.append("\n")
                
                cell['source'] = new_source
                found = True
                break
    
    if found:
        with open(NOTEBOOK_PATH, 'w', encoding='utf-8') as f:
            json.dump(nb, f, indent=2)
        print("✅ Notebook patched successfully with nltk.download")
    else:
        print("❌ Dependency cell not found!")

if __name__ == "__main__":
    fix_notebook()

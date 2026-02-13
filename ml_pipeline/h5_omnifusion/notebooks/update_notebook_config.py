
import json

base_notebook_path = r"c:\Users\thela\OneDrive\Desktop\phase 2\ml_pipeline\h5_omnifusion\notebooks\H5_OmniFusion_Colab_Runner_YouTube.ipynb"

with open(base_notebook_path, 'r', encoding='utf-8') as f:
    notebook_content = json.load(f)


config_cell = notebook_content['cells'][11]

source_text = "".join(config_cell['source'])
if "class Config:" not in source_text or "CFG = Config()" not in source_text:
    print("Error: Could not find Config class in expected cell 11. Searching...")
    for i, cell in enumerate(notebook_content['cells']):
        if cell['cell_type'] == 'code':
            src = "".join(cell['source'])
            if "class Config:" in src:
                config_cell = cell
                print(f"Found Config in cell index {i}")
                break

new_source = []
for line in config_cell['source']:
    if "EMBED_DIM: int = 768" in line:
        new_source.append(line)
        new_source.append("    # Missing paths required by H5OmniFusionPipeline init\n")
        new_source.append("    DAIC_WOZ_PATH: str = '/content/drive/MyDrive/DAIC-WOZ_Datasets/DAIC-WOZ'\n")
        new_source.append("    EXTENDED_DAIC_WOZ_PATH: str = '/content/drive/MyDrive/DAIC-WOZ_Datasets/Extended-DAIC-WOZ/data'\n")
        new_source.append("    EATD_CORPUS_PATH: str = '/content/drive/MyDrive/DAIC-WOZ_Datasets/EATD-Corpus/EATD-Corpus'\n")
        new_source.append("    OUTPUT_DAIC_WOZ: str = '/content/drive/MyDrive/DAIC-WOZ_Datasets/H5_OmniFusion_Output/DAIC-WOZ'\n")
        new_source.append("    OUTPUT_EXTENDED_DAIC: str = '/content/drive/MyDrive/DAIC-WOZ_Datasets/H5_OmniFusion_Output/Extended-DAIC'\n")
        new_source.append("    OUTPUT_EATD: str = '/content/drive/MyDrive/DAIC-WOZ_Datasets/H5_OmniFusion_Output/EATD-Corpus'\n")
    else:
        new_source.append(line)

config_cell['source'] = new_source

with open(base_notebook_path, 'w', encoding='utf-8') as f:
    json.dump(notebook_content, f, indent=2)

print("Updated H5_OmniFusion_Colab_Runner_YouTube.ipynb with missing paths.")

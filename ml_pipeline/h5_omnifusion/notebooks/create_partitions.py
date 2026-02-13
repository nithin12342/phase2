
import json

base_notebook_path = r"c:\Users\thela\OneDrive\Desktop\phase 2\ml_pipeline\h5_omnifusion\notebooks\H5_OmniFusion_Colab_Runner_YouTube.ipynb"

with open(base_notebook_path, 'r', encoding='utf-8') as f:
    notebook_content = json.load(f)

batches = [
    (0, 120),
    (120, 240),
    (240, 360),
    (360, 480),
    (480, 600),
    (600, 720),
    (720, 840),
    (840, 962) # Last batch goes to end
]

for start, end in batches:
    new_notebook = json.loads(json.dumps(notebook_content))
    
    
    loop_cell = new_notebook['cells'][-1]
    
    new_source = []
    for line in loop_cell['source']:
        if "if not df_videos.empty:" in line:
            new_source.append(f"START_IDX = {start}\n")
            new_source.append(f"END_IDX = {end}\n")
            new_source.append(f"print(f'🚀 Processing batch: {{START_IDX}} to {{END_IDX}}')\n")
            new_source.append(f"df_batch = df_videos.iloc[START_IDX:END_IDX]\n")
            new_source.append(line)
        elif "for idx, row in df_videos.iterrows():" in line:
            new_source.append("    for idx, row in df_batch.iterrows():\n")
        elif "print(f\"🚀 Starting processing of {len(df_videos)} videos...\")" in line:
             new_source.append(f"    print(f\"🚀 Starting processing of {{len(df_batch)}} videos (Indices {{START_IDX}}-{{END_IDX}})...\")\n")
        else:
            new_source.append(line)
            
    loop_cell['source'] = new_source
    
    new_filename = f"H5_OmniFusion_Colab_Runner_YouTube_{start}_{end}.ipynb"
    new_path = base_notebook_path.replace("H5_OmniFusion_Colab_Runner_YouTube.ipynb", new_filename)
    
    with open(new_path, 'w', encoding='utf-8') as f:
        json.dump(new_notebook, f, indent=2)
    
    print(f"Created {new_filename}")

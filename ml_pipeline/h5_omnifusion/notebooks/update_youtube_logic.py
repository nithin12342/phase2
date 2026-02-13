
import json
import os

base_notebook_path = r"c:\Users\thela\OneDrive\Desktop\phase 2\ml_pipeline\h5_omnifusion\notebooks\H5_OmniFusion_Colab_Runner_YouTube.ipynb"

with open(base_notebook_path, 'r', encoding='utf-8') as f:
    notebook_content = json.load(f)

target_cell = None
target_idx = -1

for i, cell in enumerate(notebook_content['cells']):
    if cell.get('metadata', {}).get('id') == 'youtube_proc_code':
        target_cell = cell
        target_idx = i
        break
    if cell['cell_type'] == 'code' and "def process_youtube_participant" in "".join(cell['source']):
        target_cell = cell
        target_idx = i
        break

if target_cell is None:
    print("Error: Could not find process_youtube_participant cell.")
    exit(1)

new_code = [
    "import glob\n",
    "import subprocess\n",
    "import shutil\n",
    "import numpy as np\n",
    "import os\n",
    "import time\n",
    "from tqdm.auto import tqdm\n",
    "\n",
    "def process_youtube_participant(pipeline, pid, video_key, label, output_dir):\n",
    "    # --- CHECKPOINT & RACE PROTECTION ---\n",
    "    out_file = os.path.join(output_dir, f\"{pid}.h5\")\n",
    "    lock_file = os.path.join(output_dir, f\"{pid}.lock\")\n",
    "    \n",
    "    # 1. Checkpoint: Skip if output exists\n",
    "    if os.path.exists(out_file):\n",
    "        return\n",
    "    \n",
    "    # 2. Race Condition: Check lock\n",
    "    if os.path.exists(lock_file):\n",
    "        try:\n",
    "            if time.time() - os.path.getmtime(lock_file) > 7200:\n",
    "                print(f\"⚠️ Removing stale lock for {pid}\")\n",
    "                try: os.remove(lock_file)\n",
    "                except: pass\n",
    "            else:\n",
    "                print(f\"🔒 Skipping {pid} (Locked by another instance)\")\n",
    "                return\n",
    "        except OSError:\n",
    "             pass\n",
    "    \n",
    "    # 3. Acquire Lock\n",
    "    try:\n",
    "        with open(lock_file, 'w') as f:\n",
    "            f.write(str(time.time()))\n",
    "    except OSError:\n",
    "        print(f\"🔒 Could not acquire lock for {pid}\")\n",
    "        return\n",
    "        \n",
    "    start_time = time.time()\n",
    "    print(f\"\\n{'='*50}\")\n",
    "    print(f\"📂 Processing {pid} (Key: {video_key}, Label: {label})\")\n",
    "    \n",
    "    url = f\"https://www.youtube.com/watch?v={video_key}\"\n",
    "    work_dir = os.path.join(CFG.TEMP_PATH, str(pid))\n",
    "    if os.path.exists(work_dir):\n",
    "        shutil.rmtree(work_dir)\n",
    "    os.makedirs(work_dir, exist_ok=True)\n",
    "    \n",
    "    try:\n",
    "        # 1. Download Video\n",
    "        print(f\"   ⬇️ Downloading {url}...\")\n",
    "        cmd = [\n",
    "            'yt-dlp', \n",
    "            '-f', 'best[ext=mp4]/best',\n",
    "            '-o', os.path.join(work_dir, 'video.%(ext)s'),\n",
    "            '--write-subs', '--write-auto-subs', '--sub-lang', 'en', '--sub-format', 'vtt',\n",
    "            '--no-check-certificate',\n",
    "            '--ignore-errors',\n",
    "            url\n",
    "        ]\n",
    "        subprocess.run(cmd, check=False)\n",
    "        \n",
    "        video_files = glob.glob(os.path.join(work_dir, 'video.*'))\n",
    "        video_files = [f for f in video_files if not f.endswith('.vtt') and not f.endswith('.part') and not f.endswith('.ytdl')]\n",
    "        \n",
    "        video_path = None\n",
    "        if video_files:\n",
    "            video_path = video_files[0]\n",
    "            print(f\"   🎥 Found video: {os.path.basename(video_path)}\")\n",
    "        else:\n",
    "            print(f\"   ❌ Download failed for {pid} (no video file found)\")\n",
    "            return\n",
    "        \n",
    "        # Extract Audio\n",
    "        audio_path = os.path.join(work_dir, 'audio.wav')\n",
    "        try:\n",
    "            subprocess.run([\n",
    "                'ffmpeg', '-y', '-i', video_path, \n",
    "                '-vn', '-acodec', 'pcm_s16le', '-ar', '16000', '-ac', '1', \n",
    "                audio_path\n",
    "            ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)\n",
    "        except Exception as e:\n",
    "            print(f\"   ⚠️ Audio extraction issue: {repr(e)}\")\n",
    "        \n",
    "        # Find subtitles\n",
    "        vtt_files = glob.glob(os.path.join(work_dir, '*.vtt'))\n",
    "        transcript_text = \"\"\n",
    "        \n",
    "        if vtt_files:\n",
    "            print(f\"   📝 Found subtitles: {os.path.basename(vtt_files[0])}\")\n",
    "            try:\n",
    "                with open(vtt_files[0], 'r', encoding='utf-8') as f:\n",
    "                    lines = f.readlines()\n",
    "                    seen_lines = set()\n",
    "                    for line in lines:\n",
    "                        line = line.strip()\n",
    "                        if '-->' in line or line == 'WEBVTT' or not line: continue\n",
    "                        if line not in seen_lines:\n",
    "                            transcript_text += line + \" \"\n",
    "                            seen_lines.add(line)\n",
    "            except Exception as e:\n",
    "                print(f\"   ⚠️ Subtitle parse error: {repr(e)}\")\n",
    "        \n",
    "        result = {'participant_id': pid, 'dataset': 'youtube_dvlog'}\n",
    "        pipeline._safe_update(result, {'phq8_score': 10.0 if str(label).lower() == 'depression' else 0.0}, 'metadata')\n",
    "        result['phq8_score'] = 10.0 if str(label).lower() == 'depression' else 0.0\n",
    "        \n",
    "        # 2. Process Audio (CORRECTED METHOD NAME)\n",
    "        if os.path.exists(audio_path):\n",
    "            print(\"   🎵 Processing Audio...\")\n",
    "            audio_feats = pipeline.audio.process_audio(audio_path, None)\n",
    "            pipeline._safe_update(result, audio_feats, 'audio')\n",
    "        \n",
    "        # 3. Process Text (CORRECTED METHOD NAME)\n",
    "        if transcript_text:\n",
    "            print(\"   📜 Processing Text...\")\n",
    "            text_feats = pipeline.text.process_text(text=transcript_text)\n",
    "            pipeline._safe_update(result, text_feats, 'text')\n",
    "        else:\n",
    "            print(\"   ⚠️ No text found\")\n",
    "            \n",
    "        # 4. Process Video/Face\n",
    "        if os.path.exists(video_path):\n",
    "            print(\"   🎬 Processing Video/Face...\")\n",
    "            frames = pipeline.video.extractor.extract(video_path)\n",
    "            \n",
    "            video_feats = pipeline.video.process_frames(frames)\n",
    "            pipeline._safe_update(result, video_feats, 'video')\n",
    "            \n",
    "            face_feats = pipeline.face.process_frames(frames)\n",
    "            pipeline._safe_update(result, face_feats, 'face')\n",
    "\n",
    "        # 5. Fusion & Tabular\n",
    "        if 'audio_embedding' not in result: result['audio_embedding'] = np.zeros(768)\n",
    "        if 'text_embedding' not in result: result['text_embedding'] = np.zeros(768)\n",
    "        if 'video_embedding' not in result: result['video_embedding'] = np.zeros(768)\n",
    "        if 'face_embedding' not in result: result['face_embedding'] = np.zeros(768)\n",
    "        \n",
    "        scalar_features = []\n",
    "        EXPECTED_SCALARS = pipeline.cfg.EXPECTED_SCALAR_FEATURES if hasattr(pipeline.cfg, 'EXPECTED_SCALAR_FEATURES') else []\n",
    "        from pipeline_fusion_main import EXPECTED_SCALAR_FEATURES\n",
    "        \n",
    "        for key in EXPECTED_SCALAR_FEATURES:\n",
    "             val = result.get(key, 0.0)\n",
    "             if isinstance(val, (np.ndarray, list)):\n",
    "                 val = 0.0\n",
    "             if isinstance(val, (int, float)) and not np.isnan(val):\n",
    "                 scalar_features.append(pipeline.num_norm.transform(val, key))\n",
    "             else:\n",
    "                 scalar_features.append(0.0)\n",
    "        \n",
    "        scalar_features.append(0.5)\n",
    "        \n",
    "        scalar_tensor = torch.tensor(scalar_features, dtype=torch.float32).unsqueeze(0).to(pipeline.cfg.DEVICE)\n",
    "        tabular_emb = pipeline.tabular_projector(scalar_tensor)\n",
    "        result['tabular_embedding'] = tabular_emb.cpu().detach().numpy().flatten()\n",
    "        \n",
    "        embeddings_to_fuse = {\n",
    "            'audio': result.get('audio_embedding'),\n",
    "            'text': result.get('text_embedding'),\n",
    "            'video': result.get('video_embedding'),\n",
    "            'face': result.get('face_embedding'),\n",
    "            'tabular': result['tabular_embedding']\n",
    "        }\n",
    "        \n",
    "        quality = {\n",
    "             'audio': float(result.get('audio_snr', 0.5) / 100),\n",
    "             'text': min(1.0, len(transcript_text.split()) / 100),\n",
    "             'video': result.get('video_quality_score', 0.5),\n",
    "             'face': result.get('face_detection_rate', 0.5),\n",
    "             'tabular': 1.0\n",
    "        }\n",
    "        \n",
    "        if pipeline.fusion:\n",
    "            try:\n",
    "                result['fusion_embedding'] = pipeline.fusion.fuse(embeddings_to_fuse, quality)\n",
    "            except Exception as e:\n",
    "                print(f\"Fusion error: {repr(e)}\")\n",
    "                result['fusion_embedding'] = np.zeros(768)\n",
    "            \n",
    "        # 6. Save\n",
    "        out_file = os.path.join(output_dir, f\"{pid}.h5\")\n",
    "        pipeline.save_to_h5([result], out_file)\n",
    "        end_time = time.time()\n",
    "        print(f\"   ✅ Saved {pid}.h5 (⏱️ {end_time - start_time:.2f}s)\")\n",
    "        \n",
    "    except Exception as e:\n",
    "        print(f\"   ❌ Error: {repr(e)}\")\n",
    "        import traceback\n",
    "        traceback.print_exc()\n",
    "    finally:\n",
    "        if os.path.exists(work_dir):\n",
    "            shutil.rmtree(work_dir)\n",
    "        if os.path.exists(lock_file):\n",
    "            try: os.remove(lock_file)\n",
    "            except: pass\n"
]

target_cell['source'] = new_code

start_cell = None
loop_cell_idx = -1

for i, cell in enumerate(notebook_content['cells']):
    if cell['cell_type'] == 'code' and "for idx, row in df_videos.iterrows():" in "".join(cell['source']):
        start_cell = cell
        loop_cell_idx = i
        break

if start_cell:
    source = "".join(start_cell['source'])
    if "tqdm" not in source:
        new_source = source.replace(
            "for idx, row in df_videos.iterrows():",
            "for idx, row in tqdm(df_videos.iterrows(), total=df_videos.shape[0], desc='Processing Videos'):"
        )
        new_source = "from tqdm.notebook import tqdm\n" + new_source
        notebook_content['cells'][loop_cell_idx]['source'] = new_source.splitlines(True)
        print("Successfully injected tqdm progress bar.")

with open(base_notebook_path, 'w', encoding='utf-8') as f:
    json.dump(notebook_content, f, indent=2)

print("Updated H5_OmniFusion_Colab_Runner_YouTube.ipynb with CORRECT API method names.")

import os
import json

def merge_json_files(folder_path, output_filename='metadata_ALL.json'):
    merged_data = {}

    for filename in os.listdir(folder_path):
        if filename.endswith('.json'):
            file_path = os.path.join(folder_path, filename)
            print(f"Reading: {filename}")
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    merged_data.update(data)
            except Exception as e:
                print(f"❌ Failed to load {filename}: {e}")

    output_path = os.path.join(folder_path, output_filename)
    with open(output_path, 'w', encoding='utf-8') as f_out:
        json.dump(merged_data, f_out, indent=4)
    print(f"\n✅ Merged JSON saved as: {output_path}")

# 👇 Your actual folder path
folder_path = r"E:\SSD DATA\dfdc_train_all2\all_in_folders"
merge_json_files(folder_path)

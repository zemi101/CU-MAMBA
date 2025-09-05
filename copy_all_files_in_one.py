import os
import shutil
import zipfile
import tarfile
import tempfile
import time
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

MAX_WORKERS = 32  # Use all logical CPU cores

def get_unique_filename(dest_folder, filename):
    base, ext = os.path.splitext(filename)
    counter = 1
    unique_name = filename
    while os.path.exists(os.path.join(dest_folder, unique_name)):
        unique_name = f"{base}_{counter}{ext}"
        counter += 1
    return unique_name

def copy_file_threadsafe(source_path, dest_folder, lock_set):
    filename = os.path.basename(source_path)
    unique_name = get_unique_filename(dest_folder, filename)
    dest_path = os.path.join(dest_folder, unique_name)
    try:
        shutil.copy2(source_path, dest_path)
        lock_set.add(unique_name)
        return f"✅ Copied: {unique_name}"
    except Exception as e:
        return f"❌ Failed to copy {source_path}: {e}"

def extract_zip_file_safe(zip_path, extract_dir):
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            for member in zip_ref.infolist():
                for attempt in range(3):  # Retry 3 times
                    try:
                        zip_ref.extract(member, extract_dir)
                        break
                    except PermissionError:
                        time.sleep(0.5)
                    except Exception as e:
                        if attempt == 2:
                            print(f"⚠️ Failed to extract {member.filename} from {os.path.basename(zip_path)} — Skipping.")
        return True
    except Exception as e:
        print(f"❌ Could not open ZIP: {zip_path} — {e}")
        return False

def extract_tar_file_safe(tar_path, extract_dir):
    try:
        with tarfile.open(tar_path, 'r:*') as tar_ref:
            tar_ref.extractall(extract_dir)
        return True
    except Exception as e:
        print(f"❌ Could not extract TAR: {tar_path} — {e}")
        return False

def extract_and_submit_files(archive_path, dest_folder, executor, lock_set):
    base_temp_dir = tempfile.mkdtemp()
    extracted = False

    try:
        temp_extract_dir = os.path.join(base_temp_dir, os.path.splitext(os.path.basename(archive_path))[0])
        os.makedirs(temp_extract_dir, exist_ok=True)

        if zipfile.is_zipfile(archive_path):
            print(f"📦 Extracting ZIP: {os.path.basename(archive_path)}")
            extracted = extract_zip_file_safe(archive_path, temp_extract_dir)
        elif tarfile.is_tarfile(archive_path):
            print(f"📦 Extracting TAR: {os.path.basename(archive_path)}")
            extracted = extract_tar_file_safe(archive_path, temp_extract_dir)
        else:
            print(f"⚠️ Unsupported archive: {archive_path}")
            return

        if not extracted:
            print(f"❌ Skipping archive due to failure: {archive_path}")
            return

        local_futures = []
        for root, _, files in os.walk(temp_extract_dir):
            for file in files:
                full_path = os.path.join(root, file)
                local_futures.append(executor.submit(copy_file_threadsafe, full_path, dest_folder, lock_set))

        for f in tqdm(as_completed(local_futures), total=len(local_futures),
                      desc=f"📤 Copying extracted: {os.path.basename(archive_path)}", leave=False):
            _ = f.result()

    finally:
        try:
            shutil.rmtree(base_temp_dir, ignore_errors=True)
            print(f"🧹 Cleaned up temp: {base_temp_dir}")
        except Exception as e:
            print(f"⚠️ Temp cleanup failed: {base_temp_dir} — {e}")

def process_files_fast(main_folder, target_folder):
    os.makedirs(target_folder, exist_ok=True)
    lock_set = set()

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        for root, _, files in os.walk(main_folder):
            if os.path.abspath(target_folder) in os.path.abspath(root):
                continue

            for file in files:
                full_path = os.path.join(root, file)

                if zipfile.is_zipfile(full_path) or tarfile.is_tarfile(full_path):
                    extract_and_submit_files(full_path, target_folder, executor, lock_set)
                else:
                    future = executor.submit(copy_file_threadsafe, full_path, target_folder, lock_set)
                    for _ in tqdm(as_completed([future]), total=1, desc=f"📁 Copying: {file}", leave=False):
                        pass

    print("\n✅ All files processed and copied successfully to 'all_in_folders'.")

if __name__ == "__main__":
    source_path = r"E:\SSD DATA\dfdc_train_all2"
    target_path = os.path.join(source_path, "all_in_folders")

    if os.path.isdir(source_path):
        print(f"🚀 Starting ultra-fast copy from: {source_path}")
        process_files_fast(source_path, target_path)
    else:
        print("❌ Invalid folder path.")

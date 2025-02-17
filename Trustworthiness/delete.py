import os
import shutil

def delete_non_answers_files(directory):
    """删除指定目录及子目录中不以 'answers.json' 结尾的文件"""
    for root, _, files in os.walk(directory):
        for file in files:
            if not file.endswith("answers.json"):
                file_path = os.path.join(root, file)
                try:
                    os.remove(file_path)
                    print(f"Deleted: {file_path}")
                except Exception as e:
                    print(f"Failed to delete {file_path}: {e}")

if __name__ == "__main__":
    target_directory = "/path/project/models_output/temporal_awareness/up2date_date_20250101"
    if os.path.exists(target_directory):
        delete_non_answers_files(target_directory)
    else:
        print(f"Directory '{target_directory}' does not exist.")

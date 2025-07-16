"""
File utility functions for resume classifier.
"""

import json
import os
import pickle
import shutil
from pathlib import Path
from typing import List, Optional

from .config import config


class FileUtils:
    """Utility class for file operations."""

    @staticmethod
    def ensure_directory(path: str) -> None:
        """
        Ensure a directory exists, create it if it doesn't.

        Args:
            path: Directory path
        """
        Path(path).mkdir(parents=True, exist_ok=True)

    @staticmethod
    def list_files(directory: str, extensions: Optional[List[str]] = None) -> List[str]:
        """
        List files in a directory with optional extension filtering.

        Args:
            directory: Directory path
            extensions: List of file extensions to include (e.g., ['.pdf', '.docx'])

        Returns:
            List of file paths
        """
        if not os.path.exists(directory):
            return []

        files = []
        for file in os.listdir(directory):
            file_path = os.path.join(directory, file)
            if os.path.isfile(file_path):
                if extensions is None or Path(file).suffix.lower() in extensions:
                    files.append(file_path)

        return sorted(files)

    @staticmethod
    def get_file_size(file_path: str) -> int:
        """
        Get file size in bytes.

        Args:
            file_path: Path to the file

        Returns:
            File size in bytes
        """
        return os.path.getsize(file_path)

    @staticmethod
    def get_file_extension(file_path: str) -> str:
        """
        Get file extension.

        Args:
            file_path: Path to the file

        Returns:
            File extension (including the dot)
        """
        return Path(file_path).suffix.lower()

    @staticmethod
    def get_file_name(file_path: str, with_extension: bool = True) -> str:
        """
        Get file name from path.

        Args:
            file_path: Path to the file
            with_extension: Whether to include the extension

        Returns:
            File name
        """
        path = Path(file_path)
        return path.name if with_extension else path.stem

    @staticmethod
    def copy_file(source: str, destination: str) -> None:
        """
        Copy a file from source to destination.

        Args:
            source: Source file path
            destination: Destination file path
        """
        shutil.copy2(source, destination)

    @staticmethod
    def move_file(source: str, destination: str) -> None:
        """
        Move a file from source to destination.

        Args:
            source: Source file path
            destination: Destination file path
        """
        shutil.move(source, destination)

    @staticmethod
    def delete_file(file_path: str) -> None:
        """
        Delete a file.

        Args:
            file_path: Path to the file to delete
        """
        if os.path.exists(file_path):
            os.remove(file_path)

    @staticmethod
    def delete_directory(directory: str) -> None:
        """
        Delete a directory and all its contents.

        Args:
            directory: Directory path to delete
        """
        if os.path.exists(directory):
            shutil.rmtree(directory)

    @staticmethod
    def save_json(data: dict, file_path: str, indent: int = 2) -> None:
        """
        Save data to a JSON file.

        Args:
            data: Data to save
            file_path: Path to save the JSON file
            indent: JSON indentation
        """
        FileUtils.ensure_directory(os.path.dirname(file_path))
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=indent, ensure_ascii=False)

    @staticmethod
    def load_json(file_path: str) -> dict:
        """
        Load data from a JSON file.

        Args:
            file_path: Path to the JSON file

        Returns:
            Loaded data
        """
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)

    @staticmethod
    def save_pickle(data: object, file_path: str) -> None:
        """
        Save data to a pickle file.

        Args:
            data: Data to save
            file_path: Path to save the pickle file
        """
        FileUtils.ensure_directory(os.path.dirname(file_path))
        with open(file_path, "wb") as f:
            pickle.dump(data, f)

    @staticmethod
    def load_pickle(file_path: str) -> object:
        """
        Load data from a pickle file.

        Args:
            file_path: Path to the pickle file

        Returns:
            Loaded data
        """
        with open(file_path, "rb") as f:
            return pickle.load(f)

    @staticmethod
    def get_directory_size(directory: str) -> int:
        """
        Get total size of a directory in bytes.

        Args:
            directory: Directory path

        Returns:
            Total size in bytes
        """
        total_size = 0
        for dirpath, dirnames, filenames in os.walk(directory):
            for filename in filenames:
                file_path = os.path.join(dirpath, filename)
                total_size += os.path.getsize(file_path)
        return total_size

    @staticmethod
    def format_file_size(size_bytes: int) -> str:
        """
        Format file size in human-readable format.

        Args:
            size_bytes: Size in bytes

        Returns:
            Formatted size string
        """
        if size_bytes == 0:
            return "0B"

        size_names = ["B", "KB", "MB", "GB", "TB"]
        i = 0
        while size_bytes >= 1024 and i < len(size_names) - 1:
            size_bytes /= 1024.0
            i += 1

        return f"{size_bytes:.1f}{size_names[i]}"

    @staticmethod
    def find_files_by_pattern(directory: str, pattern: str) -> List[str]:
        """
        Find files matching a pattern in a directory.

        Args:
            directory: Directory to search
            pattern: File pattern (e.g., "*.pdf", "resume_*")

        Returns:
            List of matching file paths
        """
        import glob

        pattern_path = os.path.join(directory, pattern)
        return glob.glob(pattern_path)

    @staticmethod
    def create_backup(file_path: str, backup_suffix: str = ".backup") -> str:
        """
        Create a backup of a file.

        Args:
            file_path: Path to the file to backup
            backup_suffix: Suffix for the backup file

        Returns:
            Path to the backup file
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        backup_path = file_path + backup_suffix
        FileUtils.copy_file(file_path, backup_path)
        return backup_path

    @staticmethod
    def restore_backup(backup_path: str, original_suffix: str = ".backup") -> str:
        """
        Restore a file from its backup.

        Args:
            backup_path: Path to the backup file
            original_suffix: Suffix of the backup file

        Returns:
            Path to the restored file
        """
        if not backup_path.endswith(original_suffix):
            raise ValueError(f"Backup file must end with {original_suffix}")

        original_path = backup_path[: -len(original_suffix)]
        FileUtils.copy_file(backup_path, original_path)
        return original_path

    @staticmethod
    def clean_filename(filename: str) -> str:
        """
        Clean a filename by removing invalid characters.

        Args:
            filename: Original filename

        Returns:
            Cleaned filename
        """
        import re

        # Remove invalid characters
        cleaned = re.sub(r'[<>:"/\\|?*]', "_", filename)
        # Remove leading/trailing spaces and dots
        cleaned = cleaned.strip(". ")
        # Limit length
        if len(cleaned) > 255:
            name, ext = os.path.splitext(cleaned)
            cleaned = name[: 255 - len(ext)] + ext
        return cleaned

    @staticmethod
    def get_file_info(file_path: str) -> dict:
        """
        Get comprehensive file information.

        Args:
            file_path: Path to the file

        Returns:
            Dictionary with file information
        """
        if not os.path.exists(file_path):
            return {"error": "File not found"}

        stat = os.stat(file_path)
        path = Path(file_path)

        return {
            "name": path.name,
            "stem": path.stem,
            "suffix": path.suffix,
            "size": stat.st_size,
            "size_formatted": FileUtils.format_file_size(stat.st_size),
            "created": stat.st_ctime,
            "modified": stat.st_mtime,
            "accessed": stat.st_atime,
            "is_file": path.is_file(),
            "is_dir": path.is_dir(),
            "exists": path.exists(),
        }

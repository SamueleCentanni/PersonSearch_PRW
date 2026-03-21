from __future__ import annotations

from typing import Iterable, List, Optional
import os

import scipy.io


def inspect_folder(dataset_root: str, folder: str, index: int) -> Optional[str]:
    """
    Return the sorted absolute path of the file at the specified index within a
    dataset subfolder. Returns None if the folder doesn't exist or the index is
    out of bounds.
    """
    folder_path = os.path.join(dataset_root, folder)
    print(f"Analysing folder: {folder_path}")

    file_list = sorted(os.listdir(folder_path))
    if index < len(file_list):
        file_x = file_list[index]
        file_x_path = os.path.join(folder_path, file_x)
        print(f"We are considering the file {file_x_path}\n")
        return file_x_path

    print("File does not exist. Modify the index!")
    return None


def show_file_keys(files: Iterable[str]) -> None:
    """Load .mat files and print their metadata keys for structure inspection."""
    for file in files:
        if os.path.isfile(file):
            file_mat = scipy.io.loadmat(file)
            print(f"File name: {os.path.basename(file)}\nKeys: {file_mat.keys()}\n")
        else:
            print(f"File {file} doesn't exists.")
            break


def show_file_content(files: Iterable[str]) -> None:
    """Extract and print the first three entries of the primary data key in .mat files."""
    for file in files:
        if os.path.isfile(file):
            file_mat = scipy.io.loadmat(file)
            key = [k for k in file_mat.keys() if not k.startswith("__")][0]
            print(f"File name: {os.path.basename(file)}\nContent: {file_mat[key][:3]}\n")
        else:
            print(f"File {file} doesn't exists.")
            break

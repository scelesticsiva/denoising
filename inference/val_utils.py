import glob
import os.path
import re

def filter_val_files(dir, val_files, ext):
    get_fname = lambda x: re.split(r'p(\d{1,3})', os.path.basename(x))[0]
    all_files = glob.glob(dir + ext)
    fname_to_fpth = {}
    for fpth in all_files:
        fname = get_fname(fpth)
        if fname not in fname_to_fpth:
            fname_to_fpth[fname] = []
        fname_to_fpth[fname].append(fpth)

    filtered_files = []
    for fname in val_files:
        filtered_files += fname_to_fpth[fname]

    return filtered_files
import os
from os.path import normpath, basename

DATADIR = r"C:\Users\Psy\Downloads\Data\Imagens"
count = 0

for subdir, dirs, files in os.walk(DATADIR):
	for file in files:
		print(basename(normpath(subdir)))
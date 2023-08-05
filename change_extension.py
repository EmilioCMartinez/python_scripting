#for this code, I had multiple images where I needed to change the extensions. Rather than do it manually, 
#I created this script to loop through all the images in my folder and change them to the extensions I needed.
import os 

#this is where the folder is located where the pictures of my crash are
path = '/Users/emiliomartinez/Downloads/Crash'

#initialize the count of files that will be modified
count = 0

#Change all the extensions of the pictures in the folder from '.HEIC' to .'jpeg'
for filename in os.listdir(path):
    if filename.endswith('.HEIC'):
        #is a slice of the filename that includes all characters up to, but not including, the last five characters.
        new_filename = filename[:-5] + '.jpeg'
        os.rename(os.path.join(path, filename), os.path.join(path, new_filename))
        count += 1
#count the number of files that were modfied
print(f"{count} files were renamed.")

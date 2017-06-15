#fileFuncs.py

import os
from glob import glob

def find(path, pattern):
	s = join(path, pattern)
	r = glob(s)
	return r

def findDir(a_dir):
	t = [os.path.join(a_dir, name) for name in os.listdir(a_dir) 
		if os.path.isdir(os.path.join(a_dir, name))]


	return sorted(t)
	
def findFiles(a_dir, suf='', sub=''):
        t = [os.path.join(a_dir, name) for name in os.listdir(a_dir)
             if os.path.isfile(os.path.join(a_dir, name)) 
             and name.endswith(suf) and sub in name]
        return sorted(t)

def isFile(file):
	return os.path.isfile(file)

def isDir(path):
	return os.path.isdir(path)

# Recursive, exhastive walk for target files. Returns all unique paths
def _sniff(start, target):
	# there is stuff here! Are any of them the target?

	files = findFiles(start, sub=target)
	# are there any there paths we can take?
	dirs = findDir(start)
	if len(dirs)==0:
		# hit a dead end, return the bin
		return files
	else:
		# there is stuff here! Do any of them contain a target?
		for d in dirs:
			matches = _sniff(d, target)
			files = files + matches
		return files
	
def sniff(start, target):

	if not os.path.isdir(start):
		raise ValueError("The starting point must be a valid directory!")
	return _sniff(start, target)

# Returns each section of the file split via file separator
def parse(fileName):
	return fileName.split(os.sep)

def readFile(filepath):
	file = open(filepath)
	data = [];
	for line in list(file):
		data.append(line.split())
	file.close()
	return data

def writeFile(filepath, frmt, data):
	file = open(filepath, 'w')
	ss = [frmt.format(*tuple(d)) for d in data]
	
	for s in ss:
		file.write(s)

	file.close()
	return True 

def getcwd():
	return os.getcwd()

def chdir(d):
	return os.chdir(d)
def absPath(a):
	return os.path.abspath(a)

def join(a,b):
	return a + os.sep + b

def fs():
	return os.pathsep

# <dir/file.ext> -> file.ext
def fileName(file):
	return os.path.split(file)[1]

# <dir/file.ext> -> file
def fileBase(file):
	return filePath(fileName(file))

# <dir/file.ext> -> dir/file
def filePath(file):
	return os.path.splitext(file)[0]

# <dir/file.ext> -> ext
def fileExt(file):
	return os.path.splitext(fileName(file))[1]

def appendSuffix(file, extra):
	return filePath(file) + extra + fileExt(file)
	
def swapDir(file, path):
    return join(path, fileName(file))

def swapExt(file, ext):
	if not "." in ext:
		ext = "." + ext 
	return filePath(file) + ext

def fileDir(file):
	return os.path.split(file)[0]

def sameDir(a,b):
	return join(fileDir(a), b)

def dirAscend(directory):
	return os.path.split(directory)[0]

def dirDescend(directory):
	return os.path.split(directory)[1]


def step(d, s):
	if s < 0:
		d = step(dirAscend(d), s + 1)
	elif s > 0:
		d = step(dirDescend(d), s - 1)
	else:
		if not d:
			d = ''
	return d

def dirJoin(d1, d2):
	return os.path.join(d1,d2)

def isDir(d):
	return os.path.isdir(d)

def ensureDir(d):
	d = os.path.realpath(d)
	if isDir(d):
		return True
	else:
		os.mkdir(d)
		return False
def deleteFile(f):
	try:
		os.remove(f)
		return True
	except:
		print("Could not remove file: " + f)
		return False

def isEmpty(d):
	return isDir(d) and not findFiles(d)

def deleteFolder(d):
	for f in findFiles(d, '', ''):
		os.remove(f)
	for di in findDir(d):
		deleteFolder(di)
	os.rmdir(d)
	return True 

def cleanDir(d):
	d = os.path.realpath(d)
	if isDir(d):
		deleteFolder(d)
	os.mkdir(d)
	


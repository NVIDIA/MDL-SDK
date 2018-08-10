#!/bin/env python

import os 
import os.path

def process(fname):
  f = file(fname, 'rb')
  c = f.read()
  f.close()
  n = c.replace('std::', 'MISTD::').replace('namespace std', 'namespace MISTD')
  if n != c:
    print fname
    f = file(fname, 'wb')
    f.write(n)
    f.close()

def process_dirs(path, ignore_list = []):
  ignore_list += ['.git', '.gitignore', '.svn', '.svnignore', 'test', 'change_namespace.py']
  ignore_list = set(ignore_list)
  ignore_dirs = [os.path.join(path, i) for i in ignore_list]

  for dirpath, dirnames, filenames in os.walk(path):
    for ignore in ignore_dirs:
      l = len(ignore)
      if dirpath[0:l] == ignore:
        # ignore
        break
    else:
      for filename in filenames:
        fullname = os.path.join(dirpath, filename)
	process(fullname)

if __name__ == "__main__":
  process_dirs('.', ['doc'])

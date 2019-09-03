# -*- coding: utf-8 -*-
"""
Created on Thu Nov  9 10:48:33 2017

@author: nivedita
"""

#!/usr/bin/env python
"""\
convert dos linefeeds (crlf) to unix (lf)
usage: dos2unix.py <input> <output>
"""

### Note: For Python 3, pickle.load gives an error. Followed suggestion from this link:
### https://github.com/udacity/ud120-projects/issues/46
original = "final_project_dataset.pkl"
destination = "final_project_dataset_unix.pkl"

content = ''
outsize = 0
with open(original, 'rb') as infile:
  content = infile.read()
with open(destination, 'wb') as output:
  for line in content.splitlines():
    outsize += len(line) + 1
    output.write(line + str.encode('\n'))

print("Done. Saved %s bytes." % (len(content)-outsize))
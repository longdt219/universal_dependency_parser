#!/usr/bin/env python

import sys
import re
import itertools
import sqlite3
import argparse

parser = argparse.ArgumentParser(description='Parameters for dictionary extraction')
parser.add_argument('-s','--source', help='Source language (3 char)', required=True)
parser.add_argument('-t','--target', help='Target language (3 char)', required=True)
parser.add_argument('-i','--db', help='database (sqlite) file ', required=True)
parser.add_argument('-o','--out', help='out file', required=True)

args = vars(parser.parse_args())
# Args is the dictionary containing the argument
source = args['source']
target = args['target']
out_file = args['out']
db_file = args['db']

conn = sqlite3.connect(db_file)
c = conn.cursor()

uid_lv = {}
lv_uid = {}
lv_lc = {}

for row in c.execute('SELECT uid, lv, lc FROM lv'):
    uid_lv[row[0]] = row[1]
    lv_uid[row[1]] = row[0]
    lv_lc[row[1]] = row[2]
    print row
def lc_to_uid(lc):
    if re.search('-', lc):
        return lc
    else:
        return lc + '-000'
    
# Get the language variety mapping  
lv_source = uid_lv[lc_to_uid(source)]
lv_target = uid_lv[lc_to_uid(target)]
print lv_source  # 187 (english) 
print lv_target   # 157 (german)


lookup_q = """
SELECT ex.tt,ex2.tt
FROM mn
JOIN ex ON (ex.ex = mn.ex)
JOIN mn mn2 ON (mn2.mn = mn.mn)
JOIN ex ex2 ON (ex2.ex = mn2.ex)
WHERE mn.ex != mn2.ex AND ex.lv = ? AND ex2.lv = ?
"""
fout = open(out_file,'wb')
#query_str = 'select ex.tt from ex where ex.lv = "eng-000"'
for row in c.execute(lookup_q, (lv_source,lv_target)):
    fout.write(row[0].encode('utf8') + '\t' + row[1].encode('utf8') + '\n')
    
fout.close()

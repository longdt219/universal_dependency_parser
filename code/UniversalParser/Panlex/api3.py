#!/usr/bin/env python
"""
I've now implemented a basic version of the lookup() function proposed in your API 2 (see attachment). 
The code creates a dictionary whose keys are concept (translation set) IDs and 
whose values are a dictionary containing translations. 
The translation dictionary's keys are language codes (e.g. "eng") and language variety UIDs (e.g. "eng-000") 
and the values are a list tuples with the translated expression text and translation quality.

Here is some example output. The code currently prints each concept ID followed by the translation dictionary object.

kamholz@db4:~/panlex-nltk$ ./api2.py lookup pterodactyl eng

9853516
{u'kor-000': [(u'\uc775\uc218\ub8e1', 3)], u'kor': [(u'\uc775\uc218\ub8e1', 3)]}

7048482
{u'fra': [(u'pt\xe9rodactyle', 6)], u'ind-000': [(u'pterodaktil', 6)], u'ita-000': [(u'pterodattilo', 6)], u'por': [(u'pterod\xe1ctilo', 6)], u'fra-000': [(u'pt\xe9rodactyle', 6)], u'ita': [(u'pterodattilo', 6)], u'por-000': [(u'pterod\xe1ctilo', 6)], u'ind': [(u'pterodaktil', 6)]}

""" 



import sys
import re
import itertools
import sqlite3

conn = sqlite3.connect('db3.sqlite')
c = conn.cursor()

translate_q = """
SELECT ex2.tt, sum(tr2.trq)
FROM tr
JOIN ex ON (ex.ex = tr.ex)
JOIN tr tr2 ON (tr2.tr = tr.tr)
JOIN ex ex2 ON (ex2.ex = tr2.ex)
WHERE ex.tt = ? AND ex.lv = ? AND ex2.lv = ?
GROUP BY ex2.tt
ORDER BY sum(tr2.trq) DESC
"""

expressions_q = """
SELECT ex.tt 
FROM ex 
WHERE ex.lv = ? 
ORDER BY ex.tt
"""

lookup_q = """
SELECT mn2.mn, mn2.trq, mn2.ap, ex2.tt, ex2.lv
FROM mn
JOIN ex ON (ex.ex = mn.ex)
JOIN mn mn2 ON (mn2.mn = mn.mn)
JOIN ex ex2 ON (ex2.ex = mn2.ex)
WHERE mn.ex != mn2.ex AND ex.tt = ? AND ex.lv = ?
ORDER BY mn2.trq DESC
"""

uid_lv = {}
lv_uid = {}
lv_lc = {}

for row in c.execute('SELECT uid, lv, lc FROM lv'):
    uid_lv[row[0]] = row[1]
    lv_uid[row[1]] = row[0]
    lv_lc[row[1]] = row[2]

def lc_to_uid(lc):
    if re.search('-', lc):
        return lc
    else:
        return lc + '-000'

def lookup(tt, uid):
    # Look up term in the language id 
    lv = uid_lv[lc_to_uid(uid)]
    # language variety i.e. eng-000
    concepts = {}

    # Lookup a term in a specific language variety 
    for i in c.execute(lookup_q, (tt,lv)):
        mn = i[0]
        uid = lv_uid[i[4]]

        if not mn in concepts:
            concepts[mn] = { 'quality': i[1], 'ap': i[2] }

        if not uid in concepts[mn]:
            concepts[mn][uid] = []

        concepts[mn][uid].append(i[3])

    for i in concepts.keys():
        print i
        print repr(concepts[i])
        print

cmd = sys.argv[1]

if cmd == 'lookup':
    lookup(sys.argv[2], sys.argv[3])
else:
    print "unknown command: {}".format(cmd)

print
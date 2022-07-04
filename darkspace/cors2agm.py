#!/usr/bin/env python3

# This script parses the output of the cors2ascii tool and
# generates a csv with flow data using the AGM key.
# AH, Nov.2020

import re
import sys
from collections import deque, defaultdict

flow_table = {}
times = deque()

flow_timeout = 3600

column_names = ('dst_ip', 'src_port', 'dst_port', 'proto', 'ttl', 'tcp_flags', 'ip_len')

def output_entry(t,src_ip):
    flow_table_entry = flow_table[src_ip]
    pkt_totalcount = sum(flow_table_entry[0].values())
    out = '%d,%s,%d' % (t,src_ip, pkt_totalcount)
    for entry in flow_table_entry:
        most_frequent = max(entry, key=lambda val: entry[val])
        out += ',%d,%s,%d' % (len(entry), most_frequent, entry[most_frequent])
    print (out)

print ('flowStart,src_ip,packetTotalCount,' + ','.join('distinct(%s),mode(%s),modeCount(%s)' % (feat,feat,feat) for feat in column_names))

t = None
what = None
for line in sys.stdin:
    if line.startswith('# CORSARO_INTERVAL_END'):
        t = None
        what = None
    elif line.startswith('# CORSARO_INTERVAL_START'):
        match = re.match('# CORSARO_INTERVAL_START ([0-9]+) ([0-9]+)', line)
        assert match is not None
        t = int(match.group(2))
        print (t, file=sys.stderr)
        assert not times or times[-1][0] <= t
        while times and times[0][0] <= t - flow_timeout:
            start_time, src_ip = times.popleft()
            output_entry(start_time,src_ip)
            del flow_table[src_ip]
    elif line.startswith('START flowtuple_'):
        match = re.match('START flowtuple_(.*) ([0-9]+)', line)
        assert match is not None
        what = match.group(1)
    elif line.startswith('END flowtuple_'):
        what = None
    else:
        assert t is not None and what is not None
        key,cnt = line.split(',')
        cnt = int(cnt)
        keys = key.split('|')
        src_ip = keys[0]
        entry = flow_table.get(src_ip)
        if entry is None:
            times.append((t,src_ip))
            def dd_factory():
                return 0
            entry = tuple( defaultdict(dd_factory) for _ in range(7) )
            flow_table[src_ip] = entry
        for table_entry, key in zip(entry, keys[1:]):
            table_entry[key] += cnt

#!/usr/bin/env python3

import pickle
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import subprocess
from multiprocessing import Pool
from datetime import datetime

d = pd.read_csv('capture.csv')
results = pickle.load(open('results.pickle', 'rb'))
obs = results['obs']
params = results['params']

obs_indices = obs[0][:,-1].astype(int)

# start tshark to extract PCAPs for each observer in the pcaps directory 
os.makedirs('pcaps/logs', exist_ok=True)
cmds = []
for i,o in enumerate(obs_indices[:200]):
    data = d.iloc[o,:]
    time_filter = 'frame.time_epoch > %.3f and frame.time_epoch < %.3f' % (data['flowStartMilliseconds']/1000 - 5*60, data['flowStartMilliseconds']/1000+data['flowDurationMilliseconds']/1000 + 5*60)
    conv_filter = 'ip.addr eq %s and ip.addr eq %s' % (data['sourceIPAddress'], data['destinationIPAddress'])
    if data['protocolIdentifier'] == 6:
        port_filter = ' and (tcp.port eq %d and tcp.port eq %d)' % (data['sourceTransportPort'], data['destinationTransportPort'])
    elif data['protocolIdentifier'] == 17:
        port_filter = ' and (udp.port eq %d and udp.port eq %d)' % (data['sourceTransportPort'], data['destinationTransportPort'])
    else:
        port_filter = ''
    cmd = "tshark -r capture.pcap -w pcaps/%d.pcap -M 200000 '(%s) and (%s)%s'" % (i, time_filter, conv_filter, port_filter)
    print (cmd)
    cmds.append(cmd)

def start_tshark(i):
    with open('pcaps/logs/%d' % i, 'wb') as out:
        subprocess.run(cmds[i], shell=True, stdout=out, stderr=out)
    return 0
with Pool(200) as p:
    p.map(start_tshark, range(len(cmds)))

#!/usr/bin/env python3

import pandas as pd
import numpy as np
from sklearn.preprocessing import scale, minmax_scale

column_names = ['duration', 'protocol_type', 'service', 'flag', 'src_bytes',
    'dst_bytes', 'land', 'wrong_fragment', 'urgent', 'hot',
    'num_failed_logins', 'logged_in', 'num_compromised', 'root_shell',
    'su_attempted', 'num_root', 'num_file_creations', 'num_shells',
    'num_access_files', 'num_outbound_cmds', 'is_host_login',
    'is_guest_login', 'count', 'srv_count', 'serror_rate',
    'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate', 'same_srv_rate',
    'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count',
    'dst_host_srv_count', 'dst_host_same_srv_rate',
    'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',
    'dst_host_srv_diff_host_rate', 'dst_host_serror_rate',
    'dst_host_srv_serror_rate', 'dst_host_rerror_rate',
    'dst_host_srv_rerror_rate', 'label']
u2r = ['buffer_overflow.', 'ftp_write.', 'imap.', 'load_module.',
    'multihop.', 'nmap.', 'perl.', 'phf.', 'pod.', 'rootkit.', 'teardrop.']

df = pd.read_csv('kddcup.data', names=column_names)
del df['service']
df = df[df['label'].isin(['normal.'] + u2r)]
nominal = ['protocol_type', 'flag']
non_nominal = [ col for col in df.columns if col not in nominal and col != 'label' ]
X = np.concatenate( [ pd.get_dummies(df[col]).values for col in nominal ] + [ scale(df[non_nominal].values) ], axis=1)
y = (df['label'] != 'normal.').values
np.savez('kddcup.npz', X=X, y=y, times=np.arange(X.shape[0]))

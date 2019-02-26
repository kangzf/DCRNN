import pandas as pd
import numpy as np
import os, sys

inter_min = 15
inter = '{}min'.format(inter_min)
DATA_PATH = 'langfang_data/'
NAME = DATA_PATH + inter + '_data.h5'
NUM_NODES = 65 if inter_min == 10 else 94
NUM_MONTH = 4
NUM_DIRECT = 4
CLEAN_THRES = 0.25


time_stamp = pd.date_range("2014-01-01 00:00","2014-0{}-30 23:59".format(NUM_MONTH),freq=inter)
TOTAL_TIME = time_stamp.size
flow = pd.DataFrame(0, columns=np.arange(1, NUM_NODES+1), index=time_stamp) # 2-D

listd = os.listdir(DATA_PATH+inter)
# example = pd.read_csv(os.path.join(DATA_PATH+inter, listd[0]),
#                            sep=",", header=None)

for each in listd:
    if 'F' in each:
        flow_node = pd.read_csv(os.path.join(DATA_PATH+inter, each),
                           sep=",", header=None)
        id_node, id_direct = each.split('_')
        id_node = int(id_node[2:])
        flow_node = flow_node.iloc[:,5:]
        assert flow_node.shape[1] == 24*60/inter_min
        flow_node = flow_node.values.flatten()
        flow_node = flow_node[:TOTAL_TIME].transpose()

        # Add to flow data
        flow[id_node] += flow_node

# Data cleaning
# for id in flow.columns:
#     col = flow.loc[:, id]
# flow = flow.loc[:, (flow == 0).any(axis=0)]
drop_cols = flow.columns[(flow == 0).sum() > CLEAN_THRES*flow.shape[1]]
flow.drop(drop_cols, axis = 1, inplace = True)
flow /= NUM_DIRECT


hdf = pd.HDFStore(NAME)
hdf.put('data', flow, format='t')
hdf.close()

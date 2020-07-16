"""
  Created by PyCharm.
  User: antonvasilev <bysslaev@gmail.com>
  Date: 05/02/2020
  Time: 01:03
 """

from pandas import DataFrame, Series
import pandas as pd
import json
#
# path = 'usagov_bitly_data2012-03-16-1331923249.txt'
# records = [json.loads(line) for line in open(path)]
# # time_zones = [rec['tz'] for rec in records if 'tz' in rec]
# # print(time_zones)
# frame = DataFrame(records)
# print(frame['tz'][:10])
#
# tz_counts = frame['tz'].value_counts()
#
# print(tz_counts[:10])
#
# frame['nk'].plot.pie()


from src.data_manager.rflab_np_manager import RflabNpDataManager

#%%

rflab_manager = RflabNpDataManager()
dataset = rflab_manager.load()
#%%
print(dataset.shape)
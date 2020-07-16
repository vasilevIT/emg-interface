"""
  Created by PyCharm.
  User: antonvasilev <bysslaev@gmail.com>
  Date: 09/01/2020
  Time: 23:45
 """
from src.data_manager.rflab_np_manager import RflabNpDataManager
from matplotlib import pyplot

rflab_manager = RflabNpDataManager()
dataset = rflab_manager.load()
pyplot.show(block=True)
pyplot.plot([1,2,3])


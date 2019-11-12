"""
  Created by PyCharm.
  User: antonvasilev <bysslaev@gmail.com>
  Date: 12/11/2019
  Time: 22:33
 """
import os


def get_root_path():
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))


def get_path(local_path):
    return get_root_path() + local_path

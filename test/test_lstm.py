"""
  Created by PyCharm.
  User: antonvasilev <bysslaev@gmail.com>
  Date: 12/11/2019
  Time: 21:45
 """

import unittest
import mock
from src.models.lstm import LstmModel


class TestLstmModel(unittest.TestCase):

    def test_number_layers(self):
        layers = 10
        model = LstmModel()
        model.set_layers(layers)
        model.build()

        self.assertEqual(len(model.get_layers()), layers * 2 + 5)


if __name__ == '__main__':
    unittest.main()

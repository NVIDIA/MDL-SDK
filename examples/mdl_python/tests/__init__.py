import unittest
import os

# run all tests 
if __name__ == '__main__':
    loader = unittest.TestLoader()
    suite = loader.discover(os.path.dirname(os.path.abspath(__file__)))
    runner = unittest.TextTestRunner()
    runner.run(suite)
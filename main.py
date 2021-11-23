from visio.example import best_photos
from visio.video import rvm_test
from visio.slide import test_ppt_class


if __name__ == '__main__':
    import sys
    path = sys.argv[1]
    test_ppt_class(path)
    # rvm_test()

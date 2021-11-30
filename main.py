from visio.example import best_photos
from visio.video import rvm_test
from visio.slide import test_ppt_class, test_layered_video
from visio.utils import test_cuda_cmap


if __name__ == '__main__':
    import sys
    path = sys.argv[1]
    # test_ppt_class(path)
    # test_layered_video()
    # rvm_test()
    test_cuda_cmap()

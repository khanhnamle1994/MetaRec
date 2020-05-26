import os
import sys
import time
import logging


class Logger:
    def __init__(self, log_dir):
        self.logger = logging.getLogger('RecSys')
        self.logger.setLevel(logging.INFO)

        # File handler
        self.log_dir = self.get_log_dir(log_dir)
        fh = logging.FileHandler(os.path.join(self.log_dir, 'log.txt'))
        fh.setLevel(logging.DEBUG)
        fh_format = logging.Formatter('%(asctime)s: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
        fh.setFormatter(fh_format)
        self.logger.addHandler(fh)

        # Console handler
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(logging.INFO)
        ch_format = logging.Formatter('%(message)s')
        ch.setFormatter(ch_format)
        self.logger.addHandler(ch)

    def info(self, msg):
        self.logger.info(msg)

    def get_log_dir(self, log_dir):
        if not os.path.exists(log_dir):
            os.mkdir(log_dir)

        log_dirs = os.listdir(log_dir)
        if len(log_dirs) == 0:
            idx = 0
        else:
            idx_list = sorted([int(d.split('_')[0]) for d in log_dirs])
            idx = idx_list[-1] + 1

        cur_log_dir = '%d_%s' % (idx, time.strftime('%Y%m%d-%H%M'))
        full_log_dir = os.path.join(log_dir, cur_log_dir)
        if not os.path.exists(full_log_dir):
            os.mkdir(full_log_dir)

        return full_log_dir
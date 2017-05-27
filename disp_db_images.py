#!/usr/bin/python

# This is a simple tool to load the images in lmdb/leveldb dataset and display
#
# In addition, it can analyze the patch according to their complexity
# measurement.
# In this case, we use the sigma(abs(val-mean)) to measure to complexity

import getopt
import sys
import tools
import glog as log
import lmdb_lib
import caffe_tools
import cPickle as pickle
import serialize_lib
import easyprogressbar as eb
import imshow_lib
import leveldb_lib
import numpy as np


def CalcCplx(patch):
    """
    This function compute the patch complexity
    """
    mean = patch.mean(0).mean(0)
    tmp = np.zeros(patch.shape)
    for i in range(patch.shape[2]):
        tmp[:, :, i] = patch[:, :, i] - mean[i]
    tmp = np.abs(tmp)
    return tmp.mean()


def main(argv):
    I = log.info
    in_file = None

    backend = 'ser'
    db_type = 'lmdb'

    skip_num = 0

    help_msg = 'disp_db_image -i <lmdb> -s [int]\n\
-i <lmdb>           The lmdb contains the datum images\n\
-s [int]            The number of skip images\n\
--backend [str]     The backend of serialize model: ser, pickle, datum\
  default: %s\n\
--db [str]          The db type: lmdb, leveldb, \
the default: %s' % (backend, db_type)

    try:
        opts, args = getopt.getopt(argv, 'hi:s:',
                                   ['db=', 'backend='])
    except getopt.GetoptError:
        print help_msg
        sys.exit(2)

    for opt, arg in opts:
        if opt == '-h':
            print help_msg
            sys.exit()
        elif opt == '-i':
            in_file = arg
        elif opt == '-s':
            skip_num = int(arg)
        elif opt == '--db':
            db_type = arg
        elif opt == '--backend':
            backend = arg

    if in_file is None:
        print help_msg
        sys.exit()

    ser = serialize_lib.serialize_numpy()
    if backend.lower() == 'ser':
        parse_func = ser.loads
    elif backend.lower() == 'pickle':
        parse_func = pickle.loads
    elif backend.lower() == 'datum':
        parse_func = caffe_tools.datum_str_to_array_im
    else:
        log.error('The unknown backend type: %s' % backend)
        sys.exit()

    if db_type.lower() == 'lmdb':
        db_creater = lmdb_lib.lmdb
    elif db_type.lower() == 'leveldb':
        db_creater = leveldb_lib.leveldb

    db = db_creater(in_file)
    db.set_val_parser(parse_func)

    ish = imshow_lib.Imshow()

    bar = eb.EasyProgressBar()
    bar.set_end(db.get_entries())
    I('Start to iter the images')
    bar.start()

    counter = 0

    for key, val in db:
        bar.update_once()
        counter += 1
        if counter < skip_num:
            continue

        ish.imshow(val)
        ch = tools.get_char('Press Q to quit, other to continue')
        if ch.lower() == 'q':
            break

    bar.finish()


if __name__ == '__main__':
    main(sys.argv[1:])

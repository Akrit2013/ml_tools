#!/usr/bin/python

# This script can generate the a lmdb database according to the
# input of the datalist
# The data inside the datalist can be mat of jpg images
#
# The image or mat data will be directly store into the lmdb
# without modification
#
# The data can be stored in datum of pickle format


import time
import getopt
import sys
import glog as log
import txt_tools
import lmdb_thread_lib
import easyprogressbar
import path_tools
import caffe_tools
import numpy as np
import matlab_tools
import image_tools
import Queue
import cPickle as pickle
import crash_on_ipy


def main(argv):
    # Define the var names
    var_name = None

    I = log.info
    in_file = None
    out_file = None
    resize_width = None
    resize_height = None
    min_val = None
    max_val = None

    ext_path = None
    is_log = False

    backend = 'pickle'

    help_msg = 'generate_single_lmdb_from_datalist.py -i <txt> -o <lmdb_prefix> \
-v <varname>\n\
-i <txt>                The datalist contains the mat data index.\n\
-o <lmdb_prefix>        The prefix name of the data and label lmdb file.\n\
-v [varname]            The name of the target var, such as data or depfill\n\
-p [path]               If set, the script will search the target path \
instead of the path in assigned in the datalist.\n\
--resize_height [num]   Resize the height of the image and label.\n\
--resize_width [num]    Resize the width of the image and label.\n\
--min_val [float]       If pixel smaller than the min_val, set it to min_val\n\
--max_val [float]       If pixel larger than the min_val, set it to max_val.\n\
--log                   If set, convert the data into log space before write.\
into the lmdb.\n\
--backend [str]         Default is pickle, but it can be set as datum.'

    try:
        opts, args = getopt.getopt(argv, 'hi:o:v:p:',
                                   ['resize_width=', 'resize_height=',
                                    'min_val=', 'max_val=', 'log',
                                    'backend='])
    except getopt.GetoptError:
        print help_msg
        sys.exit(2)

    for opt, arg in opts:
        if opt == '-h':
            print help_msg
            sys.exit()
        elif opt == '-i':
            in_file = arg
        elif opt == '-o':
            out_file = arg
        elif opt == '-v':
            var_name = arg
        elif opt == '--resize_height':
            resize_height = int(arg)
        elif opt == '--resize_width':
            resize_width = int(arg)
        elif opt == '-p':
            ext_path = arg
        elif opt == '--min_val':
            min_val = float(arg)
        elif opt == '--max_val':
            max_val = float(arg)
        elif opt == '--log':
            is_log = True
        elif opt == '--backend':
            backend = arg

    if in_file is None or out_file is None:
        print help_msg
        sys.exit()

    if is_log is True and min_val is None:
        log.warn('\033[33mWARNING\033[0m: The when using the log space, it \
should be better to set min_val to avoid the negative value')

    # Select the backend
    if backend.lower() == 'pickle':
        backend_func = pickle.dumps
    elif backend.lower() == 'datum':
        backend_func = caffe_tools.load_array_to_datum_str
    else:
        log.error('ERROR: Can not recognize the backend: %s' % backend)
        sys.exit()

    # Load the datalist
    datalist_list = txt_tools.read_lines_from_txtfile(in_file)
    I('Datalist contains %d entries' % len(datalist_list))

    # Init the lmdb writer
    I('Init the LMDB writer')
    DBThread = lmdb_thread_lib.DbWriteThread(out_file)
    DBThread.start()

    bar = easyprogressbar.EasyProgressBar()
    bar.set_end(len(datalist_list))
    I('Start the process, total samples: \033[01;31m%d\033[0m'
      % len(datalist_list))

    counter = 0
    key_queue = Queue.Queue()
    bar.start()
    while True:
        if counter < len(datalist_list):
            # Load the mat file
            im_path = datalist_list[counter]
            # Delete the end \n of the im_path
            if im_path[-1] == '\n':
                im_path = im_path[:-1]
                # Replace the path if needed
            if ext_path is not None:
                base_name = path_tools.get_base_name(im_path)
                im_path = path_tools.get_full_path(ext_path, base_name)
            rst_list = matlab_tools.load_mat(im_path, var_name)
            data = rst_list
            # replace the NaN hole in the dep to 0
            data = image_tools.remove_nan(data)
            # Resize the rgb and dep if needed
            if resize_height is not None and resize_width is not None:
                data = image_tools.imresize(data, resize_height, resize_width)
            # Replace all the near 0 to 0
            data[np.isclose(data, 0)] = 0
            # All pixel must larger than 0
            # data[data < 0] = 0
            # Regulate the value
            if min_val is not None:
                data[data < min_val] = min_val
            if max_val is not None:
                data[data > max_val] = max_val
            # If need to convert to log space
            if is_log:
                data = np.log(data)

            key = '{:0>10d}'.format(counter)

            counter += 1

            # Here might block the thread
            key_queue.put(key)
            data_str = backend_func(data)

            bar.update_once()
        else:
            # Wait the thread to finish
            if DBThread.readytojoin():
                DBThread.join()
                bar.finish()
                break
            else:
                I('Waiting threads to finish. db size \033[0;31m%d\033[0m'
                  % DBThread.get_db_entries())
                time.sleep(1)

        # Get the datum string
        DBThread.put([key, data_str])

    # Finish
    I('Finished. DB size: \033[01;31m%d\033[0m' % DBThread.get_db_entries())


if __name__ == '__main__':
    main(sys.argv[1:])

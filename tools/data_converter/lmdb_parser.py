import lmdb
import numpy as np
import cv2


lmdb_file = "/Users/qianzhiming/Desktop/data/objcls-datasets/MosaicCls/lmdb/mosaic_train_lmdb"
lmdb_env = lmdb.open(lmdb_file)
lmdb_txn = lmdb_env.begin()
lmdb_cursor = lmdb_txn.cursor()

for key, value in lmdb_cursor:
    img = cv2.imdecode(np.fromstring(value, np.uint8), 3)
    # cv2.imshow("demo", img)
    # cv2.waitKey(0)

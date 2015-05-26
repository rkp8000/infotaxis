"""
Created on Fri Mar 27 14:35:44 2015

@author: rkp

Connect to infotaxis or test_infotaxis database.
"""

import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

TESTCXN = True

if TESTCXN:
    print 'CONNECTED TO INFOTAXIS TEST DATABASE'
    engine = create_engine(os.environ['TEST_INFOTAXIS_DB_CXN_URL'])
else:
    print 'CONNECTED TO INFOTAXIS PRODUCTION DATABASE'
    engine = create_engine(os.environ['INFOTAXIS_DB_CXN_URL'])

engine.connect()

Session = sessionmaker(bind=engine)
session = Session()
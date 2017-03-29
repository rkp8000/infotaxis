"""
Created on Fri Mar 27 14:35:44 2015

@author: rkp

Connect to infotaxis or test_infotaxis database.
"""

import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

TESTCXN = False

if TESTCXN:
    print 'CONNECTED TO INFOTAXIS TEST DATABASE'
    engine = create_engine(os.environ['TEST_INFOTAXIS_DB_CXN_URL'])
else:
    print 'CONNECTED TO INFOTAXIS PRODUCTION DATABASE'
    x = raw_input('Are you sure you want to connect to the production database [y or n]?')
    if x.lower() == 'y':
        engine = create_engine(os.environ['INFOTAXIS_DB_CXN_URL'])
    else:
        raise RuntimeError('User prevented write access to database.')

engine.connect()

Session = sessionmaker(bind=engine)
session = Session()
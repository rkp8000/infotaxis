"""
Created on Fri Mar 27 14:35:44 2015

@author: rkp

Connect to infotaxis or test_infotaxis database.
"""

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from cxn_vars import *

TESTCXN = True
URLTEMPLATE = 'mysql+mysqldb://{u}:{p}@{h}:{port}/{db}'


if TESTCXN:
    db = TESTDB
else:
    db = DB

cxn_url = URLTEMPLATE.format(u=USER, p=PASSWORD, h=HOST, port=PORT, db=db)
engine = create_engine(cxn_url)
engine.connect()

Session = sessionmaker(bind=engine)
session = Session()
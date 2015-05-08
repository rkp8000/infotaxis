"""
Created on Fri Mar 27 14:35:44 2015

@author: rkp

Connect to infotaxis database.
"""

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from cxn_vars import *

TESTCXN = True
URLTEMPLATE = 'mysql+mysqldb://{u}:{p}@{h}/{db}?unix_socket={sock}'


if TESTCXN:
    db = TESTDB
else:
    db = DB

cxn_url = URLTEMPLATE.format(u=USER, p=PASSWORD, h=HOST, db=db, sock=UNIXSOCKET)
engine = create_engine(cxn_url)
engine.connect()

Session = sessionmaker(bind=engine)
session = Session()
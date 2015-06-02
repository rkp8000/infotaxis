"""Write all script descriptions to script table in database."""

FNAME = 'script_descriptions.txt'

from db_api.connect import session
from db_api.models import Script

# get all ids and descriptions
desc_dict = {}
with open(FNAME, 'rb') as f:
    lines = (line for line in f if line.rstrip())

    # construct description dictionary
    script_id = None
    for line in lines:
        if line.startswith('ID:'):
            script_id = line[3:].strip()
            desc_dict[script_id] = ['', '']
        elif line.startswith('TYPE'):
            script_type = line[5:].strip()
            desc_dict[script_id][0] = script_type
        else:
            desc_dict[script_id][1] += (line + '\n')

# write everything to database
for script_id, info in desc_dict.items():
    script = Script(id=script_id, type=info[0], description=info[1])
    try:
        session.add(script)
        session.commit()
    except Exception, e:
        session.rollback()
        print e
        continue
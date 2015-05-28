__all__ = ['connect', 'models']

from datetime import datetime
from git import Repo
from models import Script, ScriptExecution


def add_script_execution(script_id, session, multi_use=False, notes=None):
    """Write the script execution to the database."""

    # check if script_id is in Script table
    script = session.query(Script).get(script_id)
    if not script:
        raise LookupError('"{}" not found in table "script"!'.format(script_id))

    # raise error if script has already been executed and multi use is false
    if not multi_use:
        if list(session.query(ScriptExecution).filter_by(script_id=script_id)):
            raise RuntimeError('Script "{}" has already been executed!'.format(script_id))

    # get latest commit
    repo = Repo('/Users/rkp/Dropbox/Repositories/infotaxis')
    latest_commit = repo.commit('master')

    # add script execution to database
    session.add(ScriptExecution(script=script, commit=latest_commit, timestamp=datetime.now()), notes=notes)
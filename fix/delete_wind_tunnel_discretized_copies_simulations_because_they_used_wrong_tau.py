SCRIPT_ID = 'delete_wind_tunnel_discretized_copies_simulations_because_they_used_wrong_tau'
SCRIPT_NOTES = 'Delete wind tunnel discretized copies simulations and their trials.'

SIM_TEMPLATE = 'wind_tunnel_discretized_copies%'

from db_api.connect import session
from db_api import models, add_script_execution


def main():
    add_script_execution(script_id=SCRIPT_ID, session=session, notes=SCRIPT_NOTES)

    sims = session.query(models.Simulation).filter(models.Simulation.id.like(SIM_TEMPLATE))

    for sim in sims:
        for trial in sim.trials:
            session.delete(trial)
        session.delete(sim)
    session.commit()


if __name__ == '__main__':
    main()
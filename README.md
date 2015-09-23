# README

This repository contains code for generating infotactic trajectories according to http://www.nature.com/nature/journal/v445/n7126/full/nature05464.html .


### How to use this code.

1. make new database
2. run models.py; this creates all the empty database tables
3. run update_script_descriptioons.py. This writes all the script descriptions to the database; if you add a new description at some point (corresponding to making a new script), then you'll have to rerun this file
4. to redo all the simulations run the following scripts
    * make_wind_tunnel_discretized_geom_configs.py: this creates and stores the geometrical configurations for the wind tunnel trajectories, as well as their "extensions", which contain some meta info
    * open generate_discretized_wind_tunnel_trajectory_copies.py, set the desired parameters in the configuration file (such as diffusivity), make sure to update the script notes to account for this, and run it; this will make discretized copies of all the wind tunnel trajectories and store them in the database; the plume parameters estimated by the insect are important because they determine how it would theoretically update its distribution over possible plume source locations
    * open generate_wind_tunnel_discretized_matched_trials_one_for_one.py, set the desired parameters in the configuration file (such as diffusivity), update the script notes to account for this, and run it; this will generate one infotaxis trajectory for every real trajectory, with its starting location and flight duration matched
    * run the previous two scripts with different parameters to explore how varying the insect's estimate of the plume will affect the results
5. to analyze the simulations, do the following scripts
    * almost all analysis scripts can be run without regard to which is run first except for:
        * make_segment_groups_and_segments.py; this file splits up trajectories into segments triggered on plume entering and exiting; make sure to define which simulation groups you want to compare in the config file
        * make_exit_triggered_heading_ensembles.py; this file looks at ensembles of heading data triggered on plume entering and exiting; make sure to define which segment group to use in the config file
6. to view results, use the figure-making scripts in figures/wind_tunnel_discretized; just make sure to specify in the config files which data sets you want to look at and any relevant plotting parameters
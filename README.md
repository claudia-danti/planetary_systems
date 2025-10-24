# planetary_systems
This repository contains the code that i developed during my PhD in Astrophysics at the University of Copnehagen to simulate the growth and evolution of a multi-planetary system.
The code is a 1D semi-analytical pebble accretion model for planet formation. It is able to simulate the growth and migration of $n$ planets (embryos are to be inserted as a starting condition), considering pebble and drift limited size pebbles, mutual pebble filtering between planets, gas accretion and type I and II migration. The disc model can be chose between fully irradiated and viscoulsy heated + irradiated.
This code has been used for the publication [Danti et al. 2025](https://www.aanda.org/articles/aa/full_html/2025/08/aa55095-25/aa55095-25.html).
The code is structured as follows:

- multiple_planets_gas_acc.py: contains the main routine of the code that handles the description of the differential equation with pebble filtering and the ODE solver
- functions_pebble_accretion.py: contains the class that handles the pebble and gas accretion equations 
- functions.py: contains the most relevant functions used throughout the code
- functions_plotting.py: contains the main plotting functions relevant to the analysis
- .jpynb: the different Jupyter Notebooks contain plotting scripts for the analysis



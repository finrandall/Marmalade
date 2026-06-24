from interface import get_parameters
from simulation import Simulation


params = get_parameters()

simulation = Simulation(params)
simulation.run()
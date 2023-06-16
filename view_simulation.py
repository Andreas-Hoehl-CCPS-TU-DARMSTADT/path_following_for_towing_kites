""" Use this script to show some more details of a simulation result.

Specify the folder and the name of the simulation in the main and then run the script.
This will open some figures with more details of the result.

If more information is needed than you what can see, have a look at simulation/simulator to see what is stored in the
simulation result. It might also be helpful to look hat the code of the functions visualize_result in simulation and
collect_info in controller/my_MPC.
"""
from simulate import visualize_result

if __name__ == '__main__':
    folder = 'parameter_comparison_pf'
    name = 's_10.0_q11_8000.0'
    visualize_result(name, folder)

from python_kmc import dendrite_model
from python_kmc.kmc_src.d1_1 import plot_functions_1d

print("Generating KMC model...")
my_kmc_model = dendrite_model.generate_kmc_model(lattice_size=256)

print("Running KMC simulation...")
# show progress is set false, because the progess bar is slowing down the simulation
my_kmc_model.run_simulation_t_end(t_end_seconds=5e-6*50, show_progress=False) 

print("Plotting results...")
plot_functions_1d.plot_1d1_lattice(input_vector=my_kmc_model.lattice.heights)

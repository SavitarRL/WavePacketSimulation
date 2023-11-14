# WavePacketSimulation

Wave packet simulation for 2D Hermitian Goldman fermionic model given below:

$`\hat{H} = J\sum_{m,n}(\hat{c}^\dagger_{m,n+1}e^{i\phi\sigma_y}\hat{c}_{m,n} + \hat{c}^\dagger_{m,n}e^{-i\phi\sigma_y}\hat{c}_{m,n+1} + \hat{c}^\dagger_{m+1,n}e^{i\theta\sigma_x}\hat{c}_{m,n}+ \hat{c}^\dagger_{m,n}e^{-i\theta\sigma_x}\hat{c}_{m+1,n})`$

$`J`$ is the interaction strength, $`m,n`$ the lattice indices, $`\hat^{c}^\dagger(c^\dagger)`$ are fermioninc creation(annihilation) operators, `\sigma_x,\sigma_y` are Pauli operators and `\theta,\phi` are phase factors.

More details can be found in `simulation_2DHermitian.pdf`

Simulation result in real space (a 20x20 lattice as in `main.ipynb`):

![](https://github.com/SavitarRL/WavePacketSimulation/blob/main/GIF/trial_gif_20%20by%2020.gif)

Acknowledgements: Mr. Bengy Wong (University of Hong Kong)

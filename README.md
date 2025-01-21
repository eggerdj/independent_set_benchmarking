# Independent set benchmarking

This repository houses the code that runs the independent set benchmark of the quantum optimization benchmarking framework [1].
We study the two independent set problems with 17 and 52 decision variables on quantum hardware.
These problems are mapped to a quadratic binary unconstrained problem with the Lagrange multiplier formalisme.
The underlying quantum circuits are built from layers of SWAP gates [2, 3, 4].

## Repository structure

The repository contains the following folders

* `instances` houses the graphs of the two independent set instances studied here
* `notebooks` the notebooks that run the benchmarks
* `plots` a folder where the plots are stored
* `results_hardware` stores the shots from the hardware runs in json format
* `solutions` the solution bitstrings found by the quantum hardware
* `independent_set_benchmarking` the python code that the otebooks rely on

## References

[1] Will be updated when the paper is public

[2] Weidenfeller *et al.*, Quantum **6**, 870 (2022).

[3] Sack & Egger, Phys. Rev. Research **6** (1), 013223 (2024).

[4] Matsuo *et el.*, IEICE Trans. Fundamentals **106**, 1424-1431 (2023).

## IBM Public Repository Disclosure

All content in these repositories including code has been provided by IBM under the associated open source software license and IBM is under no obligation to provide enhancements, updates, or support. 
IBM developers produced this code as an open source project (not as an IBM product), and IBM makes no assertions as to the level of quality nor security, and will not be maintaining this code going forward.

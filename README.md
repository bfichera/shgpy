# ShgPy
[![DOI](https://zenodo.org/badge/267391726.svg)](https://zenodo.org/badge/latestdoi/267391726)

ShgPy is a simple toolkit for analyzing rotational-anisotropy second harmonic generation (RA-SHG) data. It depends mainly on three packages, [NumPy](https://numpy.org/), [SciPy](https://www.scipy.org/), and [SymPy](https://www.sympy.org/en/index.html) -- as well as (optionally) [Matplolib](https://matplotlib.org/) for some basic plotting capability -- to simulate, manipulate, and fit RA-SHG data in a (hopefully!) intuitive way.

Fitting RA-SHG data involves solving a complex global minimization problem with many degrees of freedom. Moreover, the fitting functions naively involve a degree of complexity due to the trigonometric nature of the problem. This software therefore takes a dual approach to fitting RA-SHG data -- first of all, all of the fits are done in Fourier space, which significantly reduces the complexity of the fitting formulas, and second of all, ShgPy makes heavy use of the [scipy.optimize.basinhopping](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.basinhopping.html) algorithm, which is particularly useful for these types of global optimization problems.

For more information, please see the [tutorials and documentation](https://bfichera.github.io/shgpy/).

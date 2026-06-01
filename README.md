# ms-eetc

Open-source code for paper: "Direct Multiple Shooting for Computationally Efficient Train Trajectory Optimization"

- [Link to publication](https://www.sciencedirect.com/science/article/pii/S0968090X23001596)

# Installation

## Setting up a virtual environment

### Windows

Tested with Python 3.9.13 on Windows 10 (10.0.19044)

1. Open a terminal in the root folder of the repository and create a virtual environment:

   ```
   python -m venv .env
   ```

   > `.env` is an example name; any other name may be used.

2. Activate the virtual environment:

   ```
   .\.env\Scripts\activate
   ```

3. Install the package:

   ```
   pip install -e .
   ```

4. Deactivate the virtual environment when finished:

   ```
   deactivate
   ```

### Linux

Same as on Windows, but activate the virtual environment with:

```
source .env/bin/activate
```

# Simulations

1. Open a terminal and activate the virtual environment (see instructions above).

2. Navigate to the `simulations` folder and run a script of your choice. For example:

   ```
   python figure5.py
   ```

   This generates Figure 5 of the paper (`figure5.pdf`).

3. LaTeX fonts on the plots will be used only if a valid LaTeX installation is found.

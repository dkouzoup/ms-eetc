# ms-eetc

Open-source code for paper: "Direct Multiple Shooting for Computationally Efficient Train Trajectory Optimization"

- [Link to preprint](http://dx.doi.org/10.2139/ssrn.4264720)

# Installation

## Setting up a virtual environment

### Windows

- Tested with Python 3.9.13 on Windows 10 (10.0.19044)

- Open terminal in root folder of repository and type `python -m venv .env` to create a virtual environment

    - `.env` is an example for the folder name, you may choose any other name

- Activate the virtual environment: `.\.env\Scripts\activate`

- Install dependencies with: `pip install -r requirements.txt`

- deactivate virtual environment by typing: `deactivate`

### Linux

- Same as on Windows but use `source .env/bin/activate` to activate virtual environment

# Simulations

- Open a terminal and activate the virtual environment (see instructions above)

- Go to the `simulations` folder and run a script of your choice

    - Example: `python figure5.py` to generate Figure 5 of paper (`figure5.pdf`)

- Latex fonds on the plots will be used only if a valid latex installation is found

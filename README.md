# XCAT-iCMR

XCAT-iCMR is a modular simulation package for dynamic interventional
cardiovascular MRI using XCAT phantoms.

The project is currently a scaffold. Planned stages include phantom
generation, MR contrast, Gd balloon simulation, k-space encoding,
undersampling, and noise.

## Development setup

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
```

Create a simulation configuration from the provided template:

```bash
cp configs/simulation.template.yaml configs/my_simulation.yaml
```

Validate it without running XCAT or generating simulation data:

```bash
xcat-icmr validate configs/my_simulation.yaml
```

The validator checks the YAML schema, relationships between sections, and
required external files. Simulation stages are not implemented yet.

Inspect the Pulseq signature, resolved MATLAB metadata, and oriented
trajectory without running the simulation:

```bash
xcat-icmr inspect-sequence configs/my_simulation.yaml
```

An existing MATLAB v7.3 `par` file can be checked field-by-field:

```bash
xcat-icmr inspect-sequence configs/my_simulation.yaml \
  --matlab-reference /path/to/par_reference.mat
```

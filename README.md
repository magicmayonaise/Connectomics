# Connectomics
Find your favorite neurons + their synaptic partners!

## FAFBseg Synaptic Partner GUI

This repository now includes a Tkinter-based desktop interface for exploring FAFB
segmentation materializations. Provide a neuron root ID and a materialization
version to discover both postsynaptic (outgoing) and presynaptic (incoming)
partners, along with synapse counts and percentages.

### Requirements

- Python 3.9+
- [`fafbseg`](https://pypi.org/project/fafbseg/) (which installs the required
  `caveclient` dependency) or `caveclient` directly

### Running the application

```bash
python -m connectomics_gui
```

Enter the integer root ID and materialization version you wish to query, then
press **Fetch partners**. The datastack defaults to `flywire_fafb_production`
but can be changed from within the GUI if you work with a different stack.

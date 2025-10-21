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

Use the **Filter partners** field in the *Postsynaptic partners* tab to narrow
down the list to specific root IDs. Provide partial IDs or separate multiple
values with spaces or commas. The table summary indicates when results have
been filtered or truncated to the top entries displayed in the table.

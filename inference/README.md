# RFdiffusion Inference

This directory contains the main inference script for running RFdiffusion protein design.

## Quick Start

```bash
# Basic unconditional design
python inference/run_inference.py 'contigmap.contigs=[100-100]' inference.output_prefix=my_design

# Motif scaffolding  
python inference/run_inference.py inference.input_pdb=input.pdb 'contigmap.contigs=[50/A10-20/50]' inference.output_prefix=scaffold_design

# Binder design
python inference/run_inference.py inference.input_pdb=target.pdb 'contigmap.contigs=[A/0 70-100]' 'ppi.hotspot_res=[A30,A31,A32]' inference.output_prefix=binder_design
```

## Main Script

**`run_inference.py`**: Main inference entry point using Hydra configuration system

### Key Parameters

- **`contigmap.contigs`**: Defines protein structure specification
- **`inference.input_pdb`**: Input PDB file for conditioning
- **`inference.output_prefix`**: Output file prefix
- **`inference.num_designs`**: Number of designs to generate (default: 1)
- **`diffuser.T`**: Number of diffusion steps (default: 50)

### Configuration Files

The script uses Hydra configs from `config/inference/`:
- **`base.yaml`**: Default inference settings
- **`symmetry.yaml`**: Symmetric oligomer generation settings

## Usage Examples

### Unconditional Generation
```bash
# Generate 100-residue protein
python inference/run_inference.py 'contigmap.contigs=[100-100]' inference.output_prefix=unconditional_100

# Generate with specific length range
python inference/run_inference.py 'contigmap.contigs=[80-120]' inference.output_prefix=variable_length
```

### Motif Scaffolding
```bash
# Scaffold residues 10-20 from chain A
python inference/run_inference.py \
    inference.input_pdb=examples/input_pdbs/5TPN.pdb \
    'contigmap.contigs=[10-40/A163-181/10-40]' \
    inference.output_prefix=motif_scaffold
```

### Binder Design
```bash
# Design binder to target protein
python inference/run_inference.py \
    inference.input_pdb=target.pdb \
    'contigmap.contigs=[A/0 70-100]' \
    'ppi.hotspot_res=[A30,A31,A32]' \
    inference.output_prefix=binder
```

### Symmetric Design
```bash
# C4 symmetric design
python inference/run_inference.py \
    --config-name symmetry \
    inference.symmetry=c4 \
    'contigmap.contigs=[100-100]' \
    inference.output_prefix=c4_symmetric
```

### Advanced Options

```bash
# Partial diffusion (design diversification)
python inference/run_inference.py \
    inference.input_pdb=starting_structure.pdb \
    'contigmap.contigs=[100-100]' \
    diffuser.partial_T=20 \
    inference.output_prefix=diversified

# Use specific model checkpoint
python inference/run_inference.py \
    'contigmap.contigs=[50-50]' \
    inference.ckpt_override_path=models/ActiveSite_ckpt.pt \
    inference.output_prefix=activesite_design

# Add auxiliary potentials
python inference/run_inference.py \
    'contigmap.contigs=[100-100]' \
    'potentials.guiding_potentials=["type:olig_contacts,weight_intra:1,weight_inter:0.1"]' \
    potentials.guide_scale=2 \
    inference.output_prefix=with_potentials
```

## Output Files

- **`{prefix}_0.pdb`**: Generated protein structure (designed regions as glycine)
- **`{prefix}_0.trb`**: Metadata including config, mappings, and pLDDT scores
- **`traj/{prefix}_0_*_traj.pdb`**: Trajectory files showing diffusion process

## Model Checkpoints

Different models are automatically selected based on the task:
- **`Base_ckpt.pt`**: General unconditional generation
- **`Complex_base_ckpt.pt`**: Protein-protein interactions
- **`ActiveSite_ckpt.pt`**: Small motif scaffolding
- **`InpaintSeq_ckpt.pt`**: Sequence inpainting during scaffolding

## Examples

See the `examples/` directory for comprehensive usage examples:
- `design_unconditional.sh`: Basic unconditional generation
- `design_motifscaffolding.sh`: Motif scaffolding examples
- `design_ppi.sh`: Protein-protein interaction design
- `design_*_oligos.sh`: Symmetric oligomer design

## Configuration System

RFdiffusion uses Hydra for configuration management. You can:

1. **Override any parameter**: `param.subparam=value`
2. **Use config files**: `--config-name=symmetry`
3. **Set config path**: `--config-path=../config/inference`

For detailed parameter descriptions, see the configuration files in `config/inference/`.

## Related Documentation

- [examples/](../examples/): Complete usage examples
- [config/inference/](../config/inference/): Configuration files
- [setup/](../setup/): Environment setup and installation
- [CLAUDE.md](../CLAUDE.md): Main project instructions
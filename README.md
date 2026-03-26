# GLADE: Gravitational Lensing Analysis and Differential Evolution

[[License: MIT](https://opensource.org/licenses/MIT)]
[[Python 3.8+](https://www.python.org/downloads/)]
[[CI](https://github.com/y31ling/glaDE/actions)]

**GLADE** is a unified gravitational lensing analysis platform that integrates multiple lens models with modern optimization algorithms. Built on top of [glafic2](https://github.com/oguri/glafic2), GLADE specializes in subhalo detection and strong lensing system modeling.

## 🌟 Features

- **Multi-Model Support**: Point Mass, NFW, King, and Pseudo-Jaffe lens models
- **Optimization**: Differential Evolution (DE) and MCMC sampling algorithms  
- **Deployment**: Automated Linux environment setup and dependency management
- **Interface**: Single entry point through `main.py` for all models
- **Toolset**: Complete analysis pipeline from modeling to visualization

## 🚀 Quick Start

### System Requirements

- **OS**: Linux (Ubuntu 18.04+ / CentOS 7+ recommended)
- **Python**: 3.8 or higher
- **Compiler**: GCC with C/C++ support
- **Storage**: At least 2GB available disk space
- **Memory**: 4GB RAM recommended for large-scale MCMC sampling

### Installation

```bash
# Clone the repository
git clone https://github.com/y31ling/glaDE.git
cd glaDE

# One-click installation (first time setup)
bash bootstrap_linux.sh

# Run the project
./run_glade.sh
```

The bootstrap script automatically:

- Installs system dependencies (apt packages)
- Downloads and compiles CFITSIO, FFTW, GSL libraries
- Builds glafic2 binary and Python modules
- Creates virtual environment and installs Python dependencies
- Generates environment scripts (`env.sh`, `run_glade.sh`)

### Quick Configuration

Edit `main.py` to configure your analysis:

```python
# Select lens model
model_use = "point_mass"  # Options: 'nfw', 'king', 'p-jaffe'

# Set data directory (containing bestfit.dat)
# Or you can simply change the parameters in the scripts directly
source_dir = "work/your_data_directory"

# Configure optimization parameters
common_overrides = {
    "active_subhalos": [1, 2, 3, 4],  # Enable all subhalos
    "DE_MAXITER": 800,                # DE iterations
    "DE_POPSIZE": 75,                 # Population size
    "MCMC_ENABLED": True,             # Enable MCMC sampling
    "MCMC_NSTEPS": 5000,             # MCMC steps
    ...
}
```

## 📁 Project Structure

```
glade/
├── main.py              # Main entry point
├── bootstrap_linux.sh   # One-click installation script
├── requirements.txt     # Python dependencies
├── runner.py           # Model execution engine
├── runtime_env.py      # Runtime environment setup
├── injector.py         # Parameter injection system
├── glafic2/            # glafic gravitational lensing engine
│   ├── *.c, *.h       # C source code
│   ├── Makefile       # Build configuration
│   └── python/        # Python bindings
├── legacy/             # Lens model implementations
│   ├── v_pointmass_1_0/  # Point mass model
│   ├── v_nfw_2_0/        # NFW profile model
│   ├── v_king_1_0/       # King profile model
│   └── v_p_jaffe_2_0/    # Pseudo-Jaffe model
├── tools/              # Analysis toolkit
│   ├── glafic_optimize.py  # glafic-based optimization
│   ├── mcmc_from_result.py # MCMC post-processing
│   ├── drawgraph.py        # Visualization tools
│   └── replot_mcmc.py      # MCMC plotting
└── samples/            # Example datasets
```

## 🔧 Supported Lens Models


| Model            | Description                 | Parameters         | Use Case            |
| ---------------- | --------------------------- | ------------------ | ------------------- |
| **Point Mass**   | Point mass approximation    | x, y, mass         | Subhalo detection   |
| **NFW**          | Navarro-Frenk-White profile | x, y, M_vir, c_vir | Dark matter halos   |
| **King**         | King profile                | x, y, mass, r_c, c | Globular clusters   |
| **Pseudo-Jaffe** | Pseudo-Jaffe profile        | x, y, σ, a, r_co   | Elliptical galaxies |


## 🎯 Usage Examples

### Basic Lens Modeling

```python
# 1. Configure model
model_use = "point_mass"
source_dir = ""

# 2. Set optimization parameters
common_overrides = {
    "active_subhalos": [1, 2, 3, 4],
    "DE_MAXITER": 650,
    "MCMC_ENABLED": True,
    "MCMC_NSTEPS": 5000,
}

# 3. Run analysis
./run_glade.sh
```

### Advanced MCMC Sampling

```python
model_overrides = {
    "point_mass": {
        "MCMC_ENABLED": True,
        "MCMC_NWALKERS": 64,
        "MCMC_NSTEPS": 10000,
        "MCMC_BURNIN": 1000,
        "fine_tuning": True,  # Enable fine-tuning mode
        "fine_tuning_configs": {
            1: {"search_radius": 0.080, "mass_guess": 1.0e5},
            2: {"search_radius": 0.070, "mass_guess": 5.0e4},
        }
    }
}
```

### Parameter Override System

GLADE uses a flexible parameter override mechanism:

- Override keys match variable names in the original model scripts
- Common parameters: `active_subhalos`, `DE_MAXITER`, `MCMC_ENABLED`
- Model-specific parameters can be set in `model_overrides`
- Generated scripts are saved as `legacy/<model_dir>/_glade_generated_run.py`

## 📊 Output and Results

Results are saved in `results/<model_name>/` with timestamp subdirectories:

- `*_best_params.txt`: Optimized parameters
- `*_corner.png`: MCMC posterior distribution plots
- `*_triptych.png`: Three-panel analysis visualization
- `*_mcmc_chain.dat`: Raw MCMC sampling chain
- `*_optresult.dat`: Optimization convergence data

## 🛠️ Analysis Toolkit

### Load Environment

```bash
source env.sh  # Load runtime environment
```

### glafic Optimization Tool

```bash
# Using glafic optimization(ameoba) to repeat optimization process for an existing result 
python tools/glafic_optimize.py results/point_mass/result_dir/

# Enable MCMC sampling
python tools/glafic_optimize.py results/point_mass/result_dir/ --mcmc --mcmc_nsteps 50000

# Verbose output with restart control
python tools/glafic_optimize.py results/point_mass/result_dir/ --verbose --max_restart 5
```

### MCMC Post-Processing

```bash
# Generate MCMC sampling from optimization results
python tools/mcmc_from_result.py results/point_mass/241126_1234/

# Replot MCMC results with custom settings
python tools/replot_mcmc.py results/point_mass/241126_1234/

# Generate publication-quality plots
python tools/drawgraph.py results/point_mass/241126_1234/
```

## 📚 Technical Details

### Core Algorithms

- **Differential Evolution**: Global optimization with adaptive parameters
- **MCMC Sampling**: Bayesian parameter estimation with `emcee`
- **Hungarian Algorithm**: Optimal image matching for multi-image systems
- **Adaptive Mesh Refinement**: High-precision lens equation solving

### Performance Features

- Multi-core parallel processing support
- Early stopping mechanisms and convergence detection
- Memory-efficient data structures
- Batch processing and intelligent caching
- GPU acceleration ready (future development)

### Optimization Features

- **Flexible subhalo selection**: Choose which images to fit subhalos near
- **Fine-tuning mode**: Independent configuration for each subhalo
- **Parameter bounds**: Automatic and custom parameter range setting
- **Convergence monitoring**: Real-time optimization progress tracking

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **glafic2**: Core gravitational lensing computation engine by Masamune Oguri, Really Great Thanks!
- **SciPy ecosystem**: Scientific computing foundation
- **emcee**: Affine-invariant MCMC sampling
- **corner**: Posterior distribution visualization
- **matplotlib**: Publication-quality plotting

## 📞 Support and Contact

- **Issues**: [GitHub Issues](https://github.com/y31ling/glaDE/issues)
- **Discussions**: [GitHub Discussions](https://github.com/y31ling/glaDE/discussions)

## 📈 Roadmap

- GPU acceleration support
- Web-based interface
- Additional lens models (Sersic, Einasto)
- Docker containerization
- Intergrate Differential Evolution into glafic command directly

## !!  Known Issues

- Encountering Multiple Errors when the subhalos list is empty, plz double check your result when you are doing that.
- Optimization-DE command doesn't work correctly, plz don't use it for now.

---


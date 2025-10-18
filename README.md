## Building the Project
```bash
cmake --build build && build/OpenGLWebcam
```

## Controls

### Filters
| Key | Filter |
|-----|--------|
| `Space` | No filter |
| `1` | Grayscale |
| `2` | Gaussian Blur |
| `3` | Edge Detection |
| `4` | Pixelation |
| `5` | Comic Art |

### GPU/CPU, default GPU
| Key | Action |
|-----|--------|
| `G` | Toggle GPU/CPU mode |

### Geometric Transformations
| Input | Transformation |
|-------|----------------|
| **Left-click + Drag** | Translate |
| **Right-click + Drag** | Rotate |
| **Scroll Wheel** | Scale |
| `R` | Reset all transformations |

### Benchmarking
| Key | Benchmark Type | Output File | Frames |
|-----|----------------|-------------|--------|
| `B` | Manual benchmark | `performance_manual.csv` | 50 |
| `A` | Basic (no transforms) | `performance_transforms.csv` | 50 |
| `T` | Full (with transforms) | `performance_transforms.csv` | 50 |
| `V` | Resolution test | `performance_resolution.csv` | 50 |
| `S` | Save manual results | `performance_manual.csv` | - |

### Other
| Key | Action |
|-----|--------|
| `Q` | Quit |

## Analyzing Results

You can analyse your results using:
```bash
python analyze_performance.py
```

## Supported Resolutions

- **VGA**: 640×480 (307,200 pixels)
- **HD**: 1280×720 (921,600 pixels)
- **Full HD**: 1920×1080 (2,073,600 pixels)
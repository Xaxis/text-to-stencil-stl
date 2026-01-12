# Text to Stencil STL

Convert text into 3D-printable stencil plates (STL files). A stencil is a plate with letters cut out so you can spray paint through it.

## Installation

```bash
pip install numpy shapely matplotlib trimesh mapbox_earcut
```

## Quick Start

**Basic usage** (uses included Warband Stencil font):

```bash
python main.py --text "HELLO" --out hello.stl
```

**Multi-line text:**

```bash
python main.py --text "HELLO\nWORLD" --out hello_world.stl
```

**Custom font:**

```bash
python main.py --text "HELLO" --font /path/to/font.ttf --out hello.stl
```

## Common Options

| Option | Default | Description |
|--------|---------|-------------|
| `--font-size` | 40 | Font size in mm |
| `--margin` | 12 | Margin around text in mm |
| `--plate-thickness` | 2.2 | Plate thickness in mm |
| `--align` | center | Text alignment (left, center, right) |
| `--rotate` | 0 | Rotate text in degrees |
| `--corner-radius` | 0 | Rounded corners in mm |

## Advanced Features

### Stencilify Mode

Turn any regular font into a stencil by adding periodic breaks:

```bash
python main.py --text "HELLO" --stencilify --out hello.stl
```

Options:
- `--stencil-gap` - Width of breaks (default: 2mm)
- `--stencil-period` - Spacing between breaks (default: 16mm)
- `--stencil-angle` - Rotation of break pattern (default: 0°)

### Bridge Mode

Add bridges to connect counter holes (centers of O, A, B, etc.):

```bash
python main.py --text "BOARD" --bridge-width 1.5 --out board.stl
```

### JSON Layout

For complex multi-text arrangements, create a JSON file:

```json
[
  {"text": "BIG", "size_mm": 60, "x": 0, "y": 30},
  {"text": "small", "size_mm": 20, "x": 0, "y": -10}
]
```

```bash
python main.py --layout layout.json --out custom.stl
```

## Complete Parameter Reference

### Required Parameters

| Parameter | Description |
|-----------|-------------|
| `--text` or `--layout` | Text string (use `\n` for newlines) OR path to JSON layout file |
| `--out` | Output STL file path |

### Font & Text Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--font` | System default | Path to TTF/OTF font file |
| `--font-size` | 40.0 | Font size in millimeters |
| `--line-spacing` | 1.15 | Line spacing multiplier |
| `--align` | center | Text alignment (left, center, right) |
| `--rotate` | 0.0 | Rotate entire text block in degrees |
| `--scale-to-width` | 0.0 | Scale layout to fit width in mm (0 = disabled) |

### Stencil Break Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--stencilify` | False | Enable stencil break mode |
| `--stencil-gap` | 2.0 | Break width in mm |
| `--stencil-period` | 16.0 | Spacing between breaks in mm |
| `--stencil-angle` | 0.0 | Rotate break pattern in degrees |
| `--stencil-phase` | 0.0 | Shift break pattern along axis in mm |
| `--stencil-direction` | vertical | Break direction (vertical or horizontal) |

### Bridge Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--bridge-width` | 0.0 | Width of bridges for counter holes in mm |

### Plate Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--margin` | 12.0 | Margin around text in mm |
| `--corner-radius` | 0.0 | Rounded corner radius in mm |
| `--plate-thickness` | 2.2 | Plate thickness in mm |

### Debug Parameters

| Parameter | Description |
|-----------|-------------|
| `--debug-text-svg` | Output path for raw text geometry SVG |
| `--debug-cutout-svg` | Output path for cutout geometry SVG |
| `--debug-svg` | Output path for final plate geometry SVG |

## Tips & Best Practices

### Choosing Between Stencilify and Bridges

- **Use `--stencilify`**: For regular fonts to create stencil-style breaks throughout the entire text
- **Use `--bridge-width`**: For stencil fonts or when you only need to connect counter holes (like in O, A, B)
- **Don't use both**: They serve similar purposes; choose one approach

### Font Recommendations

- **Stencil fonts**: Use with `--bridge-width` for clean results
- **Regular fonts**: Use with `--stencilify` to add breaks
- **Bold fonts**: Work better for stencils than thin fonts
- **Sans-serif fonts**: Generally easier to print than serif fonts

### 3D Printing Tips

- **Plate thickness**: 2-3mm works well for most applications
- **Margin**: At least 10-15mm for structural integrity
- **Font size**: Minimum 20mm for small details to print clearly
- **Bridge width**: 1-2mm is usually sufficient
- **Stencil gap**: 2-3mm ensures breaks are visible and functional

### Troubleshooting

**Problem**: Letters with holes (O, A, B) have floating centers
- **Solution**: Use `--bridge-width 1.5` or `--stencilify`

**Problem**: Text is too small/large
- **Solution**: Adjust `--font-size` or use `--scale-to-width`

**Problem**: Stencil breaks don't align well
- **Solution**: Adjust `--stencil-period`, `--stencil-angle`, or `--stencil-phase`

**Problem**: Mesh generation fails
- **Solution**: Try a different font or simplify the text

## How It Works

1. **Text Rendering**: Converts text to 2D vector paths using matplotlib
2. **Layout**: Arranges text according to alignment, spacing, and rotation
3. **Stencilify** (optional): Subtracts periodic bars to create breaks
4. **Bridges** (optional): Adds connections for counter holes
5. **Plate Generation**: Creates a plate outline with margins and rounded corners
6. **Cutout**: Subtracts text geometry from plate
7. **Extrusion**: Converts 2D geometry to 3D mesh using triangulation
8. **Export**: Saves as STL file ready for 3D printing

## License

This project is open source. Feel free to use and modify as needed.

## Contributing

Contributions are welcome! Feel free to submit issues or pull requests.

## Acknowledgments

Built with:
- [NumPy](https://numpy.org/) - Numerical computing
- [Shapely](https://shapely.readthedocs.io/) - Geometric operations
- [Matplotlib](https://matplotlib.org/) - Text rendering
- [Trimesh](https://trimsh.org/) - 3D mesh processing
- [Mapbox Earcut](https://github.com/mapbox/earcut) - Polygon triangulation
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

## Tips

**Using stencil fonts (recommended):**
- The script automatically uses the included Warband Stencil font
- Stencil fonts have built-in breaks so letters stay connected
- No need for `--stencilify` or `--bridge-width`

**Using regular fonts:**
- Add `--stencilify` to create breaks automatically
- Or use `--bridge-width 1.5` to just connect counter holes (O, A, B, etc.)

**3D Printing:**
- Default thickness (2.2mm) works well for most printers
- Use at least 10mm margin for structural strength
- Minimum font size of 20mm recommended for detail

## All Options

Run `python main.py --help` to see all available options.
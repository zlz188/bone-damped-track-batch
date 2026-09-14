# Bone Damped Track Batch

Batch-add Damped Track constraints to armature bones with one click, with
gradient influence control (constant / increasing / decreasing / proportional
along each bone chain).

Built for cloth, hair and skirt chains — especially MMD-style rigs.

## Features

- **Batch add** Damped Track constraints to a selected bone chain (or all bones)
- **Gradient influence**: constant, increase/decrease by depth step, or
  proportional along the chain length with easing curves
  (linear / ease-in / ease-out / smooth)
- **Smart filters**:
  - All bones
  - Pose "green" bones (bones that already carry constraints, reimplementing
    Blender's pose color logic without reading pixels)
  - Keyword-matched cloth/hair bones (matches bone name, MMD Japanese name and
    MMD English name, case-insensitive)
- **Manual include / exclude** keyword lists, plus an **eyedropper** to pick
  bones directly in the 3D viewport
- **Live preview**: highlight the bones the current settings would affect
- **Safe**: only manages constraints it created (prefixed `阻尼追踪_`), never
  touches your hand-tuned constraints; one-click cleanup for generated
  constraints
- Compatible with Blender 4.x and 5.0 (GPU shader fallback included)

## Requirements

- Blender 4.2 or newer (tested on 4.5 / 5.0)

## Installation

### Option A — Extensions Platform (recommended)

1. In Blender: Edit → Preferences → Get Extensions → *Install from Disk*.
2. Select the downloaded `.zip` and enable **Bone Damped Track Batch**.

### Option B — Manual

1. Download the `.zip` from the GitHub Releases page.
2. Edit → Preferences → Add-ons → Install from Disk, select the `.zip`.
3. Enable **Bone Damped Track Batch**.

## Usage

1. Select an armature and open the 3D Viewport N panel → "骨骼工具" tab.
2. Set the scope (selected sub-chain or all bones), filter, influence and
   gradient options.
3. Click **Preview** to highlight matched bones, then **Batch Add**.
4. Use **Clear** to remove all plugin-created constraints in the selected chain.

## Support

- Issues & feature requests: GitHub Issues
- This project is open source. Contributions are welcome.

## License

GPL-2.0-or-later — see [LICENSE](LICENSE).

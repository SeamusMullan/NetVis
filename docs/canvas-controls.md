# Canvas controls

How to move around the graph canvas, and how that compares with Netron. "Pan" means
the content moves with the gesture. All zoom is clamped to 2% to 400%.

## Mouse and trackpad

| Gesture | What it does |
|---|---|
| Scroll wheel, or a two-finger swipe | Pans up/down and left/right (with **Scroll wheel: Zoom**, zooms at the pointer instead) |
| Shift + scroll | Pans sideways |
| Ctrl + scroll | Zooms at the pointer, 1.1x per notch. On macOS, Cmd + scroll and Control + scroll both zoom |
| Pinch (macOS trackpad) | Zooms at the pointer (expected; not yet verified on hardware) |
| Left-drag, anywhere on the canvas, including over a node | Pans. The content stays glued to the cursor once the drag passes 6 px, and the pan continues outside the canvas while the button is held |
| Middle-drag, or Space + left-drag | Pans (same rules as left-drag) |
| Drag in the minimap | Moves the view in the minimap only; the canvas does not pan or select |
| Click | Selects the node on **release**, if the pointer did not move; clicking empty space clears the selection |
| Double-click | Expands or collapses a repeated block |
| Right-click | Node context menu |

On macOS, trackpad scrolling is meant to track your fingers 1:1, with the OS momentum
carrying on panning (expected; not yet verified on hardware). On a mouse, one notch pans 80 px. Starting a pan or zoom stops any running
fly-to animation.

## Keyboard

| Keys | What it does |
|---|---|
| Shift+Up / Shift+Down | Zoom in / out about the centre of the canvas (symmetric 1.1x steps) |
| Ctrl+= / Ctrl+-, or Ctrl+keypad +/- (Cmd instead of Ctrl on macOS) | Same. The keypad keys need the modifier too |
| Shift+Backspace, Ctrl+0 (Cmd+0 on macOS) | Actual size (100%) about the centre |
| Arrow keys | Pan 40 px per press or repeat |
| F | Fit the whole graph |
| Home | Reset the camera to its origin state (zoom 100%, no pan) |
| Esc | Close the search overlay, command palette and utility windows |

Arrow keys and Shift+Up/Down stand aside while the search overlay or the command palette
is open (they step through results there), and no single-key shortcut fires while a text
field has focus. The same zoom and fit actions are in the View menu and the command
palette.

## Scroll wheel preference

**View > Scroll wheel** (also in Preferences > Graph, and as `View: Scroll wheel` in the
command palette) chooses what a plain scroll does:

- **Pan (Netron)**, the default: scrolling pans; Ctrl/Cmd + scroll zooms.
- **Zoom**: scrolling zooms at the pointer, as in NetVis releases before this
  preference existed. A horizontal swipe does nothing in this mode.

The choice is saved in `view_prefs.json` and shared by every tab. Dragging, keys and
click-to-select behave the same in both modes. Anyone who has used NetVis
before (it finds a saved preferences file, recent files, a saved session or a cached
layout) sees a one-time notice about the new default on the first launch after
upgrading. A fresh install does not. Before this release the preferences file was
only written when a setting changed, so its absence alone does not mean a new user.
The first launch of this release records just the wheel choice in that file (so the
notice shows once) and nothing else: every other setting keeps following the
built-in defaults until you change it.

## Compared with Netron

The Netron column is read from the source of `lutzroeder/netron@df0d2df` (`source/view.js`,
`source/app.js`, `source/index.html`), so it can be re-checked.

| Gesture | Netron | NetVis |
|---|---|---|
| Plain wheel | Pans (native scroll); "Mouse Wheel: Zoom" zooms | Pans; Zoom mode zooms |
| Shift + wheel | Zooms | **Pans sideways** |
| Ctrl + wheel | Zooms, at 10x rate | Zooms, 1.1x per notch |
| Cmd + wheel (macOS) | Pans | **Zooms** |
| Trackpad pan / pinch | Pans / zooms at the pointer | Pans / zooms at the pointer (macOS; not yet verified on hardware) |
| Left-drag | Pans, even over nodes, never selects | Same |
| Click | Selects on release | Same |
| Zoom keys | Shift+Up x1.1, Shift+Down x0.9 | Shift+Up x1.1, Shift+Down /1.1 |
| Actual size | Shift+Backspace | Shift+Backspace, Ctrl/Cmd+0 |
| Zoom limits | 0.15 (or fit) to 1.4 | 0.02 to 4.0 |
| Toggle wheel mode | Cmd/Ctrl+M | View menu, Preferences, command palette |

Deliberate differences:

- **Shift + scroll pans sideways** (Netron zooms): this is the platform convention, and on
  macOS Netron's Shift-zoom is effectively a no-op because the OS already turns the delta
  horizontal.
- **Cmd + scroll zooms on macOS** (Netron scrolls): Cmd is the primary macOS modifier.
  Control + scroll zooms too, as in Netron.
- **No 10x Ctrl rate**: Netron's multiplier compensates for the synthetic Ctrl+wheel that
  Chromium sends for a pinch, which GLFW never produces. On a real mouse it would be about
  4x per notch.
- **Wider zoom limits**: the level-of-detail tiers make extreme zoom-out useful on graphs
  with 100k nodes.
- **Symmetric zoom keys**: zooming in and then out returns to where you started (Netron's
  1.1 and 0.9 do not).
- **No Cmd/Ctrl+M**: it is Minimize in the macOS app menu.
- The canvas is unbounded (Netron's scroll stops at the graph's edge). Home and F bring the
  graph back.

## Platform notes

- **macOS**: pinch and 1:1 trackpad scrolling come from a small native bridge (an
  `NSEvent` monitor) that has not yet been tried on real trackpad hardware. If the monitor
  never fires, pinch does nothing and a trackpad pans at the mouse-notch scale. If macOS
  Accessibility Zoom ("Use scroll gesture with modifier keys") is on for Control, the OS
  consumes Control + scroll first; Cmd + scroll still works.
- **Windows**: a precision-touchpad pinch reaches NetVis as Ctrl + scroll and so should
  zoom (expected, not yet verified on hardware).
- **Linux**: there is no pinch yet, because GLFW 3.4 exposes no gesture events. On
  Wayland a touchpad pans at roughly 8x finger speed, since GLFW does not say whether a
  scroll came from a mouse or a touchpad; NetVis does not guess from the size of the delta.
- **Touchscreens** are not supported (GLFW has no touch API).

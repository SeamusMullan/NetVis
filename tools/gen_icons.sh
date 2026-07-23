#!/usr/bin/env bash
# Rasterize assets/netvis.svg into the platform icon files committed under assets/:
#   netvis.ico   — Windows (multi-size, embedded in the .exe via a .rc)
#   netvis.icns  — macOS (.app bundle icon)
#   netvis.png   — 256px, for Linux desktop entries / docs
# Requires rsvg-convert (or magick) + ImageMagick. Run when assets/netvis.svg changes;
# the outputs are committed so CI/package builds need no image toolchain.
set -euo pipefail
cd "$(dirname "$0")/.."
SVG=assets/netvis.svg
OUT=assets
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

render() {  # size outfile
  if command -v rsvg-convert >/dev/null 2>&1; then
    rsvg-convert -w "$1" -h "$1" "$SVG" -o "$2"
  else
    magick -background none -density 384 "$SVG" -resize "${1}x${1}" "$2"
  fi
}

for s in 16 32 48 64 128 256 512; do render "$s" "$TMP/icon_$s.png"; done

cp "$TMP/icon_256.png" "$OUT/netvis.png"

# Windows .ico: pack the common sizes.
magick "$TMP/icon_16.png" "$TMP/icon_32.png" "$TMP/icon_48.png" \
       "$TMP/icon_64.png" "$TMP/icon_128.png" "$TMP/icon_256.png" "$OUT/netvis.ico"

# macOS .icns: ImageMagick writes the Apple icon container directly.
magick "$TMP/icon_16.png" "$TMP/icon_32.png" "$TMP/icon_128.png" \
       "$TMP/icon_256.png" "$TMP/icon_512.png" "$OUT/netvis.icns"

echo "wrote $OUT/netvis.{ico,icns,png}"

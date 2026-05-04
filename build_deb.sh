#!/bin/bash
# build_deb.sh — Build forexmind_1.0.0_amd64.deb
# Usage: ./build_deb.sh
# Requires: fpm  (sudo gem install fpm)
#           pip  (python3 -m pip install build)
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

VERSION="1.0.0"
ARCH="amd64"
NAME="forexmind"
OUT="${NAME}_${VERSION}_${ARCH}.deb"

echo "==> Checking build tools..."

if ! command -v fpm &>/dev/null; then
    echo "ERROR: fpm not found. Install with: sudo gem install fpm"
    exit 1
fi

if ! python3 -c "import build" &>/dev/null; then
    echo "ERROR: python build not found. Install with: pip install build"
    exit 1
fi

echo "==> Building Python wheel..."
rm -rf package/dist
mkdir -p package/dist
python3 -m build --wheel --outdir package/dist .
WHEEL=$(ls package/dist/${NAME}-*.whl | head -1)
echo "    Wheel: $WHEEL"

echo "==> Setting permissions on scripts..."
chmod +x package/postinst.sh package/prerm.sh package/postrm.sh package/forexmind

echo "==> Building .deb with fpm..."
fpm \
    --input-type dir \
    --output-type deb \
    --name "$NAME" \
    --version "$VERSION" \
    --architecture "$ARCH" \
    --description "AI-powered forex trading agent with ML ensemble (Claude + LightGBM + LSTM + PPO)" \
    --maintainer "ForexMind" \
    --depends "python3 (>= 3.11)" \
    --depends "python3-pip" \
    --depends "python3-venv" \
    --depends "ruby" \
    --after-install package/postinst.sh \
    --before-remove package/prerm.sh \
    --after-remove package/postrm.sh \
    --directories /opt/forexmind \
    --directories /var/lib/forexmind \
    --directories /etc/forexmind \
    --package "$OUT" \
    "$WHEEL=/usr/share/forexmind/$(basename $WHEEL)" \
    "forexmind/requirements.txt=/usr/share/forexmind/requirements.txt" \
    "forexmind/config/config.yaml=/usr/share/forexmind/config.yaml" \
    ".env.example=/usr/share/forexmind/.env.example" \
    "package/forexmind=/usr/bin/forexmind" \
    "package/forexmind.service=/etc/systemd/system/forexmind.service"

echo ""
echo "==> Done: $SCRIPT_DIR/$OUT"
echo ""
echo "Install with:"
echo "    sudo dpkg -i $OUT"
echo ""
echo "After install:"
echo "    sudo nano /etc/forexmind/.env   # Add your API keys"
echo "    sudo systemctl start forexmind"
echo "    forexmind cli"
echo ""

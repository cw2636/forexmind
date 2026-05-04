#!/bin/bash
# postinst — runs after dpkg installs all package files
set -e

VENV=/opt/forexmind/venv
SHARE=/usr/share/forexmind
DATA=/var/lib/forexmind
CONFIG=/etc/forexmind

echo "==> ForexMind: setting up..."

# ── 1. System user ─────────────────────────────────────────────────────────────
if ! id -u forexmind &>/dev/null; then
    useradd --system --no-create-home --shell /usr/sbin/nologin forexmind
    echo "    Created system user: forexmind"
fi

# ── 2. Data directories ────────────────────────────────────────────────────────
mkdir -p "$DATA/data" "$DATA/models"
chown -R forexmind:forexmind "$DATA"
echo "    Data directories: $DATA"

# ── 3. Config directory ────────────────────────────────────────────────────────
mkdir -p "$CONFIG"
# Copy default config.yaml only if not already present (preserve user edits)
if [ ! -f "$CONFIG/config.yaml" ]; then
    cp "$SHARE/config.yaml" "$CONFIG/config.yaml"
    echo "    Installed default config: $CONFIG/config.yaml"
fi
# Copy .env only if not already present
if [ ! -f "$CONFIG/.env" ]; then
    cp "$CONFIG/.env.example" "$CONFIG/.env" 2>/dev/null || \
    cp "$SHARE/.env.example"  "$CONFIG/.env" 2>/dev/null || true
    chmod 600 "$CONFIG/.env"
    echo "    Created config: $CONFIG/.env  (edit this with your API keys)"
fi
chown -R root:forexmind "$CONFIG"
chmod 750 "$CONFIG"

# ── 4. Python virtual environment ─────────────────────────────────────────────
mkdir -p /opt/forexmind
if [ ! -d "$VENV" ]; then
    echo "    Creating Python venv at $VENV ..."
    python3 -m venv "$VENV"
fi

echo "    Installing Python dependencies (this may take a few minutes)..."
"$VENV/bin/pip" install --quiet --upgrade pip
"$VENV/bin/pip" install --quiet -r "$SHARE/requirements.txt"

echo "    Installing forexmind package..."
"$VENV/bin/pip" install --quiet --no-deps "$SHARE"/forexmind-*.whl

chown -R forexmind:forexmind /opt/forexmind

# ── 5. Systemd service ─────────────────────────────────────────────────────────
if command -v systemctl &>/dev/null; then
    systemctl daemon-reload
    systemctl enable forexmind.service
    echo "    Service enabled: forexmind.service"
fi

echo ""
echo "==> ForexMind installed successfully!"
echo ""
echo "    Next steps:"
echo "    1. Edit /etc/forexmind/.env with your API keys"
echo "    2. sudo systemctl start forexmind"
echo "    3. forexmind cli     # terminal chat"
echo "       forexmind web     # web at http://localhost:8000"
echo ""

#!/bin/bash
# postrm — runs after dpkg removes package files
# Only purge data/venv on explicit "purge" action
set -e

if [ "$1" = "purge" ]; then
    echo "==> ForexMind: purging data..."
    rm -rf /opt/forexmind
    rm -rf /var/lib/forexmind
    rm -rf /etc/forexmind
    if id -u forexmind &>/dev/null; then
        userdel forexmind 2>/dev/null || true
    fi
    echo "    Purge complete."
fi

#!/bin/bash
# prerm — runs before dpkg removes package files
set -e

if command -v systemctl &>/dev/null; then
    if systemctl is-active --quiet forexmind.service; then
        systemctl stop forexmind.service
    fi
    if systemctl is-enabled --quiet forexmind.service 2>/dev/null; then
        systemctl disable forexmind.service
    fi
    systemctl daemon-reload
fi

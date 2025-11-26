#!/bin/bash
# Fix line endings from CRLF to LF
# Use tr to remove carriage returns, avoiding sed -i permission issues in WSL
tr -d '\r' < start_server.sh > start_server.sh.tmp && mv start_server.sh.tmp start_server.sh
echo "✅ Fixed line endings in start_server.sh"


#!/usr/bin/env bash
set -euo pipefail
source "$OI_INSTALL_DIR/common.sh"
export PATH="/opt/openinspect/node/bin:$PATH"
mkdir -p /opt/openinspect/tools /app/opencode-deps /opt/openinspect/code-server
cp "$OI_BUNDLE/packages/sandbox-images/locks/tools/"package*.json /opt/openinspect/tools/
cd /opt/openinspect/tools
npm ci --ignore-scripts --no-audit --no-fund
# Run only the explicitly reviewed native binary installers. npm's lifecycle defaults differ by substrate.
node node_modules/opencode-ai/postinstall.mjs
node node_modules/bun/install.js
for command in opencode bun bunx pnpm codex pi dsh; do
  ln -sf "/opt/openinspect/tools/node_modules/.bin/$command" "/usr/local/bin/$command"
done
cp "$OI_BUNDLE/packages/sandbox-images/locks/plugins/"package*.json /app/opencode-deps/
(cd /app/opencode-deps && npm ci --ignore-scripts --no-audit --no-fund)
download_dir="$(mktemp -d /tmp/openinspect-tools.XXXXXX)"
trap 'rm -rf "$download_dir"' EXIT
# ZCode has no npm distribution: build the agent CLI from the SHA-pinned source
# tarball. Only the CLI workspace's dependency graph is installed (the desktop
# and web packages stay out), and install scripts stay disabled like every npm
# install in this image.
download_checked "https://github.com/zai-org/ZCode/archive/refs/tags/v$ZCODE_VERSION.tar.gz" "$ZCODE_SHA256" "$download_dir/zcode.tar.gz"
mkdir -p /opt/openinspect/zcode
tar -xzf "$download_dir/zcode.tar.gz" -C /opt/openinspect/zcode --strip-components=1
(
  cd /opt/openinspect/zcode
  pnpm install --filter @zcode/cli... --frozen-lockfile --ignore-scripts --no-fund --store-dir /tmp/openinspect-pnpm-store
  pnpm --filter @zcode/cli... build
)
printf '#!/bin/sh\nexec node /opt/openinspect/zcode/apps/zcode-cli/packages/cli/dist/zcode.cjs "$@"\n' > /usr/local/bin/zcode
chmod 0755 /usr/local/bin/zcode
download_checked "https://github.com/vercel-labs/agent-browser/releases/download/v$AGENT_BROWSER_VERSION/agent-browser-linux-x64" "$AGENT_BROWSER_SHA256" "$download_dir/agent-browser"
install -m 0755 "$download_dir/agent-browser" /usr/local/bin/agent-browser
download_checked "https://github.com/coder/code-server/releases/download/v$CODE_SERVER_VERSION/code-server-$CODE_SERVER_VERSION-linux-amd64.tar.gz" "$CODE_SERVER_SHA256" "$download_dir/code-server.tar.gz"
tar -xzf "$download_dir/code-server.tar.gz" -C /opt/openinspect/code-server --strip-components=1
ln -sf /opt/openinspect/code-server/bin/code-server /usr/local/bin/code-server
download_checked "https://github.com/tsl0922/ttyd/releases/download/$TTYD_VERSION/ttyd.x86_64" "$TTYD_SHA256" "$download_dir/ttyd"
install -m 0755 "$download_dir/ttyd" /usr/local/bin/ttyd
download_checked "https://storage.googleapis.com/chrome-for-testing-public/$CHROME_VERSION/linux64/chrome-linux64.zip" "$CHROME_SHA256" "$download_dir/chrome.zip"
mkdir -p /opt/openinspect/chrome
unzip -q "$download_dir/chrome.zip" -d /opt/openinspect/chrome
ln -sf /opt/openinspect/chrome/chrome-linux64/chrome /usr/local/bin/google-chrome

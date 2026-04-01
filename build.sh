#!/usr/bin/env bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
source "$HOME/.cargo/env"
pip install -r requirements.txt
```

Then in Render **Settings → Build Command**, change it to:
```
chmod +x build.sh && ./build.sh

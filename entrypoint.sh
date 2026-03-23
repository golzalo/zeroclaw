#!/bin/sh

set -eu

CONFIG_DIR="${ZEROCLAW_CONFIG_DIR:-/zeroclaw-data/.zeroclaw}"
CONFIG_FILE="${CONFIG_DIR}/config.toml"
BASE_CONFIG_FILE="${ZEROCLAW_BASE_CONFIG_FILE:-/usr/local/share/zeroclaw/config.base.toml}"
SESSION_PATH="${ZEROCLAW_WHATSAPP_SESSION_PATH:-/zeroclaw-data/workspace/state/whatsapp-web/session.db}"
PAIR_PHONE_RAW="${ZEROCLAW_WHATSAPP_PAIR_PHONE:-}"

mkdir -p "${CONFIG_DIR}" "$(dirname "${SESSION_PATH}")"
cp "${BASE_CONFIG_FILE}" "${CONFIG_FILE}"
chmod 600 "${CONFIG_FILE}"

PAIR_PHONE_DIGITS="$(printf '%s' "${PAIR_PHONE_RAW}" | tr -cd '0-9')"

if [ -n "${PAIR_PHONE_DIGITS}" ]; then
  cat >> "${CONFIG_FILE}" <<EOF

[channels_config.whatsapp]
session_path = "${SESSION_PATH}"
pair_phone = "${PAIR_PHONE_DIGITS}"
allowed_numbers = ["+${PAIR_PHONE_DIGITS}"]
allow_self_chat = true
allow_direct_messages = false
allow_group_messages = false
EOF
fi

echo "--- v0.5.2 Self-Chat Only Config ---"
exec /usr/local/bin/zeroclaw "$@"

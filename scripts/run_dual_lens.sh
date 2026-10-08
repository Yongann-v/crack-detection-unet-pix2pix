#!/usr/bin/env bash
# Run Insta360 crack detection on BOTH lenses, original and undistorted, side by side.
#
#   One insta360_publisher feeds four crack_detection_node instances:
#     front_raw, front_undist, back_raw, back_undist
#   Each publishes on its own topics (/<tag>/visualization, /<tag>/detected, /<tag>/result, ...)
#   and saves captures to $WS/crack_captures/<tag>/.
#   A 2x2 window (left original, right undistorted) shows the camera with crack overlay,
#   status, FPS, crack %, temporal count and detection events.
#   Press q in the window (or Ctrl+C here) to stop everything.
#
# Examples:
#   scripts/run_dual_lens.sh                  # UNet + Pix2Pix, threshold 0.2
#   scripts/run_dual_lens.sh -t 0.5           # threshold 0.5
#   scripts/run_dual_lens.sh -m unet          # UNet only (faster)

set -euo pipefail

THRESHOLD=0.2
MODE=pix2pix
BALANCE=0.5
MIN_CRACK_PERCENT=2.0
DRY_RUN=false

usage() {
    sed -n '2,15p' "$0" | sed 's/^# \{0,1\}//'
    cat <<EOF

Options:
  -t THRESHOLD   Crack probability threshold, 0-1 (default $THRESHOLD)
  -m MODE        unet | pix2pix (default $MODE)
  -b BALANCE     Undistortion balance, 0-1 (default $BALANCE)
  -p PERCENT     Min crack % to count as a detection (default $MIN_CRACK_PERCENT)
  -d             Dry run: print the node commands without running them
  -h             Show this help
EOF
}

die() { echo "Error: $*" >&2; exit 1; }

in_unit_range() { awk -v v="$1" 'BEGIN { exit !(v ~ /^[0-9]*\.?[0-9]+$/ && v >= 0 && v <= 1) }'; }

while getopts "t:m:b:p:dh" opt; do
    case $opt in
        t) THRESHOLD=$OPTARG ;;
        m) MODE=$OPTARG ;;
        b) BALANCE=$OPTARG ;;
        p) MIN_CRACK_PERCENT=$OPTARG ;;
        d) DRY_RUN=true ;;
        h) usage; exit 0 ;;
        *) usage >&2; exit 1 ;;
    esac
done

in_unit_range "$THRESHOLD" || die "threshold must be between 0 and 1, got '$THRESHOLD'"
in_unit_range "$BALANCE" || die "balance must be between 0 and 1, got '$BALANCE'"
[[ $MODE == unet || $MODE == pix2pix ]] || die "mode must be 'unet' or 'pix2pix', got '$MODE'"

# Workspace = three levels up from this script (crack_ws/src/<repo>/scripts), unless CRACK_WS is set
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
WS=${CRACK_WS:-$(cd "$SCRIPT_DIR/../../.." && pwd)}
[[ -f $WS/install/setup.bash ]] || die "no install/setup.bash in $WS (build the workspace or set CRACK_WS)"

# ROS setup scripts reference unset variables
set +u
source /opt/ros/jazzy/setup.bash
source "$WS/install/setup.bash"
set -u

SHARE=$(ros2 pkg prefix crack_detection)/share/crack_detection
# Large shared-memory segment so the ~5 MB raw visualization frames reach local subscribers
export FASTRTPS_DEFAULT_PROFILES_FILE=$SHARE/config/fastdds_large_images.xml

use_pix2pix=false
model="UNet"
[[ $MODE == pix2pix ]] && use_pix2pix=true && model="UNet + Pix2Pix"

node_cmd() {  # node_cmd LENS UNDISTORT TAG
    local lens=$1 undistort=$2 tag=$3
    cmd=(ros2 run crack_detection crack_detection_node --ros-args
         -r __node:=crack_$tag
         --params-file "$SHARE/config/crack_detection_params.yaml"
         -p camera_topic:=/insta360/$lens/image_raw
         -p calibration_file:=calibration_data/insta360_oner_$lens.yaml
         -p undistort_enabled:=$undistort
         -p undistort_balance:=$BALANCE
         -p threshold:=$THRESHOLD
         -p min_crack_percent:=$MIN_CRACK_PERCENT
         -p use_pix2pix:=$use_pix2pix
         -p model_path:=models/best_model.pth
         -p pix2pix_model_path:=models/pix2pix_epoch_98_best.pth
         -p input_size:=384 -p window_size:=384 -p subdivisions:=2
         -p publish_visualization:=true
         -p save_directory:=crack_captures/$tag)
    # Node topics are absolute, so give each instance its own by remapping
    local topic
    for topic in visualization visualization/compressed result detected center_pixel robot_pose; do
        cmd+=(-r /crack_detection/$topic:=/$tag/$topic)
    done
    cmd+=(-r /visualization_marker:=/$tag/marker)
}

TAGS=()
for lens in front back; do
    for undistort in false true; do
        TAGS+=("$lens $undistort ${lens}_$([[ $undistort == true ]] && echo undist || echo raw)")
    done
done

echo "Dual lens: $model, threshold $THRESHOLD, min crack $MIN_CRACK_PERCENT%, balance $BALANCE"
if [[ $DRY_RUN == true ]]; then
    for t in "${TAGS[@]}"; do node_cmd $t; echo "${cmd[*]}"; echo; done
    exit 0
fi

pgrep -f "lib/crack_detection/[c]rack_detection_node" >/dev/null && \
    die "crack_detection_node is already running; stop it first (pkill -f crack_detection_node)"
pgrep -f "[a]b_undistort_test|[c]apture_calibration" >/dev/null && \
    die "the camera is in use by an A/B test or calibration capture; close it first"

cd "$WS"   # models/ paths and crack_captures/ are relative to the workspace
pids=()
cleanup() {
    trap - EXIT INT TERM
    echo "Stopping..."
    local pid
    for pid in "${pids[@]}"; do
        # ros2 run does not pass SIGTERM on, so stop the node it spawned as well
        pkill -INT -P "$pid" 2>/dev/null || true
        kill -INT "$pid" 2>/dev/null || true
    done
    wait 2>/dev/null || true
}
trap cleanup EXIT INT TERM

if pgrep -f "[i]nsta360_publisher" >/dev/null; then
    echo "Using the insta360_publisher that is already running"
else
    ros2 run insta360_ros insta360_publisher &
    pids+=($!)
    sleep 3
fi

for t in "${TAGS[@]}"; do
    node_cmd $t
    "${cmd[@]}" &
    pids+=($!)
done

python3 "$SCRIPT_DIR/dual_lens_view.py" --threshold "$THRESHOLD" \
    --min-crack-percent "$MIN_CRACK_PERCENT" --model "$model"

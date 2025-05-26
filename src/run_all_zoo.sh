# ----------------------------------------------------------
# Toggle self-attention with   ATTENTION=1 ./run_all_zoo.sh
# ----------------------------------------------------------
ENV_NAME="AntBulletEnv-v0"
OBS_REPEAT=32
OBS_NOISE=0
ATTENTION=0  # 1 or 0

# then your if-condition as is
if [[ "$ATTENTION" -eq 1 ]]; then
  CONFIG_FILE="$(pwd)/hyperparams/AntBulletEnv-v0-attn.yml"
  EXTRA_TAG="attention"
  ATTN_SUFFIX="-attn"
else
  CONFIG_FILE="$(pwd)/hyperparams/AntBulletEnv-v0.yml"
  EXTRA_TAG=""
  ATTN_SUFFIX=""
fi

# Debug print:
echo "ATTENTION is: $ATTENTION"
echo "CONFIG_FILE: $CONFIG_FILE"
echo "EXTRA_TAG: $EXTRA_TAG"
echo "ATTN_SUFFIX: $ATTN_SUFFIX"


BASE_DIR="runs"
mkdir -p "$BASE_DIR"

# EXP_GROUP="${ENV_NAME}-zoo-repeat${OBS_REPEAT}-noise${OBS_NOISE}"
EXP_GROUP="${ENV_NAME}-zoo${ATTN_SUFFIX}-repeat${OBS_REPEAT}-noise${OBS_NOISE}"
EXP_GROUP_DIR="$BASE_DIR/$EXP_GROUP"
mkdir -p "$EXP_GROUP_DIR"

i=1
while [ -d "$EXP_GROUP_DIR/run$i" ]; do
  i=$((i+1))
done
RUN_BATCH_DIR="$EXP_GROUP_DIR/run$i"
mkdir -p "$RUN_BATCH_DIR"

echo "Using run batch directory: $RUN_BATCH_DIR"

for seed in 0 1 2; do
  SEED_DIR="$RUN_BATCH_DIR/seed$seed"
  mkdir -p "$SEED_DIR"
  
  TB_LOG_DIR="$SEED_DIR/tb_logs"
  mkdir -p "$TB_LOG_DIR"
  mkdir -p "$SEED_DIR/wandb"

  echo "Launching seed $seed. TensorBoard logs: $TB_LOG_DIR"
  ANT_OBS_REPEAT=$OBS_REPEAT ANT_OBS_NOISE=$OBS_NOISE \
  WANDB_DIR="$SEED_DIR" \
    python -m rl_zoo3.train \
    --algo ppo \
    --env $ENV_NAME \
    --conf-file "$CONFIG_FILE" \
    --seed $seed \
    -f "$SEED_DIR" \
    -tb "$TB_LOG_DIR" \
    --track \
    --wandb-project-name "adlr-01" \
    --wandb-entity "adlr-01" \
    --wandb-tags "${ENV_NAME},ppo,obs_repeat${OBS_REPEAT},noise${OBS_NOISE},${EXTRA_TAG},seed${seed}" &
    # --wandb-tags "${ENV_NAME},ppo,obs_repeat${OBS_REPEAT},noise${OBS_NOISE},seed${seed}" &
    # --capture-video  # Uncomment if needed
done

wait
echo "All seeds finished."
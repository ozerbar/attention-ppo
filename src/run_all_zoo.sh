ENV_NAME="AntBulletEnv-v0"
CONFIG_FILE="$(pwd)/hyperparams/AntBulletEnv-v0.yml"
OBS_REPEAT=4
OBS_NOISE=0

BASE_DIR="runs"
mkdir -p "$BASE_DIR"

EXP_GROUP="${ENV_NAME}-zoo-repeat${OBS_REPEAT}-noise${OBS_NOISE}"
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
    --wandb-tags "${ENV_NAME},ppo,obs_repeat${OBS_REPEAT},noise${OBS_NOISE},seed${seed}" &
    # --capture-video  # Uncomment if needed
done

wait
echo "All seeds finished."
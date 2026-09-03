set -euo pipefail

# bash scripts/ours-without-roll-hcs/resnet50/within-dataset/eve.sh
# bash scripts/ours-without-roll-hcs/resnet50/within-dataset/gazecapture.sh
# bash scripts/ours-without-roll-hcs/resnet50/within-dataset/mpiifacegaze.sh

bash scripts/ours-without-roll-hcs/resnet18/within-dataset/eve.sh
bash scripts/ours-without-roll-hcs/resnet18/within-dataset/xgaze.sh
bash scripts/ours-without-roll-hcs/resnet50/within-dataset/xgaze.sh
bash scripts/ours-without-roll-hcs/resnet18/within-dataset/gazecapture.sh
bash scripts/ours-without-roll-hcs/resnet18/within-dataset/mpiifacegaze.sh

bash scripts/ours-without-roll-ccs/resnet50/within-dataset/xgaze.sh
bash scripts/ours-without-roll-ccs/resnet50/within-dataset/eve.sh
bash scripts/ours-without-roll-ccs/resnet50/within-dataset/gazecapture.sh
bash scripts/ours-without-roll-ccs/resnet50/within-dataset/mpiifacegaze.sh

bash scripts/ours-without-roll-ccs/resnet18/within-dataset/xgaze.sh
bash scripts/ours-without-roll-ccs/resnet18/within-dataset/eve.sh
bash scripts/ours-without-roll-ccs/resnet18/within-dataset/gazecapture.sh
bash scripts/ours-without-roll-ccs/resnet18/within-dataset/mpiifacegaze.sh
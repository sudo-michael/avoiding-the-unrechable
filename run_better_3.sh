#!/bin/bash
python atu/hj_sac.py --use-hj --reward-shape --track --reward-shape-penalty 10 \
    --train-dist "0.1,0.1,0.1,0.1,0.1" --eval-dist  "0.1,0.1,0.1,0.1,0.1" --env-id Safe-SingleNarrowPassage-v0 \
    --capture-video

python atu/hj_sac.py --use-hj --reward-shape --track --reward-shape-penalty 10 \
    --train-dist "0.1,0.1,0.1,0.1,0.1" --eval-dist  "0.1,0.1,0.1,0.1,0.1" --env-id Safe-SingleNarrowPassage-v0 \
    --capture-video --gamma 0.9

python atu/hj_sac.py --use-hj --reward-shape --track --reward-shape-penalty 10 \
    --train-dist "0.1,0.1,0.1,0.1,0.1" --eval-dist  "0.1,0.1,0.1,0.1,0.1" --env-id Safe-SingleNarrowPassage-v0 \
    --capture-video --gamma 0.9 --batch-size 512
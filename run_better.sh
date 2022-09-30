#!bin/bash
# python atu/hj_sac.py --use-hj --track --reward-shape-penalty 10 --seed 0 --train-dist "0.1,0.1,0.1" --eval-dist  "0.1,0.1,0.1" --env-id Safe-DubinsHallway-v2 --capture-video
# python atu/hj_sac.py --use-hj --track --reward-shape-penalty 10 --seed 0 --train-dist "0.1,0.1,0.1" --eval-dist  "0.1,0.1,0.1" --env-id Safe-DubinsHallway-v2 --capture-video --gamma 0.9
# python atu/hj_sac.py --use-hj --track --reward-shape-penalty 10 --seed 0 --train-dist "0.1,0.1,0.1" --eval-dist  "0.1,0.1,0.1" --env-id Safe-DubinsHallway-v2 --capture-video --gamma 0.9 --batch-size 512 --reward-shape
python atu/hj_sac.py --use-hj --track --reward-shape-penalty 10 --seed 0 --train-dist "0.1,0.1,0.1" --eval-dist  "0.1,0.1,0.1" --env-id Safe-DubinsHallway-v2 --capture-video --gamma 0.9 --batch-size 512  --reward-shape

#!/bin/bash 


for i in {1..5}
do
    python atu/hj_sac.py --use-hj --capture-video --reward-shape --track --reward-shape-penalty 10 --seed $i --group-name dist-chage --eval-dist 0.2 --env-id Safe-DubinsHallway4D-v0
done

for i in {1..5}
do
    python atu/hj_sac.py --use-hj --capture-video --reward-shape --track --reward-shape-penalty 10 --seed $i --group-name dist-chage --eval-dist 0.3 --env-id Safe-DubinsHallway4D-v0
done

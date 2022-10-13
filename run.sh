#!/bin/bash 

# # 3D
# for i in {1..5}
# do
#     python atu/hj_sac.py --use-hj --reward-shape --track --reward-shape-penalty 10 --seed $i --group-name reward-shape --train-dist "0.1,0.1,0.1" --eval-dist  "0.1,0.1,0.1" --env-id Safe-DubinsHallway-v1
# done

# for i in {1..5}
# do
#     python atu/hj_sac.py --use-hj --reward-shape --track --reward-shape-penalty 10 --seed $i --group-name reward-shape --train-dist "0.1,0.1,0.1" --eval-dist  "0.2,0.2,0.2" --env-id Safe-DubinsHallway-v1
# done

# for i in {1..5}
# do
#     python atu/hj_sac.py --use-hj --reward-shape --track --reward-shape-penalty 10 --seed $i --group-name reward-shape --train-dist "0.1,0.1,0.1" --eval-dist  "0.3,0.3,0.3" --env-id Safe-DubinsHallway-v1
# done

# # 4D
# for i in {1..5}
# do
#     python atu/hj_sac.py --use-hj --reward-shape --track --reward-shape-penalty 10 --seed $i --group-name reward-shape --train-dist "0.1,0.1,0.1,0.1" --eval-dist  "0.1,0.1,0.1,0.1" --env-id Safe-DubinsHallway4D-v0
# done

# for i in {1..5}
# do
#     python atu/hj_sac.py --use-hj --reward-shape --track --reward-shape-penalty 10 --seed $i --group-name reward-shape --train-dist "0.1,0.1,0.1,0.1" --eval-dist  "0.2,0.2,0.2,0.2" --env-id Safe-DubinsHallway4D-v0
# done

# for i in {1..5}
# do
#     python atu/hj_sac.py --use-hj --reward-shape --track --reward-shape-penalty 10 --seed $i --group-name reward-shape --train-dist "0.1,0.1,0.1,0.1" --eval-dist  "0.3,0.3,0.3,0.3" --env-id Safe-DubinsHallway4D-v0
# done

# 5D
for i in {1..5}
do
    python atu/hj_sac.py --use-hj --reward-shape --track --reward-shape-penalty 10 --seed $i --group-name reward-shape --train-dist "0.1,0.1,0.1,0.1,0.1" --eval-dist  "0.1,0.1,0.1,0.1,0.1" --env-id Safe-SingleNarrowPassage-v0
done

for i in {1..5}
do
    python atu/hj_sac.py --use-hj --reward-shape --track --reward-shape-penalty 10 --seed $i --group-name reward-shape --train-dist "0.1,0.1,0.1,0.1,0.1" --eval-dist  "0.2,0.2,0.2,0.2,0.2" --env-id Safe-SingleNarrowPassage-v0
done

for i in {1..5}
do
    python atu/hj_sac.py --use-hj --reward-shape --track --reward-shape-penalty 10 --seed $i --group-name reward-shape --train-dist "0.1,0.1,0.1,0.1,0.1" --eval-dist  "0.3,0.3,0.3,0.3,0.3" --env-id Safe-SingleNarrowPassage-v0
done
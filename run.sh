#!/bin/bash 

# python atu/sac.py --track --use-hj --seed 0
# python atu/sac.py --track --use-hj --seed 1
# python atu/sac.py --track --use-hj --seed 2

# python atu/sac.py --track --use-hj --seed 0 --reward-shape
# python atu/sac.py --track --use-hj --seed 1 --reward-shape
# python atu/sac.py --track --use-hj --seed 2 --reward-shape

# python atu/sac.py --track --use-hj --seed 0 --reward-shape --done-if-unsafe
# python atu/sac.py --track --use-hj --seed 1 --reward-shape --done-if-unsafe
# python atu/sac.py --track --use-hj --seed 2 --reward-shape --done-if-unsafe

# python atu/sac.py --track --use-hj --seed 0 --reward-shape --done-if-unsafe --sample-uniform
# python atu/sac.py --track --use-hj --seed 1 --reward-shape --done-if-unsafe --sample-uniform
# python atu/sac.py --track --use-hj --seed 2 --reward-shape --done-if-unsafe --sample-uniform

# python atu/sac.py --track --use-hj --seed 0 --done-if-unsafe --sample-uniform
# python atu/sac.py --track --use-hj --seed 1 --done-if-unsafe --sample-uniform
# python atu/sac.py --track --use-hj --seed 2 -done-if-unsafe --sample-uniform

# python atu/sac.py --track --use-hj --imagine-trajectory --capture-video
# python atu/sac.py --track --use-hj --imagine-trajectory --reward shape --capture-video
# python atu/sac.py --track --use-hj --imagine-trajectory --sample-uniform --capture-video
# python atu/sac.py --track --use-hj --imagine-trajectory --sample-uniform --reward shape --capture-video
# python atu/sac.py --track --use-hj --dist --haco --capture-video
# python atu/sac.py --track --use-hj --dist --haco --capture-video --sample-uniform
# python atu/sac.py --track --use-hj --dist --haco --capture-video --sample-uniform --reward-shape
# python atu/sac.py --track --use-hj --dist --capture-video --sample-uniform --reward-shape
# for i in {1..3}
# do
#     echo "use hj reward shape run with gradvdotf $i"
#     python atu/hj_sac.py --track --use-hj --reward-shape --group-name hjrs-gradvdotf-20-2  --reward-shape-penalty=20  --seed $i
# done

# for i in {1..3}
# do
#     echo "use hj reward shape run with gradvdotf $i"
#     python atu/hj_sac.py --track --use-hj --reward-shape --group-name hjrs-gradvdotf-2  --seed $i
# done

# for i in {1..3}
# do
#     echo "use hj reward shape run with gradvdotf $i"
#     python atu/hj_sac.py --track --use-hj --reward-shape --group-name hjrs-gradvdotf-2-2  --reward-shape-penalty=2 --seed $i
# done

# for i in {1..3}
# do
#     echo "use hj reward shape run with gradvdotf $i"
#     python atu/hj_sac.py --track --use-hj --reward-shape --group-name 2timesminreward --reward-shape-penalty=0 --seed $i
# done

# for i in {1..5}
# do
#     echo "sac lag $1"
#     python atu/lag_sac.py --track --group-name lag_sac_2  --seed $i
# done

# for i in {1..5}
# do
#     echo "hj rs $1"
#     python atu/sac.py --track --group-name hjrs_3  --use-hj --reward-shape --seed $i 
#     python atu/sac.py --track --group-name hjrs_3  --use-hj --reward-shape --seed $i 
# done

# python atu/hj_sac.py --track --use-hj --reward-shape --reward-shape-gradv --total-timesteps 150_000 --gamma 0.9
# python atu/hj_sac.py --track --use-hj --reward-shape --reward-shape-penalty 5 --group-name rsp-fake --fake-next-obs
# python atu/hj_sac.py --track --use-hj --reward-shape --reward-shape-penalty 10 --group-name rsp-fake --fake-next-obs
# python atu/hj_sac.py --track --use-hj --reward-shape --reward-shape-penalty 15 --group-name rsp-fake --fake-next-obs
# python atu/hj_sac.py --track --use-hj --reward-shape --reward-shape-penalty 20 --group-name rsp-fake --fake-next-obs
# python atu/hj_sac.py --track --use-hj --reward-shape --reward-shape-penalty 25 --group-name rsp-fake --fake-next-obs
# python atu/hj_sac.py --track --use-hj --reward-shape --reward-shape-penalty 40 --group-name rsp-fake --fake-next-obs
# python atu/hj_sac.py --track --use-hj --reward-shape --reward-shape-penalty 100 --group-name rsp-fake --fake-next-obs

# python atu/hj_sac.py --track --use-hj --reward-shape --reward-shape-penalty 5 --group-name rsp
# python atu/hj_sac.py --track --use-hj --reward-shape --reward-shape-penalty 10 --group-name rsp
# python atu/hj_sac.py --track --use-hj --reward-shape --reward-shape-penalty 15 --group-name rsp
# python atu/hj_sac.py --track --use-hj --reward-shape --reward-shape-penalty 20 --group-name rsp
# python atu/hj_sac.py --track --use-hj --reward-shape --reward-shape-penalty 25 --group-name rsp
# python atu/hj_sac.py --track --use-hj --reward-shape --reward-shape-penalty 40 --group-name rsp
# python atu/hj_sac.py --track --use-hj --reward-shape --reward-shape-penalty 100 --group-name rsp
# python atu/hj_sac.py --track --use-hj --reward-shape --reward-shape-penalty 10 --group-name rsp-fake --fake-next-obs
# python atu/hj_sac.py --track --use-hj --reward-shape --reward-shape-penalty 150 --group-name rsp
# python atu/hj_sac.py --track --use-hj --reward-shape --reward-shape-penalty 200 --group-name rsp
# python atu/hj_sac.py --track --use-hj --reward-shape --reward-shape-penalty 250 --group-name rsp
# python atu/hj_sac.py --track --use-hj --reward-shape --reward-shape-penalty 500 --group-name rsp
# python atu/hj_sac.py --track --use-hj --reward-shape --reward-shape-gradv --reward-shape-gradv-takeover -0.1 --group-name gvr
# python atu/hj_sac.py --track --use-hj --reward-shape --reward-shape-gradv --reward-shape-gradv-takeover -0.2 --group-name gvr
# python atu/hj_sac.py --track --use-hj --reward-shape --reward-shape-gradv --reward-shape-gradv-takeover -0.3 --group-name gvr
python atu/hj_sac.py --track --use-hj --reward-shape --reward-shape-gradv --reward-shape-gradv-takeover -0.6 --group-name gvr
python atu/hj_sac.py --track --use-hj --reward-shape --reward-shape-gradv --reward-shape-gradv-takeover -0.7 --group-name gvr
python atu/hj_sac.py --track --use-hj --reward-shape --reward-shape-gradv --reward-shape-gradv-takeover -0.8 --group-name gvr


for i in {1..3}
do
    python atu/hj_sac.py --track --use-hj --reward-shape --reward-shape-gradv --eval-dist 0.12 --group-name change-eval-dist
done

for i in {1..3}
do
    python atu/hj_sac.py --track --use-hj --reward-shape --reward-shape-gradv --eval-speed 1.1 --group-name change-eval-dist
done

for i in {1..3}
do
    python atu/hj_sac.py --track --use-hj --reward-shape --reward-shape-gradv --eval-speed 1.1 --eval-dist 0.12 --group-name change-eval-dist
done

for i in {1..3}
do
    python atu/hj_sac.py --track --use-hj --reward-shape --reward-shape-gradv --train-dist 0.0 --eval-dist 0.1 --group-name change-eval-dist
done
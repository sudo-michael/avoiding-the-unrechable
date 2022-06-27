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
for i in {1..5}
do
    echo "use hj reward shape run $i"
    python atu/sac.py --track --use-hj --reward-shape --group-name test-group2  --seed $i
done
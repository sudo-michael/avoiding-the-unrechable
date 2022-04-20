for i in {0..4} 
do
    python sac.py --track --use-hj --reward-shape --seed $i
done

for i in {0..4} 
do
    python sac.py --track --use-hj --seed $i
done

for i in {0..4} 
do
    python sac.py --track --capture-video --use-hj --reward-shape --seed $i
done
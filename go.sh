# python main.py --cuda --alpha=0.1 --hidden_size=128 --eval=True
# python main.py --cuda --alpha=0.1 --hidden_size=128 --eval=True --batch_size=512 --num_steps=100000
# python main.py --cuda --alpha=0.1 --hidden_size=128 --eval=True --batch_size=512 --num_steps=100000 --gamma=0.9
# python main.py --cuda --alpha=0.1 --hidden_size=256 --eval=True --batch_size=512 --num_steps=100000 --gamma=0.9 --updates_per_step=10
# python main.py --cuda --alpha=0.1 --hidden_size=256 --eval=True --batch_size=512 --num_steps=100000 --gamma=0.9 --updates_per_step=10 --automatic_entropy_tuning=True
# python main.py --cuda --alpha=0.5 --hidden_size=256 --eval=True --batch_size=512 --num_steps=100000 --gamma=0.9 --updates_per_step=1 --automatic_entropy_tuning=True
# python main.py --cuda --alpha=0.5 --hidden_size=256 --eval=True --batch_size=512 --num_steps=100000 --gamma=0.9 --updates_per_step=1 --automatic_entropy_tuning=True
# python main.py --cuda --env-name=dubins3d-sparse-v0 --alpha=0.5 --hidden_size=256 --eval=True --batch_size=512 --num_steps=100000 --gamma=0.9 --updates_per_step=1 --automatic_entropy_tuning=True
# python hj_dqn.py --gym-id dubins3d-discrete-v0 --learning-rate 0.001 --total-timesteps 500000 --batch-size 512 --save_model True --save_freq 100000
# python hj_dqn.py --gym-id dubins3d-discrete-v0 --learning-rate 0.001 --total-timesteps 700000 --batch-size 1024 --save_model True --save_freq 100000 
# python hj_dqn.py --gym-id dubins3d-discrete-v0 --learning-rate 0.001 --total-timesteps 700000 --batch-size 2048 --save_model True
# python hj_dqn.py --gym-id dubins3d-discrete-v0 --learning-rate 0.003 --total-timesteps 700000 --batch-size 2048 --save_model True
# DQN isn't working well
# python hj_dqn.py --gym-id dubins3d-discrete-v0 --learning-rate 0.003 --total-timesteps 700000 --batch-size 64 --save_model False
# python hj_sac.py --gym-id dubins3d-v0 --total-timesteps 200000
# python hj_sac.py --gym-id dubins3d-v0 --total-timesteps 10000 --eval True --checkpoint checkpoints/dubins3d-v0__hj_sac__2__1638915172/sac_checkpoint_60000
# python hj_sac.py --gym-id dubins3d-v0 --total-timesteps 15000 
# python hj_sac.py --gym-id dubins3d-v0 --eval --checkpoint checkpoints/dubins3d-v0__hj_sac__2__1638986185/sac_checkpoint_50000
# python hj_sac.py --gym-id dubins3d-v0 --eval --checkpoint checkpoints/dubins3d-v0__hj_sac__2__1638986185/sac_checkpoint_50000
# python hj_sac.py --gym-id dubins3d-v0 --total-timesteps 15000  --use-hj False
# python hj_sac.py --gym-id dubins3d-v0 --total-timesteps 15000  --use-hj True
# python hj_load.py --gym-id dubins3d-v0 --checkpoint checkpoints/dubins3d-v0__hj_sac__2__1639168570/sac_checkpoint_15000 --use-hj True
# python hj_sac.py --gym-id dubins3d-v0 --total-timesteps 15000 --use-hj True
# python hj_load.py --gym-id dubins3d-v0 --checkpoint checkpoints/dubins3d-v0__hj_sac__2__1639168570/sac_checkpoint_15000 --use-hj True
# python hj_load.py --gym-id dubins3d-v0 --checkpoint checkpoints/dubins3d-v0__hj_sac__2__1639168907/sac_checkpoint_15000 --use-hj True
# python hj_sac.py --gym-id dubins3d-v0 --total-timesteps 15000 --use-hj False --seed 0
# python hj_sac.py --gym-id dubins3d-v0 --total-timesteps 15000 --use-hj True --seed 0
# python hj_load.py --gym-id dubins3d-v0 --seed 0  --checkpoint checkpoints/dubins3d-v0__hj_sac__1639169471__1639169471/sac_checkpoint_15000 --use-hj True
# python hj_load.py --gym-id dubins3d-v0 --seed 0  --checkpoint checkpoints/dubins3d-v0__hj_sac__1639169375__1639169375/sac_checkpoint_15000 --use-hj False
# least restrictive cost
# python hj_load.py --gym-id dubins3d-v0 --seed 0  --checkpoint checkpoints/dubins3d-v0__hj_sac__2__1639172674/sac_checkpoint_15000 --use-hj True

# python hj_sac.py --gym-id dubins3d-v0 --total-timesteps 30000 --use-hj True --seed 0
# python hj_sac.py --gym-id dubins3d-v0 --total-timesteps 30000 --use-hj True --seed 0 --checkpoint checkpoints/dubins3d-v0__hj_sac__1644955962__1644955962/sac_checkpoint_30000 --eval True

# goal is to increase rate of success
# python hj_sac.py --gym-id dubins3d-v0 --total-timesteps 25000 --use-hj True --seed 0  --use-hirl True
# python hj_sac.py --gym-id dubins3d-v0 --total-timesteps 30000 --use-hj True --seed 0  --use-hirl True --hirl-penalty 1000
python hj_sac.py --gym-id dubins3d-v0 --total-timesteps 30000 --use-hj True --seed 0  --use-hirl True --hirl-penalty 500
python hj_sac.py --gym-id dubins3d-v0 --total-timesteps 30000 --use-hj True --seed 0  --use-hirl True --hirl-penalty 250
python hj_sac.py --gym-id dubins3d-v0 --total-timesteps 30000 --use-hj True --seed 0  --use-hirl True --hirl-penalty 100

# dubins3d-v0__hj_sac__1644976647__1644976647
# dubins3d-v0__hj_sac__1644976986__1644976986
# dubins3d-v0__hj_sac__1644977305__1644977305
# dubins3d-v0__hj_sac__1644977654__1644977654
import wandb
import numpy as np
api = wandb.Api()
r  = 'mlu/atu2/oslvt6h3'
run = api.run(r)
df = run.history(keys=['charts/gradVdotFxu'])
# print((df['charts/gradVdotFxu'] > -0.5).mean())
print((df['charts/gradVdotFxu']).mean())
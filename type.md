# Summary
The main problem with the results so far was that when using
reward shaping with fixed cost, reward shaping with gradVdotFxu, and the baseline
it was not clear which method was better in terms of how safe the agent performed 
(measured by the value of the BRT at collision) and how often the agent crashed after 
the safety controller was removed.

One reason could have been that the performance of the agents during training.
After looking at the return during today's meeting, I realized that the performance of the agents
didn't really increase overtime. This made it hard to evalite the peformance of the methods since 
the policy was never that good in the first place.

Re-thinking about the contributions of this project, I don't think that the experiments showed what we wanted.

At a high-level, one perspective of this project could be how can a computed BRT can be incorpotarated into RL in order 
for an agent to learn better in a safe manner.

This assumes that:
    1. we have some reasonable approximation of the dynamics of the robot
    2. we can compute or approximate a BRT

The benefit of having access to a BRT is that:
    1. it can provide optimal control
    2. it can capture how safe an action is
    3. safety is guarenteed as long as V(x) > 0

Currently, I'm not sure if our experiments are able to demonstrate:
    - that gradVdotF is a good learning signal
    - why a BRT is used if we have the dynamics of the robot
    - the serverity of collisons:
      - I'm not sure that measuring V(x) at collision represents how unsafe the state it.
        I believe it represents the distance to V(x) == 0, 

One thing that our experiments lack is the ability to show that a robot can learn without reseting.

Idealty, I would like to create an experiment where a drone can learn a complex task without reseting.
This would idealy show behaviouir that a BRAT could not and use an approximation of a high-dimentional system.
Like before, gradVdotFxu should be useful since it can provide feedback that an action is unsafe without a robot experiencing 
unsafe states.

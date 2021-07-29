import torch
import numpy as np
import torch
import torch.nn as nn
from typing import Callable
import random

class Model(nn.Module):
    
    def __init__(self) -> None:
        super().__init__()
        
        self.fc = nn.Sequential(
            nn.Linear(5, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 3),
            nn.Tanh(),
        )

        self._create_weights()
        for param in self.fc.parameters():
            param.requires_grad = False
        
    def _create_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)
                
    def forward(self, x):
        x = self.fc(x)

        return x

class Actor():
    """individium class"""

    def __init__(self, func: Callable):
        """init individum with random x,y in [-2,2]
        Args:
            func (Callable): fitness funciton taking x,y params
        """
        self.w = Model()
        self.fitness = func
        self.last_fitness = 0

    def eval(self) -> float:
        """evaluate fitness of this individuum
        Returns:
            [float]: fitness score (higher better)
        """
        self.last_fitness = self.fitness(self.w)
        return self.last_fitness

    def mutate(self, sigma=0.1):
        """mutate by drawing from ndist around current value with sigma"""
        for p in self.w.parameters():
            p += torch.randn_like(p)/100

def xover(a: Actor, b: Actor) -> Actor:
    """crossover between two individuals by randomly-weighted linear interpolation between their respective coefficients
    Args:
        a (Actor): parent a
        b (Actor): parent b
    Returns:
        Actor: child c
    """
    c = Actor(fittness_eval)
    
    for p in zip(c.w.parameters(), a.w.parameters(), b.w.parameters()):
        rel = torch.rand_like(p[0].data).cuda()
        p[0].data.copy_(rel * p[1].data + (1 - rel) * p[2].data)
    return c

# Main loop:
# - perform simulation steps until Webots is stopping the controller
def fittness_eval(model, do_inf=False):
    points = 0
    balls_collected = 0
    startTime = int(robot.getTime())   
    while robot.step(timestep) != -1:
        # Read the sensors:
        # Enter here functions to read sensor data, like:
        #  val = ds.getValue()
        if do_inf == False:
            theTime = robot.getTime() - startTime
            if theTime > 60*10 or (balls_collected/(1+theTime) < 1/60 and theTime > 60*3):
                reward = points + balls_collected/30.0
                print(f"finished with {balls_collected} points after {int(theTime/60)} minutes")
                return balls_collected
        imgBytes = camera.getImage()
        with torch.no_grad():
            image = (
                torch.from_numpy(
                    np.frombuffer(imgBytes, np.uint8).reshape(
                        (camera.getHeight(), camera.getWidth(), 4)
                    )[:, :, :3]
                    / 255.0
                )
                .float()
                .permute(2, 0, 1)
                .unsqueeze(0)
                .cuda()
            )
            motorSignal = model.forward(image)[0].cpu().numpy()


        
        motorRight.setVelocity(motorSignal[0]*motorRight.getMaxVelocity())
        motorLeft.setVelocity(motorSignal[1]*motorRight.getMaxVelocity())
        # Process sensor data here.

        # Enter here functions to send actuator commands, like:
        #  motor.setPosition(10.0)
        while rec.getQueueLength() > 0:
            msg_dat = rec.getData()
            rec.nextPacket()
            msg = msgpack.unpackb(msg_dat)
            points += int(msg["value"])
            balls_collected += 1

# Enter here exit cleanup code.
fittness_eval(torch.load("best.pkl"),True)
popsize = 50
maxgen = 500
use_elitism = True
allow_self_reproduction = True
pop = [indi(fittness_eval) for i in range(popsize*2)]
pop[0].w = torch.load("best.pkl")
for gen in range(maxgen):
    pop.sort(key=lambda p0: p0.eval(), reverse=True)
    best = pop[0]
    print(f"{gen}: fitness: {best.last_fitness} avg: {np.average([p.last_fitness for p in pop])}")
    torch.save(best.w, "best.pkl")

    # cross over top 10 indis of old pop
    pop = pop[0:10]
    new_pop = []
    for a in pop:
        for b in pop:
            if allow_self_reproduction == False and a == b:
                continue
            new_ind = xover(a, b)
            new_ind.mutate()
            new_pop.append(new_ind)
    random.shuffle(new_pop)
    new_pop  = new_pop[0:popsize]
    if use_elitism:
        pop = pop[0:1] + new_pop
    else:
        pop = new_pop
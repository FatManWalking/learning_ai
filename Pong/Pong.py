# -----------------------------------------------------------------------------
# Die Implementierung des Spiels habe ich aus: https://github.com/jatinmandav/Gaming-in-Python/blob/master/Pong/Pong.py
# Seine Credits:
#
# Pong
# Language - Python
# Modules - pygame, sys, random, math
#
# Controls - Arrow Keys for Right Paddle and WASD Keys for Left Paddle
#
# By - Jatin Kumar Mandav
#
# Website - https://jatinmandav.wordpress.com
#
# YouTube Channel - https://www.youtube.com/channel/UCdpf6Lz3V357cIZomPwjuFQ
# Twitter - @jatinmandav
#
# -----------------------------------------------------------------------------

# Own Parts for the Assigment are:
#
# The rulebased paddle control of the left paddle (line 81 to 96)
# and everything from line 235 onwards
# in between you will find the implementation of the game Pong
# tick_board() and after_tick() are modified game functions to let my algorithms 
# react to what is happining inside the game
# 
# the genetic algorithm is a modified version of the gentic collector bot
# implementation for webots: https://github.com/maschere/ai-lecture/blob/main/webots/controllers/genetic_collector/genetic_collector.py
# by Prof. Dr. Maximilian Scherer
# and adjusted to the Pong usecase as well es some needed finetuning for the new enviroment

import pygame
import sys
import random
from math import *
import torch
import Bild
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
from random import random, randint, sample, shuffle
from typing import Callable
import time

pygame.init()

width = 600
height = 400
display = pygame.display.set_mode((width, height))
pygame.display.set_caption("Pong!")
clock = pygame.time.Clock()

background = (27, 38, 49)
white = (236, 240, 241)
red = (203, 67, 53)
blue = (52, 152, 219)
yellow = (244, 208, 63)

top = white
bottom = white
left = white
right = white

margin = 4

scoreLeft = 0
scoreRight = 0
maxScore = 20

font = pygame.font.SysFont("Small Fonts", 30)
largeFont = pygame.font.SysFont("Small Fonts", 60)

# Regelbasierter Ansatz f??r das Spielen von Pong
# Man kann die Margin anpassen (vergr????ern) oder diese
# Funktion vor die neuste Berechnung des Balles legen
# (Paddle bewegt sich vor der neusten Bewegung des Balles)
# Um tats??chlich Punkte erzielen zu k??nnen
def rulebased(ball):
    """Einfache Abfrage der Ball und Paddle Position

    Bewegt das Paddle entsprechend zur Position des Balles.
    Spielt fehlerfrei bei entsprechender Konfiguration.
    """
    margin = 30
    if leftPaddle.y-margin <= ball.y <= leftPaddle.y+margin:
        leftChange = 0
    else:
        if leftPaddle.y+margin < ball.y:
            leftChange = 1
        else:
            leftChange = -1

    leftPaddle.move(leftChange)

# Hier beginnt die Funktionalit??t des ??bernommenen Pong-Spiel

# Draw the Boundary of Board
def boundary():
    global top, bottom, left, right
    pygame.draw.rect(display, left, (0, 0, margin, height))
    pygame.draw.rect(display, top, (0, 0, width, margin))
    pygame.draw.rect(display, right, (width-margin, 0, margin, height))
    pygame.draw.rect(display, bottom, (0, height - margin, width, margin))

    l = 25
    
    pygame.draw.rect(display, white, (width/2-margin/2, 10, margin, l))
    pygame.draw.rect(display, white, (width/2-margin/2, 60, margin, l))
    pygame.draw.rect(display, white, (width/2-margin/2, 110, margin, l))
    pygame.draw.rect(display, white, (width/2-margin/2, 160, margin, l))
    pygame.draw.rect(display, white, (width/2-margin/2, 210, margin, l))
    pygame.draw.rect(display, white, (width/2-margin/2, 260, margin, l))
    pygame.draw.rect(display, white, (width/2-margin/2, 310, margin, l))
    pygame.draw.rect(display, white, (width/2-margin/2, 360, margin, l))
    
# Paddle Class 
class Paddle:
    def __init__(self, position, speed):
        self.w = 20
        self.h = self.w*4
        self.paddleSpeed = speed
            
        if position == -1:
            self.x = 1.5*margin
        else:
            self.x = width - 1.5*margin - self.w
            
        self.y = height/2 - self.h/2

    # Show the Paddle
    def show(self):
        pygame.draw.rect(display, white, (self.x, self.y, self.w, self.h))

    # Move the Paddle
    def move(self, ydir):
        self.y += self.paddleSpeed*ydir
        if self.y < 0:
            self.y -= self.paddleSpeed*ydir
        elif self.y + self.h> height:
            self.y -= self.paddleSpeed*ydir


leftPaddle = Paddle(-1, 80)
rightPaddle = Paddle(1, 100)

# Ball Class
class Ball:
    def __init__(self, color):
        self.r = 20
        self.x = width/2 - self.r/2
        self.y = height/2 -self.r/2
        self.color = color
        self.angle = randint(-75, 75)
        if randint(0, 1):
            self.angle += 180
        
        self.speed = 100

    # Show the Ball
    def show(self):
        pygame.draw.ellipse(display, self.color, (self.x, self.y, self.r, self.r))

    # Move the Ball
    def move(self):
        global scoreLeft, scoreRight
        self.x += self.speed*cos(radians(self.angle))
        self.y += self.speed*sin(radians(self.angle))
        if self.x + self.r > width - margin:
            scoreLeft += 1
            self.angle = 180 - self.angle
        if self.x < margin:
            scoreRight += 1
            self.angle = 180 - self.angle
        if self.y < margin:
            self.angle = - self.angle
        if self.y + self.r  >=height - margin:
            self.angle = - self.angle

    # Check and Reflect the Ball when it hits the padddle
    def checkForPaddle(self):
        if self.x < width/2:
            if leftPaddle.x < self.x < leftPaddle.x + leftPaddle.w:
                if leftPaddle.y < self.y < leftPaddle.y + 10 or leftPaddle.y < self.y + self.r< leftPaddle.y + 10:
                    self.angle = -45
                if leftPaddle.y + 10 < self.y < leftPaddle.y + 20 or leftPaddle.y + 10 < self.y + self.r< leftPaddle.y + 20:
                    self.angle = -30
                if leftPaddle.y + 20 < self.y < leftPaddle.y + 30 or leftPaddle.y + 20 < self.y + self.r< leftPaddle.y + 30:
                    self.angle = -15
                if leftPaddle.y + 30 < self.y < leftPaddle.y + 40 or leftPaddle.y + 30 < self.y + self.r< leftPaddle.y + 40:
                    self.angle = -10
                if leftPaddle.y + 40 < self.y < leftPaddle.y + 50 or leftPaddle.y + 40 < self.y + self.r< leftPaddle.y + 50:
                    self.angle = 10
                if leftPaddle.y + 50 < self.y < leftPaddle.y + 60 or leftPaddle.y + 50 < self.y + self.r< leftPaddle.y + 60:
                    self.angle = 15
                if leftPaddle.y + 60 < self.y < leftPaddle.y + 70 or leftPaddle.y + 60 < self.y + self.r< leftPaddle.y + 70:
                    self.angle = 30
                if leftPaddle.y + 70 < self.y < leftPaddle.y + 80 or leftPaddle.y + 70 < self.y + self.r< leftPaddle.y + 80:
                    self.angle = 45
        else:
            if rightPaddle.x + rightPaddle.w > self.x  + self.r > rightPaddle.x:
                if rightPaddle.y < self.y < leftPaddle.y + 10 or leftPaddle.y < self.y + self.r< leftPaddle.y + 10:
                    self.angle = -135
                if rightPaddle.y + 10 < self.y < rightPaddle.y + 20 or rightPaddle.y + 10 < self.y + self.r< rightPaddle.y + 20:
                    self.angle = -150
                if rightPaddle.y + 20 < self.y < rightPaddle.y + 30 or rightPaddle.y + 20 < self.y + self.r< rightPaddle.y + 30:
                    self.angle = -165
                if rightPaddle.y + 30 < self.y < rightPaddle.y + 40 or rightPaddle.y + 30 < self.y + self.r< rightPaddle.y + 40:
                    self.angle = 170
                if rightPaddle.y + 40 < self.y < rightPaddle.y + 50 or rightPaddle.y + 40 < self.y + self.r< rightPaddle.y + 50:
                    self.angle = 190
                if rightPaddle.y + 50 < self.y < rightPaddle.y + 60 or rightPaddle.y + 50 < self.y + self.r< rightPaddle.y + 60:
                    self.angle = 165
                if rightPaddle.y + 60 < self.y < rightPaddle.y + 70 or rightPaddle.y + 60 < self.y + self.r< rightPaddle.y + 70:
                    self.angle = 150
                if rightPaddle.y + 70 < self.y < rightPaddle.y + 80 or rightPaddle.y + 70 < self.y + self.r< rightPaddle.y + 80:
                     self.angle = 135
                return 1
        return 0

# Show the Score
def showScore():
    leftScoreText = font.render("Score : " + str(scoreLeft), True, red)
    rightScoreText = font.render("Score : " + str(scoreRight), True, blue)

    display.blit(leftScoreText, (3*margin, 3*margin))
    display.blit(rightScoreText, (width/2 + 3*margin, 3*margin))

def close():
    pygame.quit()
    sys.exit()

def tick_board(ball):

    ball.move()
    rulebased(ball)

    pygame.display.update()
    clock.tick(40)
    return torch.FloatTensor([ball.y, ball.x, ball.angle, leftPaddle.y, rightPaddle.y])

def after_tick(ball):
    
    pygame.display.update()
    clock.tick(40)
    
    reflect = ball.checkForPaddle() 
    
    display.fill(background)
    showScore()

    ball.show()
    leftPaddle.show()
    rightPaddle.show()
    # global scoreRight, scoreLeft

    boundary()
    
    pygame.display.update()
    clock.tick(40)

    return scoreRight, scoreLeft, reflect

class Model(nn.Module):
    
    def __init__(self) -> None:
        super().__init__()
        
        self.fc = nn.Sequential(
            nn.Linear(5, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 2),
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

    def __init__(self, function: Callable):
        """init individum with random x,y in [-2,2]
        Args:
            function (Callable): takes to parameters for the fitness function
        """
        self.w = Model()
        self.fitness = function
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
        p[0].data.copy_(rel * p[1].data.cuda() + (1 - rel) * p[2].data.cuda())
    return c

# Main loop:
def fittness_eval(model, do_inf=False):
    ball = Ball(yellow)
    points = 0
    hit = 0
    startTime = pygame.time.get_ticks()   
    while True:

        if do_inf == False:
            theTime = pygame.time.get_ticks() - startTime
            
            if theTime>60 or points == 5:
                reward = points + hit - antiscore
                print(f"finished with {reward} points after {int(theTime/60)} minutes")
                return reward
        
        # Get the current state of the game, build a action from it
        with torch.no_grad():
            state = tick_board(ball)
            action = np.argmax(model.forward(state).cpu().numpy())
            # print(action)
            rightPaddle.move(action)
        pygame.display.update()
        clock.tick(40)
        # and then perform that action
        rightPaddle.move(action)
        
        # before the game resums and rewards are given for that action
        score, antiscore, reflect = after_tick(ball)
        points += score - antiscore
        hit += reflect

# Enter here exit cleanup code.

model = Model()

# torch.load("models/best.pth")
torch.save(model, "models/best.pth")

fittness_eval(model, False)

popsize = 20
maxgen = 500
use_elitism = True
allow_self_reproduction = True

pop = [Actor(fittness_eval) for i in range(popsize*2)]
pop[0].w = torch.load("models/best.pth")

for gen in range(maxgen):
    pop.sort(key=lambda p0: p0.eval(), reverse=True)
    best = pop[0]
    print(f"{gen}: fitness: {best.last_fitness} avg: {np.average([p.last_fitness for p in pop])}")
    torch.save(best.w, "models/best.pth")

    # cross over top 10 Actors of old pop
    pop = pop[0:10]
    new_pop = []
    for a in pop:
        for b in pop:
            if allow_self_reproduction == False and a == b:
                continue
            new_ind = xover(a, b)
            new_ind.mutate()
            new_pop.append(new_ind)
    shuffle(new_pop)
    new_pop  = new_pop[0:popsize]
    if use_elitism:
        pop = pop[0:1] + new_pop
    else:
        pop = new_pop
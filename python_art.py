import turtle
import math
import time
import threading
from multiprocessing import Process


def archimedian_turtle(some_turtle, step, nloops, diff=10):
    for _ in range(nloops):
        for i in range(3):
            some_turtle.forward(step)
            some_turtle.right(90)
        some_turtle.forward(step + diff)
        turn_angle = 180 - (math.atan(step / diff) * (180 / math.pi))
        some_turtle.right(turn_angle)
        step += diff

def rev_turtle(some_turtle, step, nloops, diff=10):
    some_turtle.forward(step)
    for _ in range(nloops):
        for i in range(3):
            some_turtle.backward(step)
            some_turtle.left(90)
        some_turtle.backward(step + diff)
        turn_angle = 180 + (math.atan(step / diff) * (180 / math.pi))
        some_turtle.right(turn_angle)
        step += diff

def make_art():
    window = turtle.Screen()
    # opens a screen
    window.bgcolor("red")
    # sets background color of screen to "red"
    jimmy = turtle.Turtle()
    jimmy.shape('turtle')
    jimmy.color('yellow')
    jimmy.speed(400)

    tim = turtle.Turtle()
    tim.shape('turtle')
    tim.color('blue')
    tim.speed(400)

    archimedian_turtle(jimmy, step=40, nloops=250, diff=5)
    rev_turtle(tim, step=40, nloops=250, diff=5)
    window.exitonclick()
    
make_art()

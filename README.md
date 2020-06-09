# GA_for_TSP_v2
Genetic Optimization Algorithm for Travelling Salesman Problems
-Mariano Crimi

## Description
This script takes a TSP standard file an attempts to solve the travelling salesman problem described in the file using genetic algorithms

## Requirements
Python 3.7
tsplib95
tkinter
scypy

## Run the script
1) Install required libraries
2) Directly run the script (TSP.py) or open the VS solution (TSP.sln) and run.
3) Select a valid TSP file (Djbuti and Qatar are added to this repository

## Algorithm
Rationale for GA
Given that TSP's are in essence discrete optimization problems, GA seemed to be best heuristic solve the problem.  

The genetic algorithm attempts to resolve the travelling salesman problem using partially mapped crossover as the crossover operations to avoid unfeasible solutions. With these operator we get two offsprings from which we kepp the one with the best fit. In terms of selection criteria the algorithm is parameterized to perform both elitism and tournament selections. 
Maybe the "novelty" introduced by my algorithm is that the # of indiviudals kept in each generation is decreasing as the process reaches the maximum number of iterations (see stopping criteria). This is inspired by the simmulated annealing methodology but also by plant breeding techniques, in which the follow a [funnel approach](https://d3i71xaburhd42.cloudfront.net/f0a70c982c85c708da2aac29579deb557db549e3/2-FigureI-1.png). 
In all honesty, the results that I got with this methodology are not far off better from the ones that I would get from a fixed population per generation, but I thought it would be fun to try.


## Hyper-parameters
Initial population size = 500
Baseline of individuals selected for the next generation=  200
Percentage of crossing between top performant individuals = 1
Percentage of the crosse altered by random mutations= 0.15
Next generation selection strategy: Elitism (top ranking individuals)

## Stopping criterion
MAX_ITERATIONS = 500


## Ideas for extension
- Use regular crossover + penalty functions instead of using partially mapped crossover
- Introduce dominance-codominance models
- Other crossover operators: Cycle Crossover Operator, Order Crossover Operator




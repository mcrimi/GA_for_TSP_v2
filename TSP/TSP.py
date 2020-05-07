import tsplib95 #add to readme file
import tkinter as tk
from tkinter import filedialog
from scipy.spatial import distance
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.optimize import Bounds
import random 

# Number of individuals in each generation 
POPULATION_SIZE = 100
# Maximum number of iterations
MAX_ITERATIONS = 100
# Percentage of top performing individuals to survive
SELECTION_RATE=0.1
# Percentage of crossing between top performant individuals
MATING_RATE=0.5
# Percentage of the population altered by random mutations
MUTATION_RATE=0.2


class Individual(object): 
	''' 
	Class representing individual in population 
	'''
	def __init__(self, chromosome,problem): 
		self.chromosome = chromosome 
		self.fitness = self.cal_fitness(problem)

	@classmethod
	def create_genome(self, problem):
		alleles = list(problem.get_nodes())
		np.random.shuffle(alleles) #constraint is embedded win the creation method
		return alleles

	def mate(self, par2, problem): 
		''' 
		Perform mating and produce new offspring 
		'''

		# chromosome for offspring 
		child_chromosome = [] 
		for gp1, gp2 in zip(self.chromosome, par2.chromosome):	 

			# random probability 
			prob = random.random() 

			# if prob is less than 0.45, insert gene 
			# from parent 1 
			if prob < 0.45: 
				child_chromosome.append(gp1) 

			# if prob is between 0.45 and 0.90, insert 
			# gene from parent 2 
			elif prob < 0.90: 
				child_chromosome.append(gp2) 

			# otherwise insert random gene(mutate), 
			# for maintaining diversity 
			#else: 
			#	child_chromosome.append(self.mutated_genes()) 

		# create new Individual(offspring) using 
		# generated chromosome for offspring 
		return Individual(child_chromosome, problem) 

	def cal_fitness(self , problem): 
		''' 
		Calculate fittness score, it is the number of 
		characters in string which differ from target 
		string. 
		'''
		cost=0
		for i in range(0, len(self.chromosome)-1):
			edge = self.chromosome[i], self.chromosome[i+1]
			cost += problem.get_weight(*edge) #we want to use: distance.euclidean(problem.node_coords[3], problem.node_coords[8]) for rounding purposes
        #adding a quadratic penalization term that adds the number of duplicated nodes in the route
		return cost + (len(np.unique(self.chromosome, return_counts=True)[1]))**2


# Driver code 
def main(): 

    global POPULATION_SIZE
    global MAX_ITERATIONS

    #Load TSP file
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename()
    problem = tsplib95.load(file_path)


    #current generation 
    generation = 1

    found = False
    #list of individuals
    population = [] 

    # create initial population 
    for _ in range(POPULATION_SIZE): 
        alleles = Individual.create_genome(problem) 
        population.append(Individual(alleles,problem))

    for _ in range(MAX_ITERATIONS):

        # sort the population in increasing order of fitness score 
        population = sorted(population, key = lambda x:x.fitness) 

        # convergence rules
        # Otherwise generate new offsprings for new generation 

        new_generation = [] 

        # Perform Elitism, that mean 10% of fittest population 
        # goes to the next generation 
        s = int(SELECTION_RATE*POPULATION_SIZE) 
        new_generation.extend(population[:s])

		# From 50% of fittest population, Individuals 
		# will mate to produce offspring 
        s = int(MATING_RATE * POPULATION_SIZE) 
        for _ in range(s): 
            parent1 = random.choice(population[:s]) 
            parent2 = random.choice(population[:s]) 
            child = parent1.mate(parent2, problem) 
            new_generation.append(child) 

        population = new_generation 

        print("Generation: {}\tGenes: {}\t Best Fitness: {}".format(generation,population[0].chromosome, population[0].fitness)) 

        generation += 1
    
    print("Generation: {}\tGenes: {}\t Best Fitness: {}".format(generation, population[0].chromosome, population[0].fitness)) 

if __name__ == '__main__': 
	main() 

#TODO
#1) Convergence rule
#2) Use distance.euclidean for distances



                

#cost function to minimize
#def J(routing,problem):
#    cost=0
#    for i in range(0, len(a)-1):
#        edge = routing[i], routing[i+1]
#        cost += problem.get_weight(*edge, ) #we want to use: distance.euclidean(problem.node_coords[3], problem.node_coords[8]) for rounding purposes
#    return cost



#J(routing=a, problem=problem)
#random.choice(list(problem.get_nodes()))

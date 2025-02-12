#!/usr/bin/env python3
"""
Improved GA for Evolving Mario Levels – Grid Encoding Version

This version includes:
  - Structured initialization that produces a level with a fixed, solid ground and mostly empty space above.
  - A mutation operator that only changes certain tiles (leaving the ground intact).
  - A multi-point column crossover to preserve horizontal structures.
  - Fixed positions for Mario’s start, flagpole, and flag.
  - Ensured tube (pipe) elements are always connected to the ground.
  - Increased enemy placement.
  - Adjusted fitness coefficients (and added an enemy bonus) to drive higher max fitness.
  
Switch between the Grid and DE encodings by adjusting the 'Individual' assignment below.
"""

import copy
import heapq
import metrics
import multiprocessing.pool as mpool
import os
import random
import time
import math

# --------------------------
# CONFIGURATION PARAMETERS
# --------------------------
WIDTH = 200
HEIGHT = 16
POPULATION_SIZE = 480
NUM_GENERATIONS = 20
MUTATION_RATE = 0.05        # probability of mutating a tile (for grid encoding)
TOURNAMENT_SIZE = 3         # number of individuals for tournament selection
SELECTION_METHOD = "mixed"  # options: "tournament", "roulette", "mixed"
OUTPUT_DIR = "levels"

# Only allow mutations on these tile types.
ALLOWED_MUTATION_TILES = ["-", "o", "X", "B"]

# These are the overall tile options (for placing new tiles in mutation)
# (Note: '?' and 'M' are left out in mutation to preserve structure.)
OPTIONS = ["-", "X", "B", "o"]

# --------------------------
# GRID ENCODING DEFINITION
# --------------------------
class Individual_Grid(object):
    __slots__ = ["genome", "_fitness"]

    def __init__(self, genome):
        # Genome is a 2D list (list of rows) of characters.
        self.genome = copy.deepcopy(genome)
        self._fitness = None

    def calculate_fitness(self):
        measurements = metrics.metrics(self.to_level())
        # Increase the solvability coefficient to boost fitness of playable levels.
        coefficients = {
            "meaningfulJumpVariance": 0.5,
            "negativeSpace": 0.6,
            "pathPercentage": 0.5,
            "emptyPercentage": 0.6,
            "linearity": -0.5,
            "solvability": 3.0,  # increased from 2.0 to 3.0
        }
        base_fitness = sum(coefficients[m] * measurements[m] for m in coefficients)
        # Bonus: if at least one enemy ("E") is present, add a small bonus.
        level_str = "".join("".join(row) for row in self.to_level())
        enemy_bonus = 0.5 if "E" in level_str else 0.0
        self._fitness = base_fitness + enemy_bonus
        return self

    def fitness(self):
        if self._fitness is None:
            self.calculate_fitness()
        return self._fitness

    # --- Mutation Operator ---
    # Only mutate rows 0..HEIGHT-2 (leave the ground intact).
    def mutate(self, genome):
        for y in range(HEIGHT - 1):  # do not change the bottom (ground) row
            for x in range(1, WIDTH - 1):  # leave leftmost and rightmost columns unchanged
                if random.random() < MUTATION_RATE:
                    if genome[y][x] in ALLOWED_MUTATION_TILES:
                        new_tile = random.choice(ALLOWED_MUTATION_TILES)
                        genome[y][x] = new_tile
        return genome

    # --- Multi-Point Column Crossover ---
    def generate_children(self, other):
        new_genome1 = copy.deepcopy(self.genome)
        new_genome2 = copy.deepcopy(self.genome)

        # Choose 2 to 5 random columns (except the borders) to swap between parents.
        crossover_points = random.randint(2, 5)
        columns = random.sample(range(1, WIDTH - 1), crossover_points)
        for x in columns:
            # Swap the entire column for all rows except the ground row.
            for y in range(HEIGHT - 1):  # Skip the ground row to preserve it.
                new_genome1[y][x] = other.genome[y][x]
                new_genome2[y][x] = self.genome[y][x]

        # Apply mutation
        new_genome1 = self.mutate(new_genome1)
        new_genome2 = self.mutate(new_genome2)
        return (Individual_Grid(new_genome1), Individual_Grid(new_genome2))

    def to_level(self):
        return self.genome

    # --- Structured Empty Individual ---
    @classmethod
    def empty_individual(cls):
        # Create an empty grid with a solid ground (all "X" in the bottom row).
        g = []
        for row in range(HEIGHT):
            if row == HEIGHT - 1:
                g.append(["X"] * WIDTH)
            else:
                g.append(["-"] * WIDTH)
        # Fix special positions:
        if HEIGHT >= 2:
            g[HEIGHT - 2][0] = "m"  # Mario's start
        if HEIGHT > 7:
            g[7][-1] = "v"        # Flagpole
        for row in range(8, min(14, HEIGHT)):
            g[row][-1] = "f"        # Flag
        for row in range(14, HEIGHT):
            g[row][-1] = "X"
        return cls(g)

    # --- Structured Random Individual ---
    @classmethod
    def random_individual(cls):
        # Start with a level that has a flat, solid ground and clear sky.
        ind = cls.empty_individual()
        g = copy.deepcopy(ind.genome)
        
        # Add horizontal platforms for Mario to jump on.
        num_platforms = random.randint(3, 7)
        for _ in range(num_platforms):
            platform_length = random.randint(3, 7)
            start_col = random.randint(5, WIDTH - platform_length - 5)
            platform_row = random.randint(3, HEIGHT - 6)
            for x in range(start_col, start_col + platform_length):
                if g[platform_row][x] == "-":
                    g[platform_row][x] = random.choice(["X", "B"])
        
        # (Removed gap creation code so that the floor remains solid.)
        
        # Add coins in the sky.
        num_coins = random.randint(5, 10)
        for _ in range(num_coins):
            coin_row = random.randint(1, HEIGHT - 3)
            coin_col = random.randint(1, WIDTH - 2)
            if g[coin_row][coin_col] == "-":
                g[coin_row][coin_col] = "o"
        
        # Add random enemies on the ground (increase count).
        num_enemies = random.randint(3, 7)
        for _ in range(num_enemies):
            enemy_col = random.randint(2, WIDTH - 3)
            enemy_row = HEIGHT - 2  # Just above the ground.
            g[enemy_row][enemy_col] = "E"
        
        # Add a tube (pipe) that is always connected to the ground.
        if random.random() < 0.3:  # 30% chance to add a tube.
            tube_col = random.randint(2, WIDTH - 3)
            tube_height = random.randint(2, 4)
            # Ensure tube's bottom touches the ground.
            tube_top_row = HEIGHT - tube_height - 1
            g[tube_top_row][tube_col] = "T"
            for row in range(tube_top_row + 1, HEIGHT):
                g[row][tube_col] = "|"
        
        # Fix special positions again to avoid overwrites.
        if HEIGHT >= 2:
            g[HEIGHT - 2][0] = "m"
        if HEIGHT > 7:
            g[7][-1] = "v"
        for row in range(8, min(14, HEIGHT)):
            g[row][-1] = "f"
        for row in range(14, HEIGHT):
            g[row][-1] = "X"
        
        return cls(g)

# --------------------------
# DESIGN ELEMENT (DE) ENCODING DEFINITION
# --------------------------
# (The DE encoding is left unchanged; you can modify it similarly if needed.)
class Individual_DE(object):
    __slots__ = ["genome", "_fitness", "_level"]

    def __init__(self, genome):
        self.genome = list(genome)
        heapq.heapify(self.genome)
        self._fitness = None
        self._level = None

    def calculate_fitness(self):
        measurements = metrics.metrics(self.to_level())
        coefficients = {
            "meaningfulJumpVariance": 0.5,
            "negativeSpace": 0.6,
            "pathPercentage": 0.5,
            "emptyPercentage": 0.6,
            "linearity": -0.5,
            "solvability": 3.0,
        }
        base_fitness = sum(coefficients[m] * measurements[m] for m in coefficients)
        level_str = "".join("".join(row) for row in self.to_level())
        enemy_bonus = 0.5 if "E" in level_str else 0.0
        self._fitness = base_fitness + enemy_bonus
        return self

    def fitness(self):
        if self._fitness is None:
            self.calculate_fitness()
        return self._fitness

    def mutate(self, new_genome):
        if random.random() < 0.1 and new_genome:
            index = random.randint(0, len(new_genome) - 1)
            de = new_genome[index]
            x, de_type = de[0], de[1]
            choice = random.random()
            if de_type == "4_block":
                y = de[2]
                breakable = de[3]
                if choice < 0.33:
                    x = offset_by_upto(x, WIDTH / 8, min=1, max=WIDTH - 2)
                elif choice < 0.66:
                    y = offset_by_upto(y, HEIGHT / 2, min=0, max=HEIGHT - 1)
                else:
                    breakable = not breakable
                new_de = (x, de_type, y, breakable)
            elif de_type == "5_qblock":
                y = de[2]
                has_powerup = de[3]
                if choice < 0.33:
                    x = offset_by_upto(x, WIDTH / 8, min=1, max=WIDTH - 2)
                elif choice < 0.66:
                    y = offset_by_upto(y, HEIGHT / 2, min=0, max=HEIGHT - 1)
                else:
                    has_powerup = not has_powerup
                new_de = (x, de_type, y, has_powerup)
            elif de_type == "3_coin":
                y = de[2]
                if choice < 0.5:
                    x = offset_by_upto(x, WIDTH / 8, min=1, max=WIDTH - 2)
                else:
                    y = offset_by_upto(y, HEIGHT / 2, min=0, max=HEIGHT - 1)
                new_de = (x, de_type, y)
            elif de_type == "7_pipe":
                h = de[2]
                if choice < 0.5:
                    x = offset_by_upto(x, WIDTH / 8, min=1, max=WIDTH - 2)
                else:
                    h = offset_by_upto(h, 2, min=2, max=HEIGHT - 4)
                new_de = (x, de_type, h)
            elif de_type == "0_hole":
                w = de[2]
                if choice < 0.5:
                    x = offset_by_upto(x, WIDTH / 8, min=1, max=WIDTH - 2)
                else:
                    w = offset_by_upto(w, 4, min=1, max=WIDTH - 2)
                new_de = (x, de_type, w)
            elif de_type == "6_stairs":
                h = de[2]
                dx = de[3]
                if choice < 0.33:
                    x = offset_by_upto(x, WIDTH / 8, min=1, max=WIDTH - 2)
                elif choice < 0.66:
                    h = offset_by_upto(h, 8, min=1, max=HEIGHT - 4)
                else:
                    dx = -dx
                new_de = (x, de_type, h, dx)
            elif de_type == "1_platform":
                w = de[2]
                y = de[3]
                madeof = de[4]
                if choice < 0.25:
                    x = offset_by_upto(x, WIDTH / 8, min=1, max=WIDTH - 2)
                elif choice < 0.5:
                    w = offset_by_upto(w, 8, min=1, max=WIDTH - 2)
                elif choice < 0.75:
                    y = offset_by_upto(y, HEIGHT, min=0, max=HEIGHT - 1)
                else:
                    madeof = random.choice(["?", "X", "B"])
                new_de = (x, de_type, w, y, madeof)
            elif de_type == "2_enemy":
                x = offset_by_upto(x, WIDTH / 8, min=1, max=WIDTH - 2)
                new_de = (x, de_type)
            new_genome.pop(index)
            heapq.heappush(new_genome, new_de)
        return new_genome

    def generate_children(self, other):
        pa = random.randint(0, len(self.genome) - 1) if self.genome else 0
        pb = random.randint(0, len(other.genome) - 1) if other.genome else 0
        a_part = self.genome[:pa] if self.genome else []
        b_part = other.genome[pb:] if other.genome else []
        child1_genome = a_part + b_part

        b_part = other.genome[:pb] if other.genome else []
        a_part = self.genome[pa:] if self.genome else []
        child2_genome = b_part + a_part

        return Individual_DE(self.mutate(child1_genome)), Individual_DE(self.mutate(child2_genome))

    def to_level(self):
        if self._level is None:
            base = Individual_Grid.empty_individual().to_level()
            for de in sorted(self.genome, key=lambda d: (d[1], d[0], d)):
                x = de[0]
                de_type = de[1]
                if de_type == "4_block":
                    y = de[2]
                    breakable = de[3]
                    base[y][x] = "B" if breakable else "X"
                elif de_type == "5_qblock":
                    y = de[2]
                    has_powerup = de[3]
                    base[y][x] = "M" if has_powerup else "?"
                elif de_type == "3_coin":
                    y = de[2]
                    base[y][x] = "o"
                elif de_type == "7_pipe":
                    h = de[2]
                    base[HEIGHT - h - 1][x] = "T"
                    for y in range(HEIGHT - h, HEIGHT):
                        base[y][x] = "|"
                elif de_type == "0_hole":
                    w = de[2]
                    for x2 in range(w):
                        base[HEIGHT - 1][clip(1, x + x2, WIDTH - 2)] = "-"
                elif de_type == "6_stairs":
                    h = de[2]
                    dx = de[3]
                    for x2 in range(1, h + 1):
                        for y in range(x2 if dx == 1 else h - x2):
                            base[clip(0, HEIGHT - y - 1, HEIGHT - 1)][clip(1, x + x2, WIDTH - 2)] = "X"
                elif de_type == "1_platform":
                    w = de[2]
                    h = de[3]
                    madeof = de[4]
                    for x2 in range(w):
                        base[clip(0, HEIGHT - h - 1, HEIGHT - 1)][clip(1, x + x2, WIDTH - 2)] = madeof
                elif de_type == "2_enemy":
                    base[HEIGHT - 2][x] = "E"
            self._level = base
        return self._level

    @classmethod
    def empty_individual(cls):
        return Individual_DE([])

    @classmethod
    def random_individual(cls):
        elt_count = random.randint(8, 128)
        g = [random.choice([
            (random.randint(1, WIDTH - 2), "0_hole", random.randint(1, 8)),
            (random.randint(1, WIDTH - 2), "1_platform", random.randint(1, 8),
             random.randint(0, HEIGHT - 1), random.choice(["?", "X", "B"])),
            (random.randint(1, WIDTH - 2), "2_enemy"),
            (random.randint(1, WIDTH - 2), "3_coin", random.randint(0, HEIGHT - 1)),
            (random.randint(1, WIDTH - 2), "4_block", random.randint(0, HEIGHT - 1),
             random.choice([True, False])),
            (random.randint(1, WIDTH - 2), "5_qblock", random.randint(0, HEIGHT - 1),
             random.choice([True, False])),
            (random.randint(1, WIDTH - 2), "6_stairs", random.randint(1, HEIGHT - 4),
             random.choice([-1, 1])),
            (random.randint(1, WIDTH - 2), "7_pipe", random.randint(2, HEIGHT - 4))
        ]) for _ in range(elt_count)]
        return Individual_DE(g)

# --------------------------
# Choose the Encoding
# --------------------------
# Uncomment the following line to use Grid encoding.
Individual = Individual_Grid  
# To use the design element encoding, comment out the above line and uncomment:
# Individual = Individual_DE

# --------------------------
# Helper Functions
# --------------------------
def offset_by_upto(val, variance, min=None, max=None):
    val += random.normalvariate(0, math.sqrt(variance))
    if min is not None and val < min:
        val = min
    if max is not None and val > max:
        val = max
    return int(val)

def clip(lo, val, hi):
    return max(lo, min(val, hi))

# --------------------------
# SELECTION STRATEGIES
# --------------------------
def tournament_selection(population, k=TOURNAMENT_SIZE):
    # Sample k candidates and ensure uniqueness based on genome string representation.
    candidates = random.sample(population, k)
    unique_candidates = list({''.join(''.join(row) for row in ind.to_level()): ind for ind in candidates}.values())
    return max(unique_candidates, key=lambda ind: ind.fitness())

def roulette_wheel_selection(population):
    min_fit = min(ind.fitness() for ind in population)
    offset = -min_fit if min_fit < 0 else 0
    total = sum(ind.fitness() + offset for ind in population)
    r = random.uniform(0, total)
    upto = 0
    for ind in population:
        weight = ind.fitness() + offset
        if upto + weight >= r:
            return ind
        upto += weight
    return population[-1]

def select_parent(population):
    if SELECTION_METHOD == "tournament":
        return tournament_selection(population)
    elif SELECTION_METHOD == "roulette":
        return roulette_wheel_selection(population)
    elif SELECTION_METHOD == "mixed":
        return tournament_selection(population) if random.random() < 0.5 else roulette_wheel_selection(population)
    else:
        raise ValueError("Unknown SELECTION_METHOD: " + SELECTION_METHOD)

# --------------------------
# Generate Successors with Elitism
# --------------------------
def generate_successors(population):
    # Use elitism: carry over the best individual unchanged.
    elite_count = 1
    sorted_pop = sorted(population, key=lambda ind: ind.fitness(), reverse=True)
    new_population = sorted_pop[:elite_count]
    
    # Fill the rest of the population by generating children.
    while len(new_population) < len(population):
        parent1 = select_parent(population)
        parent2 = select_parent(population)
        if parent1 == parent2:
            continue
        child1, child2 = parent1.generate_children(parent2)
        new_population.append(child1)
        if len(new_population) < len(population):
            new_population.append(child2)
    return new_population

# --------------------------
# GA MAIN LOOP
# --------------------------
def ga():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    batches = os.cpu_count()
    if POPULATION_SIZE % batches != 0:
        print("It is ideal if POPULATION_SIZE divides evenly into", batches, "batches.")
    batch_size = int(math.ceil(POPULATION_SIZE / batches))

    with mpool.Pool(processes=os.cpu_count()) as pool:
        init_time = time.time()
        population = [Individual.random_individual() if random.random() < 0.9
                      else Individual.empty_individual()
                      for _ in range(POPULATION_SIZE)]
        population = pool.map(Individual.calculate_fitness, population, batch_size)
        init_done = time.time()
        print("Initial population created and evaluated in: {:.2f} seconds".format(init_done - init_time))

        generation = 0
        start = time.time()
        best_overall = None
        stagnation_count = 0
        prev_best_fitness = -1

        while generation < NUM_GENERATIONS:
            now = time.time()
            best = max(population, key=lambda ind: ind.fitness())
            if best_overall is None or best.fitness() > best_overall.fitness():
                best_overall = best

            if abs(best.fitness() - prev_best_fitness) < 0.01:
                stagnation_count += 1
            else:
                stagnation_count = 0
            prev_best_fitness = best.fitness()

            if stagnation_count >= 10:
                print("Early stopping: No fitness improvement for 10 generations.")
                break

            print("\nGeneration:", generation)
            print("Max fitness:", best.fitness())
            print("Average generation time: {:.2f} sec".format((now - start) / (generation + 1)))
            print("Net time: {:.2f} sec".format(now - start))

            with open(os.path.join(OUTPUT_DIR, "last.txt"), 'w') as f:
                for row in best.to_level():
                    f.write("".join(row) + "\n")

            generation += 1
            gentime = time.time()
            next_population = generate_successors(population)
            gendone = time.time()
            print("Generated successors in: {:.2f} seconds".format(gendone - gentime))
            next_population = pool.map(Individual.calculate_fitness, next_population, batch_size)
            popdone = time.time()
            print("Calculated fitnesses in: {:.2f} seconds".format(popdone - gendone))
            population = next_population

        return population, best_overall

# --------------------------
# MAIN
# --------------------------
if __name__ == "__main__":
    final_population, best_individual = ga()
    final_sorted = sorted(final_population, key=lambda ind: ind.fitness(), reverse=True)
    print("\nBest overall fitness:", best_individual.fitness())
    now_str = time.strftime("%m_%d_%H_%M_%S")
    for k in range(min(10, len(final_sorted))):
        filename = os.path.join(OUTPUT_DIR, "{}_{}.txt".format(now_str, k))
        with open(filename, 'w') as f:
            for row in final_sorted[k].to_level():
                f.write("".join(row) + "\n")
    print("Levels saved in the '{}' directory.".format(OUTPUT_DIR))

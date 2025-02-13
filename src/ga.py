#!/usr/bin/env python3
"""
Improved GA for Evolving Mario Levels – Grid Encoding Version with FI‑2POP Extra Credit

This version includes:
  - Structured initialization that produces a level with a fixed, continuous ground.
  - A multi-point column crossover to preserve horizontal structures.
  - Enemies, coins, and mushrooms are added in fixed amounts.
  - Two populations are maintained (FI‑2POP):
       • A “feasible” (solvable) population.
       • An “infeasible” (unsolvable) population.
  - Offspring are generated separately from each sub‐population and then merged.
  - Each generation’s best level is saved to a file named with its generation number.
  
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
# HARD-WIRED PARAMETERS
# --------------------------
WIDTH = 200
HEIGHT = 16
POPULATION_SIZE = 480   # Total population size (split between feasible and infeasible)
NUM_GENERATIONS = 15
MUTATION_RATE = 0.05         # Mutation rate for non-ground tiles.
TOURNAMENT_SIZE = 3          # For selection.
SELECTION_METHOD = "mixed"   # Options: "tournament", "roulette", "mixed"
OUTPUT_DIR = "levels"
USE_2POP = True              # Set to True to run FI‑2POP extra credit version.

# Allowed tiles for mutation.
ALLOWED_MUTATION_TILES = ["-", "o", "X", "B"]  # "?" is intentionally not included.
OPTIONS = ["-", "X", "B", "o"]

# Parameters for ground generation.
START_FLAT_COLS = 5
MIN_GROUND_HEIGHT = HEIGHT - 5  # With bottom at HEIGHT-1, ensures max vertical difference of 4.

# Parameters for floating platforms.
PLATFORM_COUNT_MIN = 2
PLATFORM_COUNT_MAX = 3
PLATFORM_LENGTH_MIN = 3
PLATFORM_LENGTH_MAX = 7
# Normal platforms are placed 2 rows above local ground;
# question platforms (which use "?" blocks) are placed 4 rows above ground.

# Parameters for coins.
COIN_COUNT_MIN = 5
COIN_COUNT_MAX = 10

# Parameters for enemies.
ENEMY_COUNT_MIN = 3
ENEMY_COUNT_MAX = 7

# Parameters for tubes (pipes).
TUBE_PROBABILITY = 0.3       # 30% chance to add a tube.
TUBE_HEIGHT_MIN = 2
TUBE_HEIGHT_MAX = 3

# FI‑2POP threshold: levels with solvability metric >= this are "feasible"
SOLVABILITY_THRESHOLD = 0.5

# Parameters for diversity injection on stagnation.
STAGNATION_THRESHOLD = 3   # Generations with <0.01 improvement before diversity injection.
REINIT_PERCENT = 0.10      # Replace 10% of population when stagnation occurs.

# --------------------------
# GRID ENCODING DEFINITION
# --------------------------
class Individual_Grid(object):
    __slots__ = ["genome", "_fitness", "_measurements"]
    # _measurements caches metrics to avoid redundant computation.

    def __init__(self, genome):
        self.genome = copy.deepcopy(genome)
        self._fitness = None
        self._measurements = None

    def calculate_fitness(self):
        level_str_list = ["".join(row) for row in self.to_level()]
        if self._measurements is None:
            self._measurements = metrics.metrics(level_str_list)
        coefficients = {
            "meaningfulJumpVariance": 0.5,
            "negativeSpace": 0.6,
            "pathPercentage": 0.5,
            "emptyPercentage": 0.6,
            "linearity": -0.5,
            "solvability": 3.0,
        }
        base_fitness = sum(coefficients[m] * self._measurements[m] for m in coefficients)
        level_str = "".join(level_str_list)
        enemy_bonus = 0.5 if "E" in level_str else 0.0
        self._fitness = base_fitness + enemy_bonus
        return self

    def fitness(self):
        if self._fitness is None:
            self.calculate_fitness()
        return self._fitness

    # Mutation operator now allows "?" blocks with 20% probability.
    def mutate(self, genome):
        for y in range(HEIGHT - 1):  # Do not change ground row.
            for x in range(1, WIDTH - 1):
                if random.random() < MUTATION_RATE:
                    if random.random() < 0.2:
                        new_options = ALLOWED_MUTATION_TILES + ["?"]
                    else:
                        new_options = ALLOWED_MUTATION_TILES
                    genome[y][x] = random.choice(new_options)
        return genome

    # Multi-point column crossover (preserving ground).
    def generate_children(self, other):
        new_genome1 = copy.deepcopy(self.genome)
        new_genome2 = copy.deepcopy(self.genome)
        crossover_points = random.randint(2, 5)
        columns = random.sample(range(1, WIDTH - 1), crossover_points)
        for x in columns:
            for y in range(HEIGHT - 1):  # Skip ground row.
                new_genome1[y][x] = other.genome[y][x]
                new_genome2[y][x] = self.genome[y][x]
        new_genome1 = self.mutate(new_genome1)
        new_genome2 = self.mutate(new_genome2)
        return (Individual_Grid(new_genome1), Individual_Grid(new_genome2))

    def to_level(self):
        return self.genome

    @classmethod
    def empty_individual(cls):
        g = []
        for row in range(HEIGHT):
            if row == HEIGHT - 1:
                g.append(["X"] * WIDTH)
            else:
                g.append(["-"] * WIDTH)
        if HEIGHT >= 2:
            g[HEIGHT - 2][0] = "m"
        if HEIGHT > 7:
            g[7][-1] = "v"
        for row in range(8, min(14, HEIGHT)):
            g[row][-1] = "f"
        for row in range(14, HEIGHT):
            g[row][-1] = "X"
        return cls(g)

    @classmethod
    def random_individual(cls):
        grid = [["-"] * WIDTH for _ in range(HEIGHT)]
        ground_heights = [None] * WIDTH
        for col in range(START_FLAT_COLS):
            ground_heights[col] = HEIGHT - 1
        current_height = HEIGHT - 1
        for col in range(START_FLAT_COLS, WIDTH):
            step = random.choice([-1, 0, 1])
            new_height = current_height + step
            new_height = max(MIN_GROUND_HEIGHT, min(new_height, HEIGHT - 1))
            ground_heights[col] = new_height
            current_height = new_height
        for col in range(WIDTH):
            for row in range(ground_heights[col], HEIGHT):
                grid[row][col] = "X"
        # Add floating platforms.
        num_platforms = random.randint(PLATFORM_COUNT_MIN, PLATFORM_COUNT_MAX)
        for _ in range(num_platforms):
            platform_length = random.randint(PLATFORM_LENGTH_MIN, PLATFORM_LENGTH_MAX)
            start_col = random.randint(START_FLAT_COLS, WIDTH - platform_length - 5)
            local_min = min(ground_heights[x] for x in range(start_col, start_col + platform_length))
            if random.random() < 0.33:
                platform_row = local_min - 4
            else:
                platform_row = local_min - 2
            if platform_row < 2 or platform_row >= HEIGHT - 6:
                continue
            for x in range(start_col, start_col + platform_length):
                if grid[platform_row][x] == "-":
                    if platform_row == local_min - 4:
                        grid[platform_row][x] = "?"
                    else:
                        grid[platform_row][x] = random.choice(["X", "B"])
        # Add coins.
        num_coins = random.randint(COIN_COUNT_MIN, COIN_COUNT_MAX)
        for _ in range(num_coins):
            col = random.randint(1, WIDTH - 2)
            if ground_heights[col] > 2:
                coin_row = random.randint(1, ground_heights[col] - 2)
                if grid[coin_row][col] == "-":
                    grid[coin_row][col] = "o"
        # Add enemies.
        num_enemies = random.randint(ENEMY_COUNT_MIN, ENEMY_COUNT_MAX)
        for _ in range(num_enemies):
            col = random.randint(2, WIDTH - 3)
            enemy_row = ground_heights[col] - 1
            if enemy_row >= 0:
                grid[enemy_row][col] = "E"
        # NEW: Add mushrooms ("M"). We'll add 2 to 5 mushrooms randomly above ground.
        num_mushrooms = random.randint(4, 8)
        for _ in range(num_mushrooms):
            col = random.randint(1, WIDTH - 2)
            # Ensure there's room above the ground (at least 3 rows clearance).
            if ground_heights[col] > 3:
                mush_row = random.randint(1, ground_heights[col] - 3)
                if grid[mush_row][col] == "-":
                    grid[mush_row][col] = "M"
        # Add tube (pipe).
        if random.random() < TUBE_PROBABILITY:
            tube_col = random.randint(2, WIDTH - 3)
            tube_height = random.randint(TUBE_HEIGHT_MIN, TUBE_HEIGHT_MAX)
            ground = ground_heights[tube_col]
            if ground - tube_height >= 0:
                tube_top = ground - tube_height
                grid[tube_top][tube_col] = "T"
                for row in range(tube_top + 1, ground):
                    grid[row][tube_col] = "|"
        if HEIGHT >= 2:
            grid[HEIGHT - 2][0] = "m"
        if HEIGHT > 7:
            grid[7][-1] = "v"
        for row in range(8, min(14, HEIGHT)):
            grid[row][-1] = "f"
        for row in range(14, HEIGHT):
            grid[row][-1] = "X"
        return cls(grid)

# --------------------------
# DESIGN ELEMENT (DE) ENCODING DEFINITION
# --------------------------
# (Unchanged for brevity.)
class Individual_DE(object):
    __slots__ = ["genome", "_fitness", "_level"]
    def __init__(self, genome):
        self.genome = list(genome)
        heapq.heapify(self.genome)
        self._fitness = None
        self._level = None
    def calculate_fitness(self):
        level_str_list = ["".join(row) for row in self.to_level()]
        measurements = metrics.metrics(level_str_list)
        coefficients = {
            "meaningfulJumpVariance": 0.5,
            "negativeSpace": 0.6,
            "pathPercentage": 0.5,
            "emptyPercentage": 0.6,
            "linearity": -0.5,
            "solvability": 3.0,
        }
        base_fitness = sum(coefficients[m] * measurements[m] for m in coefficients)
        level_str = "".join(level_str_list)
        enemy_bonus = 0.5 if "E" in level_str else 0.0
        self._fitness = base_fitness + enemy_bonus
        return self
    def fitness(self):
        if self._fitness is None:
            self.calculate_fitness()
        return self._fitness
    def to_level(self):
        return Individual_Grid.empty_individual().to_level()
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
# Extra Credit: FI-2POP Functions
# --------------------------
def partition_population(pop):
    feasible = []
    infeasible = []
    for ind in pop:
        if hasattr(ind, "_measurements") and ind._measurements is not None:
            meas = ind._measurements
        else:
            level_str_list = ["".join(row) for row in ind.to_level()]
            meas = metrics.metrics(level_str_list)
            ind._measurements = meas
        if meas["solvability"] >= SOLVABILITY_THRESHOLD:
            feasible.append(ind)
        else:
            infeasible.append(ind)
    return feasible, infeasible

def generate_successors_2pop(pop_feasible, pop_infeasible):
    target_size = POPULATION_SIZE // 2
    combined = pop_feasible + pop_infeasible
    if len(pop_feasible) < target_size:
        needed = target_size - len(pop_feasible)
        pop_feasible.extend(random.sample(combined, needed))
    if len(pop_infeasible) < target_size:
        needed = target_size - len(pop_infeasible)
        pop_infeasible.extend(random.sample(combined, needed))
    new_feasible = generate_successors(pop_feasible)
    new_infeasible = generate_successors(pop_infeasible)
    return new_feasible + new_infeasible

def generate_successors(population):
    elite_count = 2
    sorted_pop = sorted(population, key=lambda ind: ind.fitness(), reverse=True)
    new_population = sorted_pop[:elite_count]
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
def ga_2pop():
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
            # Instead of early stopping, if stagnation for STAGNATION_THRESHOLD generations, reinitialize REINIT_PERCENT.
            if stagnation_count >= STAGNATION_THRESHOLD:
                print("Stagnation detected ({} generations). Reinitializing 10% of population.".format(stagnation_count))
                num_reinit = int(REINIT_PERCENT * len(population))
                for i in range(num_reinit):
                    idx = random.randint(0, len(population) - 1)
                    population[idx] = Individual.random_individual()
                stagnation_count = 0

            print("\nGeneration:", generation)
            print("Max fitness:", best.fitness())
            print("Average generation time: {:.2f} sec".format((now - start) / (generation + 1)))
            print("Net time: {:.2f} sec".format(now - start))
            gen_filename = os.path.join(OUTPUT_DIR, f"gen_{generation}.txt")
            with open(gen_filename, 'w') as f:
                for row in best.to_level():
                    f.write("".join(row) + "\n")
            feasible, infeasible = partition_population(population)
            gentime = time.time()
            next_population = generate_successors_2pop(feasible, infeasible)
            gendone = time.time()
            print("Generated successors (2POP) in: {:.2f} seconds".format(gendone - gentime))
            next_population = pool.map(Individual.calculate_fitness, next_population, batch_size)
            popdone = time.time()
            print("Calculated fitnesses in: {:.2f} seconds".format(popdone - gendone))
            population = next_population
            generation += 1
        return population, best_overall

def ga_main():
    if USE_2POP:
        return ga_2pop()
    else:
        return ga_2pop()

# --------------------------
# MAIN
# --------------------------
if __name__ == "__main__":
    final_population, best_individual = ga_main()
    final_sorted = sorted(final_population, key=lambda ind: ind.fitness(), reverse=True)
    print("\nBest overall fitness:", best_individual.fitness())
    now_str = time.strftime("%m_%d_%H_%M_%S")
    for k in range(min(10, len(final_sorted))):
        filename = os.path.join(OUTPUT_DIR, f"{now_str}_{k}.txt")
        with open(filename, 'w') as f:
            for row in final_sorted[k].to_level():
                f.write("".join(row) + "\n")
    print("Levels saved in the '{}' directory.".format(OUTPUT_DIR))

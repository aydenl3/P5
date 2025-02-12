#!/usr/bin/env python3
"""
Improved GA for Evolving Mario Levels – Grid Encoding Version

This version includes:
  - Structured initialization that produces a level with a fixed, continuous ground.
    The ground is generated with a smooth mountain profile (climbable steps, never floating)
    and with a maximum vertical difference of 4 blocks.
  - Floating platforms are placed either 2 rows above the ground or, if designated as a
    "question platform," 4 rows above the ground so that Mario can headbut it.
  - A mutation operator that only changes certain tiles (leaving the ground intact).
  - A multi-point column crossover to preserve horizontal structures.
  - Fixed positions for Mario’s start, flagpole, and flag.
  - Tubes (pipes) are always connected to the ground.
  - Enemies and coins are added in fixed amounts.
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
POPULATION_SIZE = 480
NUM_GENERATIONS = 10
MUTATION_RATE = 0.05         # probability of mutating a tile (for grid encoding)
TOURNAMENT_SIZE = 3          # number of individuals for tournament selection
SELECTION_METHOD = "mixed"   # options: "tournament", "roulette", "mixed"
OUTPUT_DIR = "levels"

# Allowed tiles for mutation.
ALLOWED_MUTATION_TILES = ["-", "o", "X", "B"]  # "?" is intentionally omitted from mutation.
OPTIONS = ["-", "X", "B", "o"]

# Parameters for ground (mountain) generation.
START_FLAT_COLS = 5
# With bottom at HEIGHT-1, use HEIGHT-5 so that the maximum vertical difference is 4 blocks.
MIN_GROUND_HEIGHT = HEIGHT - 5

# Parameters for floating platforms.
PLATFORM_COUNT_MIN = 2
PLATFORM_COUNT_MAX = 3
PLATFORM_LENGTH_MIN = 3
PLATFORM_LENGTH_MAX = 7
# Floating platforms will be anchored exactly 2 rows above the lowest ground for a normal platform,
# or 4 rows above for a "question platform" (so Mario can headbut it).

# Parameters for coins.
COIN_COUNT_MIN = 5
COIN_COUNT_MAX = 10

# Parameters for enemies.
ENEMY_COUNT_MIN = 3
ENEMY_COUNT_MAX = 7

# Parameters for tubes (pipes).
TUBE_PROBABILITY = 0.3       # 30% chance to add a tube per level.
TUBE_HEIGHT_MIN = 2
TUBE_HEIGHT_MAX = 4

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
        coefficients = {
            "meaningfulJumpVariance": 0.5,
            "negativeSpace": 0.6,
            "pathPercentage": 0.5,
            "emptyPercentage": 0.6,
            "linearity": -0.5,
            "solvability": 3.0,
        }
        base_fitness = sum(coefficients[m] * measurements[m] for m in coefficients)
        # Bonus if at least one enemy ("E") is present.
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
        for y in range(HEIGHT - 1):
            for x in range(1, WIDTH - 1):
                if random.random() < MUTATION_RATE:
                    if genome[y][x] in ALLOWED_MUTATION_TILES:
                        genome[y][x] = random.choice(ALLOWED_MUTATION_TILES)
        return genome

    # --- Multi-Point Column Crossover ---
    def generate_children(self, other):
        new_genome1 = copy.deepcopy(self.genome)
        new_genome2 = copy.deepcopy(self.genome)
        crossover_points = random.randint(2, 5)
        columns = random.sample(range(1, WIDTH - 1), crossover_points)
        for x in columns:
            for y in range(HEIGHT - 1):  # Skip the ground row.
                new_genome1[y][x] = other.genome[y][x]
                new_genome2[y][x] = self.genome[y][x]
        new_genome1 = self.mutate(new_genome1)
        new_genome2 = self.mutate(new_genome2)
        return (Individual_Grid(new_genome1), Individual_Grid(new_genome2))

    def to_level(self):
        return self.genome

    # --- Structured Empty Individual ---
    @classmethod
    def empty_individual(cls):
        g = []
        for row in range(HEIGHT):
            if row == HEIGHT - 1:
                g.append(["X"] * WIDTH)
            else:
                g.append(["-"] * WIDTH)
        if HEIGHT >= 2:
            g[HEIGHT - 2][0] = "m"  # Mario's start.
        if HEIGHT > 7:
            g[7][-1] = "v"       # Flagpole.
        for row in range(8, min(14, HEIGHT)):
            g[row][-1] = "f"       # Flag.
        for row in range(14, HEIGHT):
            g[row][-1] = "X"
        return cls(g)

    # --- Structured Random Individual with Clean Ground and Floating Platforms ---
    @classmethod
    def random_individual(cls):
        # Create an empty grid.
        grid = [["-"] * WIDTH for _ in range(HEIGHT)]
        ground_heights = [None] * WIDTH

        # Force the first few columns to be flat for Mario's start.
        for col in range(START_FLAT_COLS):
            ground_heights[col] = HEIGHT - 1

        # Generate a ground profile for columns START_FLAT_COLS..WIDTH-1.
        current_height = HEIGHT - 1
        for col in range(START_FLAT_COLS, WIDTH):
            step = random.choice([-1, 0, 1])
            new_height = current_height + step
            new_height = max(MIN_GROUND_HEIGHT, min(new_height, HEIGHT - 1))
            ground_heights[col] = new_height
            current_height = new_height

        # Fill in the ground.
        for col in range(WIDTH):
            for row in range(ground_heights[col], HEIGHT):
                grid[row][col] = "X"

        # --- Add Floating Platforms ---
        # For each platform, decide whether it is a question platform.
        num_platforms = random.randint(PLATFORM_COUNT_MIN, PLATFORM_COUNT_MAX)
        for _ in range(num_platforms):
            platform_length = random.randint(PLATFORM_LENGTH_MIN, PLATFORM_LENGTH_MAX)
            start_col = random.randint(START_FLAT_COLS, WIDTH - platform_length - 5)
            local_min = min(ground_heights[x] for x in range(start_col, start_col + platform_length))
            # Decide platform type: 33% chance to be a question platform.
            if random.random() < 0.33:
                # For a question platform, set the platform 4 rows above the lowest ground.
                platform_row = local_min - 4
            else:
                platform_row = local_min - 2
            if platform_row < 2 or platform_row >= HEIGHT - 6:
                continue
            for x in range(start_col, start_col + platform_length):
                if grid[platform_row][x] == "-":
                    # If this platform is a question platform, force "?".
                    if platform_row == local_min - 4:
                        grid[platform_row][x] = "?"
                    else:
                        grid[platform_row][x] = random.choice(["X", "B"])

        # --- Add Coins ---
        num_coins = random.randint(COIN_COUNT_MIN, COIN_COUNT_MAX)
        for _ in range(num_coins):
            col = random.randint(1, WIDTH - 2)
            if ground_heights[col] > 2:
                coin_row = random.randint(1, ground_heights[col] - 2)
                if grid[coin_row][col] == "-":
                    grid[coin_row][col] = "o"

        # --- Add Enemies ---
        num_enemies = random.randint(ENEMY_COUNT_MIN, ENEMY_COUNT_MAX)
        for _ in range(num_enemies):
            col = random.randint(2, WIDTH - 3)
            enemy_row = ground_heights[col] - 1
            if enemy_row >= 0:
                grid[enemy_row][col] = "E"

        # --- Add a Tube (Pipe) ---
        if random.random() < TUBE_PROBABILITY:
            tube_col = random.randint(2, WIDTH - 3)
            tube_height = random.randint(TUBE_HEIGHT_MIN, TUBE_HEIGHT_MAX)
            ground = ground_heights[tube_col]
            if ground - tube_height >= 0:
                tube_top = ground - tube_height
                grid[tube_top][tube_col] = "T"
                for row in range(tube_top + 1, ground):
                    grid[row][tube_col] = "|"

        # --- Fix Special Positions ---
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
# (Unchanged; modify as needed.)
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
# Generate Successors with Elitism
# --------------------------
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
            # Save the best level for this generation to a file named "gen_<generation>.txt".
            gen_filename = os.path.join(OUTPUT_DIR, f"gen_{generation}.txt")
            with open(gen_filename, 'w') as f:
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
        filename = os.path.join(OUTPUT_DIR, f"{now_str}_{k}.txt")
        with open(filename, 'w') as f:
            for row in final_sorted[k].to_level():
                f.write("".join(row) + "\n")
    print("Levels saved in the '{}' directory.".format(OUTPUT_DIR))

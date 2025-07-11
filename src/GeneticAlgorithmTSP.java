import java.util.*;
import java.util.concurrent.*;
import java.util.stream.IntStream;

public class GeneticAlgorithmTSP {
    
    // Problem parameters - TSP với 5000 cities
    private static final int NUM_CITIES = 5000;
    private static final int POPULATION_SIZE = 10000;
    private static final int MAX_GENERATIONS = 1000000;
    private static final int NUM_THREADS = 4;
    private static final int ELITE_SIZE = 500; // Top 5% individuals
    private static final double MUTATION_RATE = 0.02;
    private static final double CROSSOVER_RATE = 0.8;
    private static final int TOURNAMENT_SIZE = 5;
    
    // City coordinates
    private static double[][] cities;
    private static double[][] distanceMatrix;
    
    // Random number generator
    private static final Random random = new Random(42);
    
    // Initialize cities với random coordinates
    static {
        cities = new double[NUM_CITIES][2];
        for (int i = 0; i < NUM_CITIES; i++) {
            cities[i][0] = random.nextDouble() * 10000; // x coordinate
            cities[i][1] = random.nextDouble() * 10000; // y coordinate
        }
        
        // Precompute distance matrix
        distanceMatrix = new double[NUM_CITIES][NUM_CITIES];
        for (int i = 0; i < NUM_CITIES; i++) {
            for (int j = 0; j < NUM_CITIES; j++) {
                if (i != j) {
                    double dx = cities[i][0] - cities[j][0];
                    double dy = cities[i][1] - cities[j][1];
                    distanceMatrix[i][j] = Math.sqrt(dx * dx + dy * dy);
                } else {
                    distanceMatrix[i][j] = 0.0;
                }
            }
        }
    }
    
    // Individual representation
    static class Individual implements Comparable<Individual> {
        int[] chromosome;
        double fitness;
        boolean fitnessCalculated;
        
        public Individual(int[] chromosome) {
            this.chromosome = chromosome.clone();
            this.fitnessCalculated = false;
        }
        
        public Individual() {
            this.chromosome = new int[NUM_CITIES];
            // Random permutation
            for (int i = 0; i < NUM_CITIES; i++) {
                chromosome[i] = i;
            }
            shuffleArray(chromosome);
            this.fitnessCalculated = false;
        }
        
        public double getFitness() {
            if (!fitnessCalculated) {
                calculateFitness();
            }
            return fitness;
        }
        
        private void calculateFitness() {
            double totalDistance = 0.0;
            for (int i = 0; i < NUM_CITIES; i++) {
                int currentCity = chromosome[i];
                int nextCity = chromosome[(i + 1) % NUM_CITIES];
                totalDistance += distanceMatrix[currentCity][nextCity];
            }
            // Fitness is inverse of total distance
            fitness = 1.0 / (1.0 + totalDistance);
            fitnessCalculated = true;
        }
        
        public double getTotalDistance() {
            double totalDistance = 0.0;
            for (int i = 0; i < NUM_CITIES; i++) {
                int currentCity = chromosome[i];
                int nextCity = chromosome[(i + 1) % NUM_CITIES];
                totalDistance += distanceMatrix[currentCity][nextCity];
            }
            return totalDistance;
        }
        
        @Override
        public int compareTo(Individual other) {
            return Double.compare(other.getFitness(), this.getFitness());
        }
        
        public Individual clone() {
            return new Individual(this.chromosome);
        }
    }
    
    // Utility method để shuffle array
    private static void shuffleArray(int[] array) {
        for (int i = array.length - 1; i > 0; i--) {
            int index = random.nextInt(i + 1);
            int temp = array[index];
            array[index] = array[i];
            array[i] = temp;
        }
    }
    
    // Tournament selection
    private static Individual tournamentSelection(List<Individual> population) {
        Individual best = null;
        for (int i = 0; i < TOURNAMENT_SIZE; i++) {
            Individual candidate = population.get(random.nextInt(population.size()));
            if (best == null || candidate.getFitness() > best.getFitness()) {
                best = candidate;
            }
        }
        return best.clone();
    }
    
    // Order Crossover (OX)
    private static Individual[] orderCrossover(Individual parent1, Individual parent2) {
        int[] child1 = new int[NUM_CITIES];
        int[] child2 = new int[NUM_CITIES];
        
        // Initialize với -1
        Arrays.fill(child1, -1);
        Arrays.fill(child2, -1);
        
        // Random crossover points
        int start = random.nextInt(NUM_CITIES);
        int end = random.nextInt(NUM_CITIES);
        if (start > end) {
            int temp = start;
            start = end;
            end = temp;
        }
        
        // Copy segment from parent1 to child1
        for (int i = start; i <= end; i++) {
            child1[i] = parent1.chromosome[i];
        }
        
        // Copy segment from parent2 to child2
        for (int i = start; i <= end; i++) {
            child2[i] = parent2.chromosome[i];
        }
        
        // Fill remaining positions
        fillRemainingPositions(child1, parent2.chromosome, end + 1);
        fillRemainingPositions(child2, parent1.chromosome, end + 1);
        
        return new Individual[]{new Individual(child1), new Individual(child2)};
    }
    
    private static void fillRemainingPositions(int[] child, int[] parentChromosome, int startPos) {
        Set<Integer> used = new HashSet<>();
        for (int gene : child) {
            if (gene != -1) {
                used.add(gene);
            }
        }
        
        int pos = startPos % NUM_CITIES;
        int parentPos = startPos % NUM_CITIES;
        
        while (used.size() < NUM_CITIES) {
            if (pos >= NUM_CITIES) pos = 0;
            if (parentPos >= NUM_CITIES) parentPos = 0;
            
            if (child[pos] == -1) {
                while (used.contains(parentChromosome[parentPos])) {
                    parentPos = (parentPos + 1) % NUM_CITIES;
                }
                child[pos] = parentChromosome[parentPos];
                used.add(parentChromosome[parentPos]);
                parentPos = (parentPos + 1) % NUM_CITIES;
            }
            pos++;
        }
    }
    
    // Swap mutation
    private static void swapMutation(Individual individual) {
        if (random.nextDouble() < MUTATION_RATE) {
            int pos1 = random.nextInt(NUM_CITIES);
            int pos2 = random.nextInt(NUM_CITIES);
            
            int temp = individual.chromosome[pos1];
            individual.chromosome[pos1] = individual.chromosome[pos2];
            individual.chromosome[pos2] = temp;
            
            individual.fitnessCalculated = false;
        }
    }
    
    // Inversion mutation
    private static void inversionMutation(Individual individual) {
        if (random.nextDouble() < MUTATION_RATE / 2) {
            int start = random.nextInt(NUM_CITIES);
            int end = random.nextInt(NUM_CITIES);
            if (start > end) {
                int temp = start;
                start = end;
                end = temp;
            }
            
            // Reverse the segment
            while (start < end) {
                int temp = individual.chromosome[start];
                individual.chromosome[start] = individual.chromosome[end];
                individual.chromosome[end] = temp;
                start++;
                end--;
            }
            
            individual.fitnessCalculated = false;
        }
    }
    
    // 2-opt local search
    private static void twoOptImprovement(Individual individual) {
        boolean improved = true;
        int iterations = 0;
        final int MAX_2OPT_ITERATIONS = 100;
        
        while (improved && iterations < MAX_2OPT_ITERATIONS) {
            improved = false;
            iterations++;
            
            for (int i = 0; i < NUM_CITIES - 1 && !improved; i++) {
                for (int j = i + 1; j < NUM_CITIES && !improved; j++) {
                    if (j - i == 1) continue; // Skip adjacent cities
                    
                    double currentDistance = getTwoOptDistance(individual.chromosome, i, j);
                    double newDistance = getTwoOptDistanceAfterSwap(individual.chromosome, i, j);
                    
                    if (newDistance < currentDistance) {
                        // Perform 2-opt swap
                        twoOptSwap(individual.chromosome, i, j);
                        individual.fitnessCalculated = false;
                        improved = true;
                    }
                }
            }
        }
    }
    
    private static double getTwoOptDistance(int[] chromosome, int i, int j) {
        int city1 = chromosome[i];
        int city2 = chromosome[(i + 1) % NUM_CITIES];
        int city3 = chromosome[j];
        int city4 = chromosome[(j + 1) % NUM_CITIES];
        
        return distanceMatrix[city1][city2] + distanceMatrix[city3][city4];
    }
    
    private static double getTwoOptDistanceAfterSwap(int[] chromosome, int i, int j) {
        int city1 = chromosome[i];
        int city2 = chromosome[(i + 1) % NUM_CITIES];
        int city3 = chromosome[j];
        int city4 = chromosome[(j + 1) % NUM_CITIES];
        
        return distanceMatrix[city1][city3] + distanceMatrix[city2][city4];
    }
    
    private static void twoOptSwap(int[] chromosome, int i, int j) {
        int start = (i + 1) % NUM_CITIES;
        int end = j;
        
        while (start != end && start != (end + 1) % NUM_CITIES) {
            int temp = chromosome[start];
            chromosome[start] = chromosome[end];
            chromosome[end] = temp;
            
            start = (start + 1) % NUM_CITIES;
            end = (end - 1 + NUM_CITIES) % NUM_CITIES;
        }
    }
    
    // GA Worker thread
    static class GAWorker implements Callable<Individual> {
        private final int threadId;
        private final int generationsPerThread;
        private final List<Individual> initialPopulation;
        
        public GAWorker(int threadId, int generationsPerThread, List<Individual> initialPopulation) {
            this.threadId = threadId;
            this.generationsPerThread = generationsPerThread;
            this.initialPopulation = new ArrayList<>(initialPopulation);
        }
        
        @Override
        public Individual call() throws Exception {
            List<Individual> population = new ArrayList<>(initialPopulation);
            Individual bestIndividual = null;
            double bestFitness = 0.0;
            
            for (int generation = 0; generation < generationsPerThread; generation++) {
                // Evaluation
                population.parallelStream().forEach(Individual::getFitness);
                
                // Sort population
                Collections.sort(population);
                
                // Update best
                if (population.get(0).getFitness() > bestFitness) {
                    bestFitness = population.get(0).getFitness();
                    bestIndividual = population.get(0).clone();
                }
                
                // Create new population
                List<Individual> newPopulation = new ArrayList<>();
                
                // Elitism - keep best individuals
                for (int i = 0; i < ELITE_SIZE; i++) {
                    newPopulation.add(population.get(i).clone());
                }
                
                // Generate offspring
                while (newPopulation.size() < POPULATION_SIZE) {
                    Individual parent1 = tournamentSelection(population);
                    Individual parent2 = tournamentSelection(population);
                    
                    Individual[] offspring;
                    if (random.nextDouble() < CROSSOVER_RATE) {
                        offspring = orderCrossover(parent1, parent2);
                    } else {
                        offspring = new Individual[]{parent1.clone(), parent2.clone()};
                    }
                    
                    // Mutation
                    for (Individual child : offspring) {
                        swapMutation(child);
                        inversionMutation(child);
                        
                        // Apply 2-opt with probability
                        if (random.nextDouble() < 0.1) {
                            twoOptImprovement(child);
                        }
                        
                        if (newPopulation.size() < POPULATION_SIZE) {
                            newPopulation.add(child);
                        }
                    }
                }
                
                population = newPopulation;
                
                // Progress reporting
                if (generation % 1000 == 0) {
                    System.out.printf("Thread %d - Generation %d: Best Distance = %.2f, Fitness = %.8f\n",
                            threadId, generation, bestIndividual.getTotalDistance(), bestFitness);
                }
                
                // Diversity maintenance
                if (generation % 10000 == 0) {
                    maintainDiversity(population);
                }
            }
            
            return bestIndividual;
        }
        
        private void maintainDiversity(List<Individual> population) {
            // Replace worst 10% with random individuals
            int replaceCount = POPULATION_SIZE / 10;
            for (int i = POPULATION_SIZE - replaceCount; i < POPULATION_SIZE; i++) {
                population.set(i, new Individual());
            }
        }
    }
    
    // Island model for parallel GA
    static class IslandGAWorker implements Callable<Individual> {
        private final int islandId;
        private final int generations;
        private final List<Individual> population;
        private final BlockingQueue<Individual> migrationQueue;
        
        public IslandGAWorker(int islandId, int generations, 
                             List<Individual> population, BlockingQueue<Individual> migrationQueue) {
            this.islandId = islandId;
            this.generations = generations;
            this.population = new ArrayList<>(population);
            this.migrationQueue = migrationQueue;
        }
        
        @Override
        public Individual call() throws Exception {
            Individual bestIndividual = null;
            double bestFitness = 0.0;
            
            for (int generation = 0; generation < generations; generation++) {
                // Standard GA operations
                population.parallelStream().forEach(Individual::getFitness);
                Collections.sort(population);
                
                if (population.get(0).getFitness() > bestFitness) {
                    bestFitness = population.get(0).getFitness();
                    bestIndividual = population.get(0).clone();
                }
                
                // Migration every 100 generations
                if (generation % 100 == 0 && generation > 0) {
                    // Send best individual to migration queue
                    if (!migrationQueue.offer(population.get(0).clone())) {
                        // Queue full, skip migration
                    }
                    
                    // Receive immigrant
                    Individual immigrant = migrationQueue.poll();
                    if (immigrant != null) {
                        // Replace worst individual
                        population.set(population.size() - 1, immigrant);
                    }
                }
                
                // Create new generation
                List<Individual> newPopulation = new ArrayList<>();
                
                // Elitism
                for (int i = 0; i < ELITE_SIZE / NUM_THREADS; i++) {
                    newPopulation.add(population.get(i).clone());
                }
                
                // Generate offspring
                while (newPopulation.size() < POPULATION_SIZE / NUM_THREADS) {
                    Individual parent1 = tournamentSelection(population);
                    Individual parent2 = tournamentSelection(population);
                    
                    Individual[] offspring;
                    if (random.nextDouble() < CROSSOVER_RATE) {
                        offspring = orderCrossover(parent1, parent2);
                    } else {
                        offspring = new Individual[]{parent1.clone(), parent2.clone()};
                    }
                    
                    for (Individual child : offspring) {
                        swapMutation(child);
                        inversionMutation(child);
                        
                        if (random.nextDouble() < 0.05) { // 5% chance for 2-opt
                            twoOptImprovement(child);
                        }
                        
                        if (newPopulation.size() < POPULATION_SIZE / NUM_THREADS) {
                            newPopulation.add(child);
                        }
                    }
                }
                
                population.clear();
                population.addAll(newPopulation);
                
                if (generation % 5000 == 0) {
                    System.out.printf("Island %d - Generation %d: Best Distance = %.2f\n",
                            islandId, generation, bestIndividual.getTotalDistance());
                }
            }
            
            return bestIndividual;
        }
    }
    
    public static void main(String[] args) throws InterruptedException, ExecutionException {
        System.out.println("=== Genetic Algorithm for Large TSP ===");
        System.out.printf("Cities: %d\n", NUM_CITIES);
        System.out.printf("Population Size: %d\n", POPULATION_SIZE);
        System.out.printf("Max Generations: %d\n", MAX_GENERATIONS);
        System.out.printf("Threads: %d\n", NUM_THREADS);
        System.out.printf("Crossover Rate: %.2f\n", CROSSOVER_RATE);
        System.out.printf("Mutation Rate: %.4f\n", MUTATION_RATE);
        
        long startTime = System.currentTimeMillis();
        
        // Initialize population
        System.out.println("Initializing population...");
        List<Individual> initialPopulation = new ArrayList<>();
        for (int i = 0; i < POPULATION_SIZE; i++) {
            initialPopulation.add(new Individual());
        }
        
        // Calculate initial best
        Individual initialBest = Collections.min(initialPopulation);
        System.out.printf("Initial best distance: %.2f\n", initialBest.getTotalDistance());
        
        // Run Island Model GA
        ExecutorService executor = Executors.newFixedThreadPool(NUM_THREADS);
        BlockingQueue<Individual> migrationQueue = new LinkedBlockingQueue<>(NUM_THREADS * 2);
        
        List<Future<Individual>> futures = new ArrayList<>();
        int generationsPerIsland = MAX_GENERATIONS / NUM_THREADS;
        
        for (int i = 0; i < NUM_THREADS; i++) {
            int startIdx = i * (POPULATION_SIZE / NUM_THREADS);
            int endIdx = (i + 1) * (POPULATION_SIZE / NUM_THREADS);
            List<Individual> islandPopulation = new ArrayList<>(
                initialPopulation.subList(startIdx, endIdx));
            
            futures.add(executor.submit(new IslandGAWorker(i, generationsPerIsland, 
                                                          islandPopulation, migrationQueue)));
        }
        
        // Wait for completion
        Individual bestSolution = null;
        double bestDistance = Double.MAX_VALUE;
        
        for (Future<Individual> future : futures) {
            Individual result = future.get();
            double distance = result.getTotalDistance();
            if (distance < bestDistance) {
                bestDistance = distance;
                bestSolution = result;
            }
        }
        
        executor.shutdown();
        
        long endTime = System.currentTimeMillis();
        long executionTime = endTime - startTime;
        
        System.out.println("\n=== OPTIMIZATION RESULTS ===");
        System.out.printf("Best Distance Found: %.2f\n", bestDistance);
        System.out.printf("Improvement: %.2f%%\n", 
                         (initialBest.getTotalDistance() - bestDistance) / initialBest.getTotalDistance() * 100);
        System.out.printf("Execution Time: %d hours %d minutes %d seconds\n",
                         executionTime / 3600000,
                         (executionTime % 3600000) / 60000,
                         (executionTime % 60000) / 1000);
        
        // Performance statistics
        long totalOperations = (long) MAX_GENERATIONS * POPULATION_SIZE * NUM_CITIES;
        double operationsPerSecond = (double) totalOperations / (executionTime / 1000.0);
        
        System.out.printf("Total Operations: %,d\n", totalOperations);
        System.out.printf("Operations/Second: %,.0f\n", operationsPerSecond);
        System.out.printf("Memory Usage: %.2f MB\n",
                         (Runtime.getRuntime().totalMemory() - Runtime.getRuntime().freeMemory()) / 1048576.0);
        
        // Display solution path (first 20 cities)
        System.out.println("\nBest tour (first 20 cities):");
        for (int i = 0; i < Math.min(20, NUM_CITIES); i++) {
            int cityId = bestSolution.chromosome[i];
            System.out.printf("City %d: (%.2f, %.2f)\n", 
                             cityId, cities[cityId][0], cities[cityId][1]);
        }
        
        // Verify solution validity
        boolean[] visited = new boolean[NUM_CITIES];
        for (int city : bestSolution.chromosome) {
            if (visited[city]) {
                System.out.println("ERROR: Invalid solution - duplicate city!");
                return;
            }
            visited[city] = true;
        }
        
        for (boolean v : visited) {
            if (!v) {
                System.out.println("ERROR: Invalid solution - missing city!");
                return;
            }
        }
        
        System.out.println("Solution verified: All cities visited exactly once.");
    }
}

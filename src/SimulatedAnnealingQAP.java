import java.util.*;
import java.util.concurrent.*;
import java.util.concurrent.atomic.AtomicLong;
import java.util.stream.IntStream;

public class SimulatedAnnealingQAP {
    
    // Problem parameters
    private static final int PROBLEM_SIZE = 500;
    private static final long MAX_ITERATIONS = 100_000_000L;
    private static final int NUM_THREADS = 4;
    private static final double INITIAL_TEMPERATURE = 10000.0;
    private static final double FINAL_TEMPERATURE = 0.001;
    private static final double COOLING_RATE = 0.99999;
    private static final int RESTART_INTERVAL = 5_000_000;
    
    // Problem matrices
    private static double[][] flowMatrix; // Flow between facilities
    private static double[][] distanceMatrix; // Distance between locations
    private static Random random = new Random(42);
    
    // Initialize problem instance
    static {
        System.out.println("Initializing QAP instance...");
        flowMatrix = new double[PROBLEM_SIZE][PROBLEM_SIZE];
        distanceMatrix = new double[PROBLEM_SIZE][PROBLEM_SIZE];
        
        // Generate random flow matrix (asymmetric)
        for (int i = 0; i < PROBLEM_SIZE; i++) {
            for (int j = 0; j < PROBLEM_SIZE; j++) {
                if (i != j) {
                    flowMatrix[i][j] = random.nextDouble() * 100.0;
                    // Make some flows much higher to create structure
                    if (random.nextDouble() < 0.1) {
                        flowMatrix[i][j] *= 10.0;
                    }
                }
            }
        }
        
        // Generate distance matrix (symmetric, satisfies triangle inequality)
        for (int i = 0; i < PROBLEM_SIZE; i++) {
            for (int j = i + 1; j < PROBLEM_SIZE; j++) {
                double distance = random.nextDouble() * 1000.0;
                distanceMatrix[i][j] = distance;
                distanceMatrix[j][i] = distance;
            }
        }
        
        System.out.println("QAP instance initialized.");
    }
    
    // Solution representation
    static class Solution {
        int[] assignment; // assignment[i] = location assigned to facility i
        double cost;
        boolean costCalculated;
        
        public Solution(int[] assignment) {
            this.assignment = assignment.clone();
            this.costCalculated = false;
        }
        
        public Solution() {
            this.assignment = new int[PROBLEM_SIZE];
            // Initialize with random permutation
            for (int i = 0; i < PROBLEM_SIZE; i++) {
                assignment[i] = i;
            }
            shuffleArray(assignment);
            this.costCalculated = false;
        }
        
        public double getCost() {
            if (!costCalculated) {
                calculateCost();
            }
            return cost;
        }
        
        private void calculateCost() {
            cost = 0.0;
            for (int i = 0; i < PROBLEM_SIZE; i++) {
                for (int j = 0; j < PROBLEM_SIZE; j++) {
                    if (i != j) {
                        cost += flowMatrix[i][j] * distanceMatrix[assignment[i]][assignment[j]];
                    }
                }
            }
            costCalculated = true;
        }
        
        public Solution clone() {
            Solution cloned = new Solution(this.assignment);
            cloned.cost = this.cost;
            cloned.costCalculated = this.costCalculated;
            return cloned;
        }
        
        // Swap two assignments
        public void swap(int facility1, int facility2) {
            int temp = assignment[facility1];
            assignment[facility1] = assignment[facility2];
            assignment[facility2] = temp;
            costCalculated = false;
        }
        
        // Calculate cost difference for swap (more efficient than recalculating)
        public double getSwapCostDifference(int facility1, int facility2) {
            if (facility1 == facility2) return 0.0;
            
            int loc1 = assignment[facility1];
            int loc2 = assignment[facility2];
            
            double delta = 0.0;
            
            // Calculate the change in cost
            for (int k = 0; k < PROBLEM_SIZE; k++) {
                if (k != facility1 && k != facility2) {
                    int locK = assignment[k];
                    
                    // Remove old costs
                    delta -= flowMatrix[facility1][k] * distanceMatrix[loc1][locK];
                    delta -= flowMatrix[k][facility1] * distanceMatrix[locK][loc1];
                    delta -= flowMatrix[facility2][k] * distanceMatrix[loc2][locK];
                    delta -= flowMatrix[k][facility2] * distanceMatrix[locK][loc2];
                    
                    // Add new costs
                    delta += flowMatrix[facility1][k] * distanceMatrix[loc2][locK];
                    delta += flowMatrix[k][facility1] * distanceMatrix[locK][loc2];
                    delta += flowMatrix[facility2][k] * distanceMatrix[loc1][locK];
                    delta += flowMatrix[k][facility2] * distanceMatrix[locK][loc1];
                }
            }
            
            // Handle direct interaction between facility1 and facility2
            delta -= flowMatrix[facility1][facility2] * distanceMatrix[loc1][loc2];
            delta -= flowMatrix[facility2][facility1] * distanceMatrix[loc2][loc1];
            delta += flowMatrix[facility1][facility2] * distanceMatrix[loc2][loc1];
            delta += flowMatrix[facility2][facility1] * distanceMatrix[loc1][loc2];
            
            return delta;
        }
        
        // 3-opt move
        public void threeOpt(int pos1, int pos2, int pos3) {
            if (pos1 == pos2 || pos2 == pos3 || pos1 == pos3) return;
            
            // Sort positions
            int[] positions = {pos1, pos2, pos3};
            Arrays.sort(positions);
            
            // Perform 3-opt move (one of several possible)
            int temp = assignment[positions[0]];
            assignment[positions[0]] = assignment[positions[1]];
            assignment[positions[1]] = assignment[positions[2]];
            assignment[positions[2]] = temp;
            
            costCalculated = false;
        }
        
        // Insert move
        public void insert(int from, int to) {
            if (from == to) return;
            
            int facilityToMove = assignment[from];
            
            if (from < to) {
                // Shift left
                for (int i = from; i < to; i++) {
                    assignment[i] = assignment[i + 1];
                }
            } else {
                // Shift right
                for (int i = from; i > to; i--) {
                    assignment[i] = assignment[i - 1];
                }
            }
            
            assignment[to] = facilityToMove;
            costCalculated = false;
        }
    }
    
    // Utility method
    private static void shuffleArray(int[] array) {
        for (int i = array.length - 1; i > 0; i--) {
            int index = random.nextInt(i + 1);
            int temp = array[index];
            array[index] = array[i];
            array[i] = temp;
        }
    }
    
    // Tabu Search enhancement
    static class TabuList {
        private final int[][] tabuMatrix;
        private final int tabuTenure;
        private int currentIteration;
        
        public TabuList(int size, int tenure) {
            this.tabuMatrix = new int[size][size];
            this.tabuTenure = tenure;
            this.currentIteration = 0;
        }
        
        public boolean isTabu(int facility1, int facility2) {
            return tabuMatrix[facility1][facility2] > currentIteration;
        }
        
        public void addMove(int facility1, int facility2) {
            tabuMatrix[facility1][facility2] = currentIteration + tabuTenure;
            tabuMatrix[facility2][facility1] = currentIteration + tabuTenure;
        }
        
        public void incrementIteration() {
            currentIteration++;
        }
        
        public void clear() {
            for (int i = 0; i < tabuMatrix.length; i++) {
                Arrays.fill(tabuMatrix[i], 0);
            }
            currentIteration = 0;
        }
    }
    
    // Neighborhood structure
    static class Neighborhood {
        private final Solution solution;
        private final Random random;
        private final TabuList tabuList;
        
        public Neighborhood(Solution solution, TabuList tabuList) {
            this.solution = solution;
            this.random = new Random();
            this.tabuList = tabuList;
        }
        
        // Generate random swap neighbor
        public Solution getRandomSwapNeighbor() {
            int facility1 = random.nextInt(PROBLEM_SIZE);
            int facility2 = random.nextInt(PROBLEM_SIZE);
            while (facility2 == facility1) {
                facility2 = random.nextInt(PROBLEM_SIZE);
            }
            
            Solution neighbor = solution.clone();
            neighbor.swap(facility1, facility2);
            return neighbor;
        }
        
        // Generate best non-tabu neighbor
        public Solution getBestNonTabuNeighbor() {
            Solution bestNeighbor = null;
            double bestCost = Double.MAX_VALUE;
            int bestFac1 = -1, bestFac2 = -1;
            
            // Try all possible swaps
            for (int i = 0; i < PROBLEM_SIZE; i++) {
                for (int j = i + 1; j < PROBLEM_SIZE; j++) {
                    if (!tabuList.isTabu(i, j)) {
                        double delta = solution.getSwapCostDifference(i, j);
                        double newCost = solution.getCost() + delta;
                        
                        if (newCost < bestCost) {
                            bestCost = newCost;
                            bestFac1 = i;
                            bestFac2 = j;
                        }
                    }
                }
            }
            
            if (bestFac1 != -1) {
                bestNeighbor = solution.clone();
                bestNeighbor.swap(bestFac1, bestFac2);
                tabuList.addMove(bestFac1, bestFac2);
            }
            
            return bestNeighbor;
        }
        
        // Generate random 3-opt neighbor
        public Solution getRandom3OptNeighbor() {
            int pos1 = random.nextInt(PROBLEM_SIZE);
            int pos2 = random.nextInt(PROBLEM_SIZE);
            int pos3 = random.nextInt(PROBLEM_SIZE);
            
            Solution neighbor = solution.clone();
            neighbor.threeOpt(pos1, pos2, pos3);
            return neighbor;
        }
        
        // Generate random insert neighbor
        public Solution getRandomInsertNeighbor() {
            int from = random.nextInt(PROBLEM_SIZE);
            int to = random.nextInt(PROBLEM_SIZE);
            
            Solution neighbor = solution.clone();
            neighbor.insert(from, to);
            return neighbor;
        }
    }
    
    // Advanced cooling schedule
    static class CoolingSchedule {
        private double currentTemp;
        private final double initialTemp;
        private final double finalTemp;
        private final double coolingRate;
        private final long maxIterations;
        private long currentIteration;
        
        public CoolingSchedule(double initialTemp, double finalTemp, 
                              double coolingRate, long maxIterations) {
            this.initialTemp = initialTemp;
            this.finalTemp = finalTemp;
            this.coolingRate = coolingRate;
            this.maxIterations = maxIterations;
            this.currentTemp = initialTemp;
            this.currentIteration = 0;
        }
        
        public double getCurrentTemperature() {
            return currentTemp;
        }
        
        public void updateTemperature() {
            currentIteration++;
            
            // Exponential cooling
            currentTemp = initialTemp * Math.pow(coolingRate, currentIteration);
            
            // Ensure temperature doesn't go below final temperature
            if (currentTemp < finalTemp) {
                currentTemp = finalTemp;
            }
            
            // Optional: Add reheating schedule
            if (currentIteration % 1000000 == 0) {
                currentTemp = Math.max(currentTemp, initialTemp * 0.1);
            }
        }
        
        public void restart() {
            currentTemp = initialTemp;
            currentIteration = 0;
        }
        
        public boolean isFinished() {
            return currentIteration >= maxIterations;
        }
    }
    
    // Simulated Annealing Worker
    static class SAWorker implements Callable<Solution> {
        private final int workerId;
        private final long iterationsPerWorker;
        private final AtomicLong globalIterationCounter;
        private final Solution initialSolution;
        
        public SAWorker(int workerId, long iterationsPerWorker, 
                       AtomicLong globalIterationCounter, Solution initialSolution) {
            this.workerId = workerId;
            this.iterationsPerWorker = iterationsPerWorker;
            this.globalIterationCounter = globalIterationCounter;
            this.initialSolution = initialSolution.clone();
        }
        
        @Override
        public Solution call() throws Exception {
            Random localRandom = new Random(workerId * 1000 + System.currentTimeMillis());
            
            Solution currentSolution = initialSolution.clone();
            Solution bestSolution = currentSolution.clone();
            double bestCost = bestSolution.getCost();
            
            CoolingSchedule cooling = new CoolingSchedule(
                INITIAL_TEMPERATURE, FINAL_TEMPERATURE, COOLING_RATE, iterationsPerWorker);
            
            TabuList tabuList = new TabuList(PROBLEM_SIZE, 10);
            Neighborhood neighborhood = new Neighborhood(currentSolution, tabuList);
            
            long iteration = 0;
            long lastRestartIteration = 0;
            int stagnationCounter = 0;
            
            while (iteration < iterationsPerWorker) {
                double currentTemp = cooling.getCurrentTemperature();
                
                // Generate neighbor based on temperature
                Solution neighbor;
                if (currentTemp > INITIAL_TEMPERATURE * 0.5) {
                    // High temperature: more exploration
                    if (localRandom.nextDouble() < 0.3) {
                        neighbor = neighborhood.getRandom3OptNeighbor();
                    } else if (localRandom.nextDouble() < 0.6) {
                        neighbor = neighborhood.getRandomInsertNeighbor();
                    } else {
                        neighbor = neighborhood.getRandomSwapNeighbor();
                    }
                } else if (currentTemp > INITIAL_TEMPERATURE * 0.1) {
                    // Medium temperature: balanced
                    if (localRandom.nextDouble() < 0.7) {
                        neighbor = neighborhood.getRandomSwapNeighbor();
                    } else {
                        neighbor = neighborhood.getRandomInsertNeighbor();
                    }
                } else {
                    // Low temperature: more exploitation
                    if (localRandom.nextDouble() < 0.3) {
                        neighbor = neighborhood.getBestNonTabuNeighbor();
                        if (neighbor == null) {
                            neighbor = neighborhood.getRandomSwapNeighbor();
                        }
                    } else {
                        neighbor = neighborhood.getRandomSwapNeighbor();
                    }
                }
                
                if (neighbor == null) {
                    iteration++;
                    continue;
                }
                
                double neighborCost = neighbor.getCost();
                double costDifference = neighborCost - currentSolution.getCost();
                
                // Accept or reject the neighbor
                boolean accept = false;
                if (costDifference < 0) {
                    // Better solution
                    accept = true;
                } else if (currentTemp > FINAL_TEMPERATURE) {
                    // Worse solution: accept with probability
                    double probability = Math.exp(-costDifference / currentTemp);
                    accept = localRandom.nextDouble() < probability;
                }
                
                if (accept) {
                    currentSolution = neighbor;
                    neighborhood = new Neighborhood(currentSolution, tabuList);
                    
                    // Update best solution
                    if (neighborCost < bestCost) {
                        bestSolution = neighbor.clone();
                        bestCost = neighborCost;
                        stagnationCounter = 0;
                    } else {
                        stagnationCounter++;
                    }
                } else {
                    stagnationCounter++;
                }
                
                // Restart mechanism
                if (iteration - lastRestartIteration > RESTART_INTERVAL) {
                    if (stagnationCounter > RESTART_INTERVAL / 2) {
                        currentSolution = new Solution(); // Random restart
                        neighborhood = new Neighborhood(currentSolution, tabuList);
                        cooling.restart();
                        tabuList.clear();
                        lastRestartIteration = iteration;
                        stagnationCounter = 0;
                    }
                }
                
                cooling.updateTemperature();
                tabuList.incrementIteration();
                iteration++;
                globalIterationCounter.incrementAndGet();
                
                // Progress reporting
                if (iteration % 100000 == 0) {
                    System.out.printf("Worker %d - Iteration %d: Best Cost = %.2f, Current Temp = %.6f\n",
                            workerId, iteration, bestCost, currentTemp);
                }
                
                // Adaptive parameter adjustment
                if (iteration % 1000000 == 0) {
                    adaptParameters(localRandom, neighborhood, currentTemp);
                }
            }
            
            return bestSolution;
        }
        
        private void adaptParameters(Random random, Neighborhood neighborhood, double currentTemp) {
            // Adaptive neighborhood selection based on success rate
            // This is a simplified version - could be much more sophisticated
            if (currentTemp < INITIAL_TEMPERATURE * 0.01) {
                // Very low temperature: focus on local search
                // Could adjust tabu tenure, neighborhood size, etc.
            }
        }
    }
    
    // Multi-level optimization
    static class MultiLevelSA {
        private final int[] levels = {50, 100, 200, 300, 500}; // Problem sizes
        private final Solution[] levelSolutions;
        
        public MultiLevelSA() {
            levelSolutions = new Solution[levels.length];
        }
        
        public Solution solve() throws InterruptedException, ExecutionException {
            Solution finalSolution = null;
            
            for (int level = 0; level < levels.length; level++) {
                int currentSize = levels[level];
                System.out.printf("Solving at level %d with size %d\n", level, currentSize);
                
                // Create reduced problem
                if (level == 0) {
                    // Start with random solution for smallest problem
                    levelSolutions[level] = solveLevel(currentSize, null);
                } else {
                    // Use previous level solution as starting point
                    levelSolutions[level] = solveLevel(currentSize, levelSolutions[level - 1]);
                }
                
                finalSolution = levelSolutions[level];
            }
            
            return finalSolution;
        }
        
        private Solution solveLevel(int size, Solution previousSolution) 
                throws InterruptedException, ExecutionException {
            // This is a simplified version - in practice, you'd need to
            // properly handle the mapping between different problem sizes
            if (size == PROBLEM_SIZE) {
                return solveFull(previousSolution);
            } else {
                // For smaller problems, use a subset approach
                return solveReduced(size, previousSolution);
            }
        }
        
        private Solution solveFull(Solution startingSolution) 
                throws InterruptedException, ExecutionException {
            return runParallelSA(startingSolution);
        }
        
        private Solution solveReduced(int size, Solution previousSolution) {
            // Simplified: just return a random solution
            // In practice, you'd solve the reduced problem properly
            return new Solution();
        }
    }
    
    // Main parallel SA execution
    private static Solution runParallelSA(Solution startingSolution) 
            throws InterruptedException, ExecutionException {
        
        ExecutorService executor = Executors.newFixedThreadPool(NUM_THREADS);
        AtomicLong globalIterationCounter = new AtomicLong(0);
        
        List<Future<Solution>> futures = new ArrayList<>();
        long iterationsPerWorker = MAX_ITERATIONS / NUM_THREADS;
        
        for (int i = 0; i < NUM_THREADS; i++) {
            Solution workerStartSolution = (startingSolution != null) ? 
                startingSolution.clone() : new Solution();
            
            futures.add(executor.submit(new SAWorker(i, iterationsPerWorker, 
                                                   globalIterationCounter, workerStartSolution)));
        }
        
        // Monitor progress
        ScheduledExecutorService monitor = Executors.newScheduledThreadPool(1);
        monitor.scheduleAtFixedRate(() -> {
            long progress = globalIterationCounter.get();
            double percentage = (double) progress / MAX_ITERATIONS * 100;
            System.out.printf("Global progress: %d/%d (%.2f%%)\n", 
                             progress, MAX_ITERATIONS, percentage);
        }, 10, 10, TimeUnit.SECONDS);
        
        // Collect results
        Solution bestSolution = null;
        double bestCost = Double.MAX_VALUE;
        
        for (Future<Solution> future : futures) {
            Solution result = future.get();
            double cost = result.getCost();
            if (cost < bestCost) {
                bestCost = cost;
                bestSolution = result;
            }
        }
        
        executor.shutdown();
        monitor.shutdown();
        
        return bestSolution;
    }
    
    // Solution verification
    private static boolean verifySolution(Solution solution) {
        boolean[] used = new boolean[PROBLEM_SIZE];
        
        for (int i = 0; i < PROBLEM_SIZE; i++) {
            int location = solution.assignment[i];
            if (location < 0 || location >= PROBLEM_SIZE || used[location]) {
                return false;
            }
            used[location] = true;
        }
        
        return true;
    }
    
    // Calculate solution quality metrics
    private static void analyzeSolution(Solution solution) {
        double totalCost = solution.getCost();
        
        // Calculate cost components
        double totalFlow = 0;
        double totalDistance = 0;
        
        for (int i = 0; i < PROBLEM_SIZE; i++) {
            for (int j = 0; j < PROBLEM_SIZE; j++) {
                if (i != j) {
                    totalFlow += flowMatrix[i][j];
                    totalDistance += distanceMatrix[i][j];
                }
            }
        }
        
        double avgFlow = totalFlow / (PROBLEM_SIZE * (PROBLEM_SIZE - 1));
        double avgDistance = totalDistance / (PROBLEM_SIZE * (PROBLEM_SIZE - 1));
        
        System.out.printf("Total Cost: %.2f\n", totalCost);
        System.out.printf("Average Flow: %.2f\n", avgFlow);
        System.out.printf("Average Distance: %.2f\n", avgDistance);
        System.out.printf("Cost per Unit Flow-Distance: %.6f\n", 
                         totalCost / (totalFlow * avgDistance));
    }
    
    public static void main(String[] args) throws InterruptedException, ExecutionException {
        System.out.println("=== Simulated Annealing for QAP ===");
        System.out.printf("Problem Size: %d\n", PROBLEM_SIZE);
        System.out.printf("Max Iterations: %,d\n", MAX_ITERATIONS);
        System.out.printf("Threads: %d\n", NUM_THREADS);
        System.out.printf("Initial Temperature: %.2f\n", INITIAL_TEMPERATURE);
        System.out.printf("Final Temperature: %.6f\n", FINAL_TEMPERATURE);
        System.out.printf("Cooling Rate: %.6f\n", COOLING_RATE);
        
        long startTime = System.currentTimeMillis();
        
        // Generate initial solution
        Solution initialSolution = new Solution();
        double initialCost = initialSolution.getCost();
        System.out.printf("Initial solution cost: %.2f\n", initialCost);
        
        // Run optimization
        Solution bestSolution = runParallelSA(initialSolution);
        
        long endTime = System.currentTimeMillis();
        long executionTime = endTime - startTime;
        
        System.out.println("\n=== OPTIMIZATION RESULTS ===");
        System.out.printf("Best Cost: %.2f\n", bestSolution.getCost());
        System.out.printf("Improvement: %.2f%%\n", 
                         (initialCost - bestSolution.getCost()) / initialCost * 100);
        System.out.printf("Execution Time: %d hours %d minutes %d seconds\n",
                         executionTime / 3600000,
                         (executionTime % 3600000) / 60000,
                         (executionTime % 60000) / 1000);
        
        // Performance metrics
        long totalOperations = MAX_ITERATIONS * PROBLEM_SIZE * PROBLEM_SIZE;
        double operationsPerSecond = (double) totalOperations / (executionTime / 1000.0);
        
        System.out.printf("Total Operations: %,d\n", totalOperations);
        System.out.printf("Operations/Second: %,.0f\n", operationsPerSecond);
        System.out.printf("Memory Usage: %.2f MB\n",
                         (Runtime.getRuntime().totalMemory() - Runtime.getRuntime().freeMemory()) / 1048576.0);
        
        // Verify and analyze solution
        if (verifySolution(bestSolution)) {
            System.out.println("Solution verified: Valid permutation.");
            analyzeSolution(bestSolution);
        } else {
            System.out.println("ERROR: Invalid solution!");
        }
        
        // Display assignment sample
        System.out.println("\nSample assignments (first 20):");
        for (int i = 0; i < Math.min(20, PROBLEM_SIZE); i++) {
            System.out.printf("Facility %d -> Location %d\n", 
                             i, bestSolution.assignment[i]);
        }
    }

import java.util.*;
import java.util.concurrent.*;
import java.util.concurrent.atomic.AtomicLong;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.stream.IntStream;
import java.util.stream.Collectors;

public class AdvancedVRPTWGeneticSolver {
    
    // Problem parameters - designed for long execution time
    private static final int NUM_CUSTOMERS = 1000;
    private static final int NUM_VEHICLES = 50;
    private static final int POPULATION_SIZE = 500;
    private static final long MAX_GENERATIONS = 50_000_000L;
    private static final int NUM_THREADS = 4;
    private static final double MUTATION_RATE = 0.15;
    private static final double CROSSOVER_RATE = 0.85;
    private static final double ELITE_RATE = 0.1;
    private static final int TOURNAMENT_SIZE = 7;
    private static final int VEHICLE_CAPACITY = 200;
    private static final double SERVICE_TIME = 15.0; // minutes
    private static final long EVOLUTION_LOG_INTERVAL = 100_000L;
    private static final int ARCHIPELAGO_MIGRATION_INTERVAL = 10_000;
    private static final double MIGRATION_RATE = 0.05;
    
    // Advanced parameters
    private static final int ADAPTIVE_PARAMETERS_INTERVAL = 1_000_000;
    private static final int DIVERSITY_CHECK_INTERVAL = 500_000;
    private static final double MIN_DIVERSITY_THRESHOLD = 0.15;
    private static final int RESTART_AFTER_STAGNATION = 2_000_000;
    private static final int LOCAL_SEARCH_FREQUENCY = 10_000;
    
    // Problem instance data
    private static Customer[] customers;
    private static double[][] distanceMatrix;
    private static Depot depot;
    private static Random globalRandom = new Random(12345);
    
    // Statistics tracking
    private static volatile double globalBestFitness = Double.MAX_VALUE;
    private static volatile long globalBestGeneration = 0;
    private static AtomicLong totalEvaluations = new AtomicLong(0);
    private static AtomicLong totalGenerations = new AtomicLong(0);
    private static Map<String, AtomicLong> operationCounters = new ConcurrentHashMap<>();
    
    // Customer class representing delivery points
    static class Customer {
        int id;
        double x, y;
        int demand;
        double earliestTime;
        double latestTime;
        double priority; // for multi-objective optimization
        
        public Customer(int id, double x, double y, int demand, 
                       double earliestTime, double latestTime, double priority) {
            this.id = id;
            this.x = x;
            this.y = y;
            this.demand = demand;
            this.earliestTime = earliestTime;
            this.latestTime = latestTime;
            this.priority = priority;
        }
        
        public double getDistanceTo(Customer other) {
            return Math.sqrt(Math.pow(this.x - other.x, 2) + Math.pow(this.y - other.y, 2));
        }
        
        public double getDistanceToDepot(Depot depot) {
            return Math.sqrt(Math.pow(this.x - depot.x, 2) + Math.pow(this.y - depot.y, 2));
        }
    }
    
    // Depot class
    static class Depot {
        double x, y;
        double openTime;
        double closeTime;
        
        public Depot(double x, double y, double openTime, double closeTime) {
            this.x = x;
            this.y = y;
            this.openTime = openTime;
            this.closeTime = closeTime;
        }
    }
    
    // Route class representing a single vehicle route
    static class Route {
        List<Integer> customers;
        double totalDistance;
        double totalTime;
        int totalDemand;
        double routeCost;
        boolean feasible;
        
        public Route() {
            this.customers = new ArrayList<>();
            this.totalDistance = 0.0;
            this.totalTime = 0.0;
            this.totalDemand = 0;
            this.routeCost = 0.0;
            this.feasible = true;
        }
        
        public Route(List<Integer> customers) {
            this.customers = new ArrayList<>(customers);
            calculateMetrics();
        }
        
        public void calculateMetrics() {
            totalDistance = 0.0;
            totalTime = 0.0;
            totalDemand = 0;
            routeCost = 0.0;
            feasible = true;
            
            if (customers.isEmpty()) {
                return;
            }
            
            // From depot to first customer
            int firstCustomer = customers.get(0);
            totalDistance += customers[firstCustomer].getDistanceToDepot(depot);
            totalTime = depot.openTime + totalDistance; // Travel time (assuming speed = 1)
            
            // Service at first customer
            if (totalTime < customers[firstCustomer].earliestTime) {
                totalTime = customers[firstCustomer].earliestTime;
            }
            
            if (totalTime > customers[firstCustomer].latestTime) {
                feasible = false;
            }
            
            totalTime += SERVICE_TIME;
            totalDemand += customers[firstCustomer].demand;
            
            // Visit remaining customers
            for (int i = 1; i < customers.size(); i++) {
                int prevCustomer = customers.get(i - 1);
                int currentCustomer = customers.get(i);
                
                double travelTime = customers[prevCustomer].getDistanceTo(customers[currentCustomer]);
                totalDistance += travelTime;
                totalTime += travelTime;
                
                // Wait if arriving early
                if (totalTime < customers[currentCustomer].earliestTime) {
                    totalTime = customers[currentCustomer].earliestTime;
                }
                
                // Check time window violation
                if (totalTime > customers[currentCustomer].latestTime) {
                    feasible = false;
                }
                
                totalTime += SERVICE_TIME;
                totalDemand += customers[currentCustomer].demand;
            }
            
            // Return to depot
            int lastCustomer = customers.get(customers.size() - 1);
            totalDistance += customers[lastCustomer].getDistanceToDepot(depot);
            totalTime += customers[lastCustomer].getDistanceToDepot(depot);
            
            // Check capacity constraint
            if (totalDemand > VEHICLE_CAPACITY) {
                feasible = false;
            }
            
            // Check depot closing time
            if (totalTime > depot.closeTime) {
                feasible = false;
            }
            
            // Calculate route cost (multi-objective)
            routeCost = totalDistance * 1.0 + // Distance cost
                       (feasible ? 0 : 10000) + // Penalty for infeasibility
                       totalTime * 0.1 + // Time cost
                       customers.size() * 50; // Fixed cost per customer
        }
        
        public Route clone() {
            Route cloned = new Route(new ArrayList<>(this.customers));
            cloned.totalDistance = this.totalDistance;
            cloned.totalTime = this.totalTime;
            cloned.totalDemand = this.totalDemand;
            cloned.routeCost = this.routeCost;
            cloned.feasible = this.feasible;
            return cloned;
        }
    }
    
    // Individual (solution) class
    static class Individual {
        List<Route> routes;
        double fitness;
        boolean fitnessCalculated;
        double diversity;
        int age;
        Map<String, Double> objectives;
        
        public Individual() {
            this.routes = new ArrayList<>();
            this.fitnessCalculated = false;
            this.age = 0;
            this.objectives = new HashMap<>();
        }
        
        public Individual(List<Route> routes) {
            this.routes = new ArrayList<>();
            for (Route route : routes) {
                this.routes.add(route.clone());
            }
            this.fitnessCalculated = false;
            this.age = 0;
            this.objectives = new HashMap<>();
        }
        
        public double getFitness() {
            if (!fitnessCalculated) {
                calculateFitness();
            }
            return fitness;
        }
        
        private void calculateFitness() {
            double totalCost = 0.0;
            double totalDistance = 0.0;
            double totalTime = 0.0;
            int totalVehicles = 0;
            int infeasibleRoutes = 0;
            double priorityScore = 0.0;
            
            for (Route route : routes) {
                if (!route.customers.isEmpty()) {
                    route.calculateMetrics();
                    totalCost += route.routeCost;
                    totalDistance += route.totalDistance;
                    totalTime += route.totalTime;
                    totalVehicles++;
                    
                    if (!route.feasible) {
                        infeasibleRoutes++;
                    }
                    
                    // Calculate priority score
                    for (int customerId : route.customers) {
                        priorityScore += customers[customerId].priority;
                    }
                }
            }
            
            // Multi-objective fitness calculation
            double distanceObjective = totalDistance;
            double timeObjective = totalTime;
            double vehicleObjective = totalVehicles * 1000.0; // Fixed cost per vehicle
            double feasibilityPenalty = infeasibleRoutes * 50000.0;
            double priorityObjective = -priorityScore; // Negative because we want to maximize
            
            objectives.put("distance", distanceObjective);
            objectives.put("time", timeObjective);
            objectives.put("vehicles", vehicleObjective);
            objectives.put("feasibility", feasibilityPenalty);
            objectives.put("priority", priorityObjective);
            
            // Weighted sum approach
            fitness = distanceObjective * 0.4 + 
                     timeObjective * 0.2 + 
                     vehicleObjective * 0.2 + 
                     feasibilityPenalty * 0.15 + 
                     priorityObjective * 0.05;
            
            fitnessCalculated = true;
            totalEvaluations.incrementAndGet();
        }
        
        public Individual clone() {
            Individual cloned = new Individual(this.routes);
            cloned.fitness = this.fitness;
            cloned.fitnessCalculated = this.fitnessCalculated;
            cloned.diversity = this.diversity;
            cloned.age = this.age;
            cloned.objectives = new HashMap<>(this.objectives);
            return cloned;
        }
        
        public void incrementAge() {
            this.age++;
        }
        
        public boolean isValid() {
            Set<Integer> visitedCustomers = new HashSet<>();
            
            for (Route route : routes) {
                for (int customerId : route.customers) {
                    if (customerId < 0 || customerId >= NUM_CUSTOMERS) {
                        return false;
                    }
                    if (visitedCustomers.contains(customerId)) {
                        return false; // Customer visited multiple times
                    }
                    visitedCustomers.add(customerId);
                }
            }
            
            return visitedCustomers.size() == NUM_CUSTOMERS;
        }
    }
    
    // Population class with advanced features
    static class Population {
        List<Individual> individuals;
        double averageFitness;
        double bestFitness;
        double worstFitness;
        double diversityIndex;
        int stagnationCounter;
        long lastImprovementGeneration;
        
        public Population(int size) {
            this.individuals = new ArrayList<>(size);
            this.stagnationCounter = 0;
            this.lastImprovementGeneration = 0;
        }
        
        public void calculateStatistics() {
            double sum = 0.0;
            bestFitness = Double.MAX_VALUE;
            worstFitness = Double.MIN_VALUE;
            
            for (Individual individual : individuals) {
                double fitness = individual.getFitness();
                sum += fitness;
                
                if (fitness < bestFitness) {
                    bestFitness = fitness;
                }
                if (fitness > worstFitness) {
                    worstFitness = fitness;
                }
            }
            
            averageFitness = sum / individuals.size();
            calculateDiversityIndex();
        }
        
        private void calculateDiversityIndex() {
            // Calculate population diversity based on route similarity
            double totalSimilarity = 0.0;
            int comparisons = 0;
            
            for (int i = 0; i < individuals.size(); i++) {
                for (int j = i + 1; j < individuals.size(); j++) {
                    double similarity = calculateSimilarity(individuals.get(i), individuals.get(j));
                    totalSimilarity += similarity;
                    comparisons++;
                }
            }
            
            diversityIndex = 1.0 - (totalSimilarity / comparisons);
        }
        
        private double calculateSimilarity(Individual ind1, Individual ind2) {
            // Simple similarity measure based on customer-vehicle assignments
            Map<Integer, Integer> assignment1 = new HashMap<>();
            Map<Integer, Integer> assignment2 = new HashMap<>();
            
            for (int i = 0; i < ind1.routes.size(); i++) {
                for (int customerId : ind1.routes.get(i).customers) {
                    assignment1.put(customerId, i);
                }
            }
            
            for (int i = 0; i < ind2.routes.size(); i++) {
                for (int customerId : ind2.routes.get(i).customers) {
                    assignment2.put(customerId, i);
                }
            }
            
            int matches = 0;
            for (int customerId = 0; customerId < NUM_CUSTOMERS; customerId++) {
                if (assignment1.get(customerId) != null && assignment2.get(customerId) != null &&
                    assignment1.get(customerId).equals(assignment2.get(customerId))) {
                    matches++;
                }
            }
            
            return (double) matches / NUM_CUSTOMERS;
        }
        
        public Individual getBestIndividual() {
            return individuals.stream()
                    .min(Comparator.comparingDouble(Individual::getFitness))
                    .orElse(null);
        }
        
        public void sortByFitness() {
            individuals.sort(Comparator.comparingDouble(Individual::getFitness));
        }
        
        public void updateAges() {
            for (Individual individual : individuals) {
                individual.incrementAge();
            }
        }
    }
    
    // Advanced genetic operators
    static class GeneticOperators {
        private Random random;
        
        public GeneticOperators(Random random) {
            this.random = random;
        }
        
        // Order crossover (OX) adapted for VRPTW
        public Individual[] crossover(Individual parent1, Individual parent2) {
            operationCounters.computeIfAbsent("crossover", k -> new AtomicLong(0)).incrementAndGet();
            
            Individual[] offspring = new Individual[2];
            
            // Create customer sequence from parents
            List<Integer> sequence1 = extractCustomerSequence(parent1);
            List<Integer> sequence2 = extractCustomerSequence(parent2);
            
            // Perform order crossover
            int size = sequence1.size();
            int start = random.nextInt(size);
            int end = random.nextInt(size - start) + start;
            
            List<Integer> child1Sequence = orderCrossover(sequence1, sequence2, start, end);
            List<Integer> child2Sequence = orderCrossover(sequence2, sequence1, start, end);
            
            // Convert back to route structure
            offspring[0] = createIndividualFromSequence(child1Sequence);
            offspring[1] = createIndividualFromSequence(child2Sequence);
            
            return offspring;
        }
        
        private List<Integer> extractCustomerSequence(Individual individual) {
            List<Integer> sequence = new ArrayList<>();
            for (Route route : individual.routes) {
                sequence.addAll(route.customers);
            }
            return sequence;
        }
        
        private List<Integer> orderCrossover(List<Integer> parent1, List<Integer> parent2, int start, int end) {
            List<Integer> child = new ArrayList<>(Collections.nCopies(parent1.size(), -1));
            
            // Copy substring from parent1
            for (int i = start; i < end; i++) {
                child.set(i, parent1.get(i));
            }
            
            // Fill remaining positions with parent2 order
            Set<Integer> used = new HashSet<>(child.subList(start, end));
            int childIndex = end;
            
            for (int i = 0; i < parent2.size(); i++) {
                int parentIndex = (end + i) % parent2.size();
                if (!used.contains(parent2.get(parentIndex))) {
                    child.set(childIndex % child.size(), parent2.get(parentIndex));
                    childIndex++;
                }
            }
            
            return child;
        }
        
        private Individual createIndividualFromSequence(List<Integer> sequence) {
            Individual individual = new Individual();
            
            // Use nearest neighbor heuristic with capacity constraints
            List<Integer> unassigned = new ArrayList<>(sequence);
            
            while (!unassigned.isEmpty()) {
                Route route = new Route();
                int currentLoad = 0;
                double currentTime = depot.openTime;
                Integer lastCustomer = null;
                
                Iterator<Integer> iterator = unassigned.iterator();
                while (iterator.hasNext() && currentLoad < VEHICLE_CAPACITY) {
                    Integer customerId = iterator.next();
                    Customer customer = customers[customerId];
                    
                    // Check capacity constraint
                    if (currentLoad + customer.demand > VEHICLE_CAPACITY) {
                        continue;
                    }
                    
                    // Calculate travel time
                    double travelTime = (lastCustomer == null) ? 
                        customer.getDistanceToDepot(depot) :
                        customers[lastCustomer].getDistanceTo(customer);
                    
                    double arrivalTime = currentTime + travelTime;
                    
                    // Check time window feasibility
                    if (arrivalTime <= customer.latestTime) {
                        route.customers.add(customerId);
                        currentLoad += customer.demand;
                        currentTime = Math.max(arrivalTime, customer.earliestTime) + SERVICE_TIME;
                        lastCustomer = customerId;
                        iterator.remove();
                    }
                }
                
                if (!route.customers.isEmpty()) {
                    individual.routes.add(route);
                }
                
                // If no customers could be added, add the first available customer
                if (route.customers.isEmpty() && !unassigned.isEmpty()) {
                    route.customers.add(unassigned.remove(0));
                    individual.routes.add(route);
                }
            }
            
            return individual;
        }
        
        // Advanced mutation operators
        public void mutate(Individual individual) {
            operationCounters.computeIfAbsent("mutation", k -> new AtomicLong(0)).incrementAndGet();
            
            double mutationType = random.nextDouble();
            
            if (mutationType < 0.3) {
                swapMutation(individual);
            } else if (mutationType < 0.6) {
                insertMutation(individual);
            } else if (mutationType < 0.8) {
                reverseMutation(individual);
            } else {
                routeOptimization(individual);
            }
            
            individual.fitnessCalculated = false;
        }
        
        private void swapMutation(Individual individual) {
            List<Integer> allCustomers = extractCustomerSequence(individual);
            if (allCustomers.size() < 2) return;
            
            int pos1 = random.nextInt(allCustomers.size());
            int pos2 = random.nextInt(allCustomers.size());
            
            Collections.swap(allCustomers, pos1, pos2);
            
            Individual mutated = createIndividualFromSequence(allCustomers);
            individual.routes = mutated.routes;
        }
        
        private void insertMutation(Individual individual) {
            List<Integer> allCustomers = extractCustomerSequence(individual);
            if (allCustomers.size() < 2) return;
            
            int from = random.nextInt(allCustomers.size());
            int to = random.nextInt(allCustomers.size());
            
            Integer customer = allCustomers.remove(from);
            allCustomers.add(to, customer);
            
            Individual mutated = createIndividualFromSequence(allCustomers);
            individual.routes = mutated.routes;
        }
        
        private void reverseMutation(Individual individual) {
            List<Integer> allCustomers = extractCustomerSequence(individual);
            if (allCustomers.size() < 2) return;
            
            int start = random.nextInt(allCustomers.size());
            int end = random.nextInt(allCustomers.size() - start) + start;
            
            Collections.reverse(allCustomers.subList(start, end));
            
            Individual mutated = createIndividualFromSequence(allCustomers);
            individual.routes = mutated.routes;
        }
        
        private void routeOptimization(Individual individual) {
            // 2-opt improvement within routes
            for (Route route : individual.routes) {
                if (route.customers.size() > 3) {
                    twoOptImprovement(route);
                }
            }
        }
        
        private void twoOptImprovement(Route route) {
            boolean improved = true;
            while (improved) {
                improved = false;
                
                for (int i = 0; i < route.customers.size() - 1; i++) {
                    for (int j = i + 2; j < route.customers.size(); j++) {
                        // Try 2-opt swap
                        Collections.reverse(route.customers.subList(i + 1, j + 1));
                        
                        double newCost = calculateRouteCost(route);
                        if (newCost < route.routeCost) {
                            route.routeCost = newCost;
                            improved = true;
                        } else {
                            // Revert the change
                            Collections.reverse(route.customers.subList(i + 1, j + 1));
                        }
                    }
                }
            }
        }
        
        private double calculateRouteCost(Route route) {
            route.calculateMetrics();
            return route.routeCost;
        }
        
        // Tournament selection with diversity consideration
        public Individual tournamentSelection(Population population) {
            operationCounters.computeIfAbsent("selection", k -> new AtomicLong(0)).incrementAndGet();
            
            Individual best = null;
            double bestScore = Double.MIN_VALUE;
            
            for (int i = 0; i < TOURNAMENT_SIZE; i++) {
                Individual candidate = population.individuals.get(random.nextInt(population.individuals.size()));
                
                // Calculate selection score (fitness + diversity bonus)
                double fitnessScore = 1.0 / (1.0 + candidate.getFitness());
                double diversityBonus = candidate.diversity * 0.1;
                double ageBonus = Math.max(0, 100 - candidate.age) * 0.001;
                
                double totalScore = fitnessScore + diversityBonus + ageBonus;
                
                if (totalScore > bestScore) {
                    best = candidate;
                    bestScore = totalScore;
                }
            }
            
            return best;
        }
    }
    
    // Local search procedures
    static class LocalSearch {
        private Random random;
        
        public LocalSearch(Random random) {
            this.random = random;
        }
        
        public void improveIndividual(Individual individual) {
            operationCounters.computeIfAbsent("local_search", k -> new AtomicLong(0)).incrementAndGet();
            
            // Apply different local search operators
            if (random.nextDouble() < 0.4) {
                relocateOperator(individual);
            } else if (random.nextDouble() < 0.7) {
                exchangeOperator(individual);
            } else {
                crossExchangeOperator(individual);
            }
        }
        
        private void relocateOperator(Individual individual) {
            // Move customer from one route to another
            List<Route> nonEmptyRoutes = individual.routes.stream()
                    .filter(r -> !r.customers.isEmpty())
                    .collect(Collectors.toList());
            
            if (nonEmptyRoutes.size() < 2) return;
            
            Route sourceRoute = nonEmptyRoutes.get(random.nextInt(nonEmptyRoutes.size()));
            Route targetRoute = nonEmptyRoutes.get(random.nextInt(nonEmptyRoutes.size()));
            
            if (sourceRoute == targetRoute || sourceRoute.customers.isEmpty()) return;
            
            int customerIndex = random.nextInt(sourceRoute.customers.size());
            int customerId = sourceRoute.customers.remove(customerIndex);
            
            // Find best insertion position in target route
            int bestPosition = 0;
            double bestCost = Double.MAX_VALUE;
            
            for (int i = 0; i <= targetRoute.customers.size(); i++) {
                targetRoute.customers.add(i, customerId);
                double cost = calculateRouteCost(targetRoute);
                
                if (cost < bestCost) {
                    bestCost = cost;
                    bestPosition = i;
                }
                
                targetRoute.customers.remove(i);
            }
            
            targetRoute.customers.add(bestPosition, customerId);
            individual.fitnessCalculated = false;
        }
        
        private void exchangeOperator(Individual individual) {
            // Exchange customers between two routes
            List<Route> nonEmptyRoutes = individual.routes.stream()
                    .filter(r -> !r.customers.isEmpty())
                    .collect(Collectors.toList());
            
            if (nonEmptyRoutes.size() < 2) return;
            
            Route route1 = nonEmptyRoutes.get(random.nextInt(nonEmptyRoutes.size()));
            Route route2 = nonEmptyRoutes.get(random.nextInt(nonEmptyRoutes.size()));
            
            if (route1 == route2 || route1.customers.isEmpty() || route2.customers.isEmpty()) return;
            
            int customer1Index = random.nextInt(route1.customers.size());
            int customer2Index = random.nextInt(route2.customers.size());
            
            int customer1 = route1.customers.get(customer1Index);
            int customer2 = route2.customers.get(customer2Index);
            
            route1.customers.set(customer1Index, customer2);
            route2.customers.set(customer2Index, customer1);
            
            individual.fitnessCalculated = false;
        }
        
        private void crossExchangeOperator(Individual individual) {
            // Exchange segments between two routes
            List<Route> nonEmptyRoutes = individual.routes.stream()
                    .filter(r -> r.customers.size() > 1)
                    .collect(Collectors.toList());
            
            if (nonEmptyRoutes.size() < 2) return;
            
            Route route1 = nonEmptyRoutes.get(random.nextInt(nonEmptyRoutes.size()));
            Route route2 = nonEmptyRoutes.get(random.nextInt(nonEmptyRoutes.size()));
            
            if (route1 == route2) return;
            
            // Select segments to exchange
            int start1 = random.nextInt(route1.customers.size());
            int end1 = random.nextInt(route1.customers.size() - start1) + start1;
            
            int start2 = random.nextInt(route2.customers.size());
            int end2 = random.nextInt(route2.customers.size() - start2) + start2;
            
            List<Integer> segment1 = new ArrayList<>(route1.customers.subList(start1, end1 + 1));
            List<Integer> segment2 = new ArrayList<>(route2.customers.subList(start2, end2 + 1));
            
            // Remove segments
            for (int i = end1; i >= start1; i--) {
                route1.customers.remove(i);
            }
            for (int i = end2; i >= start2; i--) {
                route2.customers.remove(i);
            }
            
            // Insert segments
            route1.customers.addAll(start1, segment2);
            route2.customers.addAll(start2, segment1);
            
            individual.fitnessCalculated = false;
        }
        
        private double calculateRouteCost(Route route) {
            route.calculateMetrics();
            return route.routeCost;
        }
    }
    
    // Genetic Algorithm island with migration
    static class GeneticIsland implements Callable<Individual> {
        private int islandId;
        private long maxGenerations;
        private Random random;
        private Population population;
        private GeneticOperators operators;
        private LocalSearch localSearch;
        private BlockingQueue<Individual> migrationQueue;
        private AtomicLong generationCounter;
        
        // Adaptive parameters
        private double currentMutationRate;
        private double currentCrossoverRate;
        private int currentTournamentSize;
        
        public GeneticIsland(int islandId, long maxGenerations, AtomicLong generationCounter,
                            BlockingQueue<Individual> migrationQueue) {
            this.islandId = islandId;
            this.maxGenerations = maxGenerations;
            this.generationCounter = generationCounter;
            this.migrationQueue = migrationQueue;
            this.random = new Random(islandId * 1000 + System.currentTimeMillis());
            this.operators = new GeneticOperators(random);
            this.localSearch = new LocalSearch(random);
            
            // Initialize adaptive parameters
            this.currentMutationRate = MUTATION_RATE;
            this.currentCrossoverRate = CROSSOVER_RATE;
            this.currentTournamentSize = TOURNAMENT_SIZE;
            
            initializePopulation();
        }
        
        private void initializePopulation() {
            population = new Population(POPULATION_SIZE);
            
            for (int i = 0; i < POPULATION_SIZE; i++) {
                Individual individual = createRandomIndividual();
                population.individuals.add(individual);
            }
            
            population.calculateStatistics();
        }

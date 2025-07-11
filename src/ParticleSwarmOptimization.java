import java.util.concurrent.*;
import java.util.Random;
import java.util.ArrayList;
import java.util.List;

public class ParticleSwarmOptimization {
    
    // Bài toán tối ưu hóa phức tạp: Tìm minimum của hàm Rastrigin 100 chiều
    // f(x) = 10*n + Σ(xi² - 10*cos(2π*xi)) với n=100
    private static final int DIMENSIONS = 100;
    private static final int POPULATION_SIZE = 2000;
    private static final int MAX_ITERATIONS = 1000000; // 1 triệu iterations
    private static final double SEARCH_SPACE_MIN = -5.12;
    private static final double SEARCH_SPACE_MAX = 5.12;
    private static final int NUM_THREADS = 4;
    
    // Hàm mục tiêu - Rastrigin function (rất khó tối ưu)
    public static double rastriginFunction(double[] x) {
        double sum = 10.0 * x.length;
        for (int i = 0; i < x.length; i++) {
            sum += x[i] * x[i] - 10.0 * Math.cos(2.0 * Math.PI * x[i]);
        }
        return sum;
    }
    
    // Hàm mục tiêu phức tạp khác - Ackley function
    public static double ackleyFunction(double[] x) {
        double sum1 = 0;
        double sum2 = 0;
        int n = x.length;
        
        for (int i = 0; i < n; i++) {
            sum1 += x[i] * x[i];
            sum2 += Math.cos(2 * Math.PI * x[i]);
        }
        
        return -20 * Math.exp(-0.2 * Math.sqrt(sum1 / n)) - 
               Math.exp(sum2 / n) + 20 + Math.E;
    }
    
    // Particle class
    static class Particle {
        double[] position;
        double[] velocity;
        double[] personalBest;
        double personalBestValue;
        Random random;
        
        public Particle(int dimensions) {
            this.position = new double[dimensions];
            this.velocity = new double[dimensions];
            this.personalBest = new double[dimensions];
            this.random = new Random();
            
            // Khởi tạo position và velocity ngẫu nhiên
            for (int i = 0; i < dimensions; i++) {
                position[i] = SEARCH_SPACE_MIN + random.nextDouble() * 
                             (SEARCH_SPACE_MAX - SEARCH_SPACE_MIN);
                velocity[i] = (random.nextDouble() - 0.5) * 2.0;
                personalBest[i] = position[i];
            }
            
            personalBestValue = rastriginFunction(position);
        }
        
        public void updateVelocity(double[] globalBest, double w, double c1, double c2) {
            for (int i = 0; i < position.length; i++) {
                double r1 = random.nextDouble();
                double r2 = random.nextDouble();
                
                velocity[i] = w * velocity[i] + 
                             c1 * r1 * (personalBest[i] - position[i]) + 
                             c2 * r2 * (globalBest[i] - position[i]);
                
                // Giới hạn velocity
                velocity[i] = Math.max(-2.0, Math.min(2.0, velocity[i]));
            }
        }
        
        public void updatePosition() {
            for (int i = 0; i < position.length; i++) {
                position[i] += velocity[i];
                
                // Giới hạn trong search space
                position[i] = Math.max(SEARCH_SPACE_MIN, 
                                     Math.min(SEARCH_SPACE_MAX, position[i]));
            }
            
            // Cập nhật personal best
            double currentValue = rastriginFunction(position);
            if (currentValue < personalBestValue) {
                personalBestValue = currentValue;
                System.arraycopy(position, 0, personalBest, 0, position.length);
            }
        }
    }
    
    // PSO Worker thread
    static class PSOWorker implements Callable<Double> {
        private final List<Particle> particles;
        private final double[] globalBest;
        private final int startIteration;
        private final int endIteration;
        private final int threadId;
        
        public PSOWorker(List<Particle> particles, double[] globalBest, 
                        int startIteration, int endIteration, int threadId) {
            this.particles = particles;
            this.globalBest = globalBest;
            this.startIteration = startIteration;
            this.endIteration = endIteration;
            this.threadId = threadId;
        }
        
        @Override
        public Double call() throws Exception {
            double bestValue = Double.MAX_VALUE;
            Random random = new Random();
            
            for (int iteration = startIteration; iteration < endIteration; iteration++) {
                // Adaptive parameters
                double w = 0.9 - 0.4 * iteration / MAX_ITERATIONS; // Inertia weight
                double c1 = 2.0; // Cognitive parameter
                double c2 = 2.0; // Social parameter
                
                // Cập nhật particles
                for (Particle particle : particles) {
                    particle.updateVelocity(globalBest, w, c1, c2);
                    particle.updatePosition();
                    
                    if (particle.personalBestValue < bestValue) {
                        bestValue = particle.personalBestValue;
                    }
                }
                
                // Thêm nhiễu để tránh local optima
                if (iteration % 10000 == 0) {
                    for (Particle particle : particles) {
                        if (random.nextDouble() < 0.1) { // 10% chance
                            for (int i = 0; i < DIMENSIONS; i++) {
                                particle.position[i] += (random.nextDouble() - 0.5) * 0.1;
                            }
                        }
                    }
                    
                    System.out.printf("Thread %d - Iteration %d: Best = %.6f\n", 
                                    threadId, iteration, bestValue);
                }
                
                // Điều kiện dừng sớm
                if (bestValue < 1e-6) {
                    System.out.printf("Thread %d converged at iteration %d\n", 
                                    threadId, iteration);
                    break;
                }
            }
            
            return bestValue;
        }
    }
    
    public static void main(String[] args) throws InterruptedException, ExecutionException {
        System.out.println("Starting Particle Swarm Optimization...");
        System.out.printf("Dimensions: %d\n", DIMENSIONS);
        System.out.printf("Population: %d\n", POPULATION_SIZE);
        System.out.printf("Max Iterations: %d\n", MAX_ITERATIONS);
        System.out.printf("Threads: %d\n", NUM_THREADS);
        
        long startTime = System.currentTimeMillis();
        
        // Khởi tạo particles
        List<Particle> particles = new ArrayList<>();
        for (int i = 0; i < POPULATION_SIZE; i++) {
            particles.add(new Particle(DIMENSIONS));
        }
        
        // Tìm global best ban đầu
        double[] globalBest = new double[DIMENSIONS];
        double globalBestValue = Double.MAX_VALUE;
        
        for (Particle particle : particles) {
            if (particle.personalBestValue < globalBestValue) {
                globalBestValue = particle.personalBestValue;
                System.arraycopy(particle.personalBest, 0, globalBest, 0, DIMENSIONS);
            }
        }
        
        // Tạo thread pool
        ExecutorService executor = Executors.newFixedThreadPool(NUM_THREADS);
        
        // Chia công việc cho các threads
        int iterationsPerThread = MAX_ITERATIONS / NUM_THREADS;
        List<Future<Double>> futures = new ArrayList<>();
        
        for (int i = 0; i < NUM_THREADS; i++) {
            int startIter = i * iterationsPerThread;
            int endIter = (i == NUM_THREADS - 1) ? MAX_ITERATIONS : (i + 1) * iterationsPerThread;
            
            // Mỗi thread có subset particles riêng
            List<Particle> threadParticles = particles.subList(
                i * POPULATION_SIZE / NUM_THREADS, 
                (i + 1) * POPULATION_SIZE / NUM_THREADS
            );
            
            futures.add(executor.submit(new PSOWorker(threadParticles, globalBest, 
                                                    startIter, endIter, i)));
        }
        
        // Chờ tất cả threads hoàn thành
        double finalBestValue = Double.MAX_VALUE;
        for (Future<Double> future : futures) {
            double result = future.get();
            if (result < finalBestValue) {
                finalBestValue = result;
            }
        }
        
        executor.shutdown();
        
        long endTime = System.currentTimeMillis();
        long executionTime = endTime - startTime;
        
        System.out.println("\n=== OPTIMIZATION RESULTS ===");
        System.out.printf("Final Best Value: %.10f\n", finalBestValue);
        System.out.printf("Execution Time: %d hours %d minutes %d seconds\n", 
                         executionTime / 3600000, 
                         (executionTime % 3600000) / 60000, 
                         (executionTime % 60000) / 1000);
        
        // Performance metrics
        long totalOperations = (long) MAX_ITERATIONS * POPULATION_SIZE * DIMENSIONS;
        double operationsPerSecond = (double) totalOperations / (executionTime / 1000.0);
        
        System.out.printf("Total Operations: %,d\n", totalOperations);
        System.out.printf("Operations/Second: %,.0f\n", operationsPerSecond);
        System.out.printf("Memory Usage: %.2f MB\n", 
                         (Runtime.getRuntime().totalMemory() - Runtime.getRuntime().freeMemory()) / 1048576.0);
        
        // Hiển thị một số giá trị của solution tốt nhất
        System.out.println("\nSample solution values:");
        for (int i = 0; i < Math.min(10, DIMENSIONS); i++) {
            System.out.printf("x[%d] = %.6f\n", i, globalBest[i]);
        }
    }
}

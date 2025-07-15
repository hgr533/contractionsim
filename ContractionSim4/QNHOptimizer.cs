using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using static TzimtzumSimulation;

namespace ContractionSim4
{
    internal class QNHOptimizer : OptimizerBase
    {
        private double ChaosFactor;
        //private object material;
        private readonly ThreadLocal<Random> ThreadLocalRandom;
        public double BestFitness { get; private set; }
        public new double[] BestSolution { get; private set; }
        public QNHOptimizer(int dimension, double chaosFactor, int? seed = null) : base(dimension, seed)
        {
            if (dimension <= 0) throw new ArgumentException("Dimension must be positive.", nameof(dimension));
            ChaosFactor = Math.Clamp(chaosFactor, 0.1, 10.0);
            BestSolution = new double[dimension]; //Ensure valid initialization
            BestFitness = double.MinValue;

            // Fixed: Create ThreadLocal properly and use instance Random
            ThreadLocalRandom = new ThreadLocal<Random>(() =>
            {
                try
                {
                    return new Random(seed ?? Environment.TickCount + Thread.CurrentThread.ManagedThreadId);
                }
                catch (Exception ex)
                {
                    Console.WriteLine($"Random initialization failed: {ex.Message}");
                    return new Random(); // Fallback
                }
            }, true); // Track all threads
        }

        // This is the required override for the abstract method
        public override (double[] Solution, double Fitness) Optimize(int maxIterations, Random random, double[] bestSolution)
        {
            return OptimizeInternal(maxIterations, random, bestSolution, null);
        }

        public (double[] Solution, double Fitness) OptimizeInternal(int maxIterations, Random random, double[]? bestSolution, double[]? bestSolution1)
        {
            // Handle null bestSolution with a default
            double[] intialSolution = bestSolution ?? bestSolution1 ?? Enumerable.Range(0, Dimension)
                .Select(_ => ThreadLocalRandom.Value?.NextDouble() ?? 0.0 * 100 - 50).ToArray();
            if (intialSolution.Length != Dimension)
                Array.Resize(ref intialSolution, Dimension);
            
            double[] currentSolution = (double[]) intialSolution.Clone();
            double[] velocity = new double[Dimension];
            double time = 0.0;

            for (int iter = 0; iter < maxIterations; iter++)
            {
                if (ThreadLocalRandom.Value == null) throw new InvalidOperationException("ThreadLocalRandom is null.");

                // Hyperbolic transformation with chaotic modulation
                double theta = ChaosFactor * (ThreadLocalRandom.Value.NextDouble() - 0.5);
                double chaosValue = ChaoticPerturbation(iter / (double)maxIterations);

                for (int i = 0; i < Dimension; i++)
                {
                    double hyperbolicShift = Math.Cosh(theta) * currentSolution[i] + Math.Sinh(theta) * velocity[i];
                    currentSolution[i] = hyperbolicShift + chaosValue * (ThreadLocalRandom.Value.NextDouble() - 0.5);
                }

                currentSolution = ClampSolution(currentSolution);

                // Evaluate fitness with multiferroic context
                double fitness = EvaluateFitness(currentSolution, time);

                if (fitness > BestFitness)
                {
                    BestFitness = fitness;
                    Array.Copy(currentSolution, BestSolution, Dimension);
                }

                // Unified velocity update
                for (int i = 0; i < Dimension; i++)
                {
                    velocity[i] = 0.7 * Math.Sinh(theta) * velocity[i] + 0.3 * (BestSolution[i] - currentSolution[i]);
                }

                // Fixed: Use proper time increment (assuming Seconds is a property)
                time += GetTimeIncrement();

                if (iter > 10 && Math.Abs(BestFitness - fitness) < 1e-6) break;
            }

            return (BestSolution, BestFitness);
        }

        private double ChaoticPerturbation(double normalizedTime)
        {
            if (ThreadLocalRandom.Value == null) return 0.0; // Safe default
            double r = 3.9 + 0.1 * Math.Sin(normalizedTime * Math.PI);
            double x = 0.5 + 1e-10;
            for (int i = 0; i < 20; i++)
                x = r * x * (1 - x);
            return Math.Clamp(x, -1.0, 1.0) * ChaosFactor;
        }

        private double[] ClampSolution(double[] solution) =>
            solution.Select(x => Math.Clamp(x, -50.0, 50.0)).ToArray();

        // Helper method to get time increment - adjust based on your PhysicsConstants
        public double GetTimeIncrement()
        {
            // Replace this with your actual PhysicsConstants access
            // For example: return PhysicsConstants.ODEIntegratorStepSize;
            /*return (double)PhysicsConstants.ODEIntegratorStepSize;*/ // Default step size
            return (double)PhysicsConstants.ODEIntegratorStepSize;
        }

        // Fundamental constants
        //LightSpeed = 299792458.0;
        //PlanckLength = 1.616e-35;
        //PlanckConstant = 6.62607015e-34; // J*s
        //GravitationalConstant = 6.67430e-11;
        //Gamma = 0.237;
        //CoulombConstant = 8.99e9;
        //VacuumPermittivity = 8.854187817e-12; // F/m
        //VacuumPermeability = 4 * Math.PI * 1e-7; // H/m
        //BoltzmannConstant = 1.380649e-23; // J/K
        //ElementaryCharge = 1.602176634e-19; // C

        // Material scales
        //DefaultPolarization = 0.1; // C/m^2
        //DefaultMagnetization = 1e-6; // A/m
        //DefaultStrain = 0.0;

        // Simulation parameters
        //ODEIntegratorStepSize = 1e-6;
        //DefaultTimeStep = 1e-6; // seconds

        // Multiferroic coupling
        //TypicalCouplingStrength = 1e-8;

        // Numerical
        //Tolerance = 1e-10;
        //SmallNumber = 1e-12;

        // Hyperbolic transform parameters (for your optimizer if needed)
        //DefaultChaosFactor = 0.5;
        public void SimulatePhysicsStep()
        {
            double dt = GetTimeIncrement();

            // Numerical
            double tolerance = Math.Min(PhysicsConstants.Tolerance, 1e-10);
            double smallNumber = Math.Min(PhysicsConstants.SmallNumber, 1e-12);

            // Simulation parameters
            double defaultTimeStep = Math.Min(PhysicsConstants.DefaultTimeStep, 1e-6);

            // Fundamental constants
            double elementaryCharge = Math.Min(PhysicsConstants.ElementaryCharge, 1.602176634e-19);
            double boltzmannConstant = Math.Min(PhysicsConstants.BoltzmannConstant, 1.380649e-23);
            double vacuumPermeability = Math.Min(PhysicsConstants.VacuumPermeability, 4 * Math.PI * 1e-7);
            double vacuumPermittivity = Math.Min(PhysicsConstants.VacuumPermittivity, 8.854187817e-12);
            double coulombConstant = Math.Min(PhysicsConstants.CoulombConstant, 8.99e9);
            double gamma = Math.Min(PhysicsConstants.Gamma, 0.237);
            double gravitationalConstant = Math.Min(PhysicsConstants.GravitationalConstant, 6.67430e-11);
            double planckConstant = Math.Min(PhysicsConstants.PlanckConstant, 6.62607015e-34);
            double lightSpeed = Math.Min(PhysicsConstants.LightSpeed, 299792458);
            double defaultChaosFactor = Math.Min(PhysicsConstants.DefaultChaosFactor, 0.5);

            // sample field strength within limits
            double electricField = Math.Min(PhysicsConstants.MaxElectricField, 5e5);
            double magneticField = Math.Min(PhysicsConstants.MaxMagneticField, 0.5);
            double stress = Math.Min(PhysicsConstants.MaxMechanicStress, 5e7);

            // apply coupling
            double polarization = PhysicsConstants.DefaultPolarization +
                PhysicsConstants.TypicalCouplingStrength * electricField;

            double magnetization = PhysicsConstants.DefaultMagnetization +
                PhysicsConstants.TypicalCouplingStrength * magneticField;

            double strain = PhysicsConstants.DefaultStrain +
                PhysicsConstants.TypicalCouplingStrength * stress;

            Console.WriteLine($"dt={dt:E3}, P={polarization:E3}, M={magnetization:E3}, S={strain:E3}, T={tolerance:E3}, N={smallNumber:E3}, TS={defaultTimeStep:E3}," +
                $"E={elementaryCharge:E3}, B={boltzmannConstant:E3}, P={vacuumPermeability:E3}, VP={vacuumPermittivity:E3}, C={coulombConstant:E3}, G={gamma:E3}, GC={gravitationalConstant:E3}" +
                $"PC={planckConstant:E3}, LS={lightSpeed:E3}, D={defaultChaosFactor:E3}");
        }
    }
}

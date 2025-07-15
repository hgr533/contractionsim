using System;
using System.Collections.Generic;
using System.Linq;
using System.Numerics;
using System.Threading;
using System.Timers;
using UnitsNet;

namespace ContractionSim4
{
    public abstract class OptimizerBase : IDisposable
    {
        protected Random Random;

        public OptimizerBase(int dimension, int? seed)
        {
            Dimension = dimension;
            Seed = seed;
            Random = new Random(seed ?? Environment.TickCount);
            BestSolution = new double[dimension];
        }

        public int Dimension { get; }
        public int? Seed { get; }
        public double[] BestSolution { get; protected set; }

        // Abstract method that derived classes must implement
        public abstract (double[] Solution, double Fitness) Optimize(int maxIterations, Random random, double[] bestSolution);

        // Virtual method that can be overridden
        protected virtual double EvaluateFitness(double[] solution, double time)
        {
            // Default implementation: minimize distance to origin
            return -solution.Select(x => x * x).Sum();
        }

        public virtual double[] GetBestSolution()
        {
            return (double[])BestSolution.Clone();
        }

        public virtual Random GetRandom()
        {
            return Random;
        }
        public override bool Equals(object? obj)
        {
            return obj is OptimizerBase other &&
                   Dimension == other.Dimension &&
                   BestSolution.SequenceEqual(other.BestSolution);
        }
        public override int GetHashCode()
        {
            return HashCode.Combine(Dimension, BestSolution.Length);
        }

        // IDisposable implementation
        private bool disposed = false;

        protected virtual void Dispose(bool disposing)
        {
            if (!disposed)
            {
                if (disposing)
                {
                    // Dispose managed resources
                }
                disposed = true;
            }
        }
        public void Dispose()
        {
            Dispose(true);
            GC.SuppressFinalize(this);
        }
    }
}
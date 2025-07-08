using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using UnitsNet;
using static TzimtzumSimulation;

namespace ContractionSim4
{
    // Real-time visualization of optimization landscape
    public interface IOptimizationVisualizer
    {
        Merkabah CreateEnergyLandscape(MerkabahThrone throne);

        // Display pre-created data
        void Render(Merkabah data);

        // Convenience methods that do both
        void RenderEnergyLandscape(MerkabahThrone throne)
        {
            var data = CreateEnergyLandscape(throne);
            Render(data);
        }
        object CreatePlotChaoticAttractor(BayesianOptimizer bestchaosSequence);
        void Plot(object data_1);
        void PlotChaoticAttractor(BayesianOptimizer bestchaosSequence)
        {
            var data_1 = CreatePlotChaoticAttractor(bestchaosSequence);
            Plot(data_1);
        }
    }

    //    // Performance analytics
    //    //public class IOptimizationAnalytics : QAOAOptimizer
    //    //{
    //    //    public double ConvergenceRate { get; }
    //    //    public double ExplorationDiversity { get; }
    //    //    public Dictionary<string, double> PhysicsMetrics { get; }

    //    //    public QAOAOptimizer(int rotationSpeed)
    //    //    {
    //    //        ConvergenceRate = rotationSpeed;
    //    //    }
    //    //}
}

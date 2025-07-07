using System;
using System.Collections.Generic;

// Example usage
public class Program
{
    public static void Main()
    {
        Console.WriteLine("=== Multiferroic Material Optimization Examples ===\n");

        // Example 1: Sensor application
        Console.WriteLine("1. Sensor Application:");
        var sensorOptimizer = new MultiferroicOptimizer(ApplicationType.Sensor);
        sensorOptimizer.OptimizeForApplication();

        Console.WriteLine("\n2. Actuator Application:");
        var actuatorOptimizer = new MultiferroicOptimizer(ApplicationType.Actuator);
        actuatorOptimizer.OptimizeForApplication();

        // Example 3: Custom configuration
        Console.WriteLine("\n3. Custom Configuration:");
        var customConfig = new MaterialConfig
        {
            InitialPolarization = 0.15,
            CouplingStrength = 3e-8,
            ElectricFieldAmplitude = 2e3,
            SpatialFrequency = 0.15,
            TemporalFrequency = 0.08
        };
        var customOptimizer = new MultiferroicOptimizer(customConfig);
        customOptimizer.OptimizeForApplication();

        // Example 4: Application switching
        Console.WriteLine("\n4. Application Switching:");
        var adaptiveOptimizer = new MultiferroicOptimizer(ApplicationType.Memory);
        adaptiveOptimizer.OptimizeForApplication();
        adaptiveOptimizer.AdaptToNewApplication(ApplicationType.EnergyHarvesting);
        adaptiveOptimizer.OptimizeForApplication();

        // Example 5: Application with custom tweaks
        Console.WriteLine("\n5. Sensor with Custom Tweaks:");
        var tweakedSensor = MaterialFactory.CreateMaterial(ApplicationType.Sensor, config =>
        {
            config.CouplingStrength *= 2.0;  // Double the coupling
            config.SpatialFrequency = 0.5;   // Higher spatial resolution
        });

        var tweakedOptimizer = new MultiferroicOptimizer(ApplicationType.Custom);
        // Use the tweaked material...
    }
}
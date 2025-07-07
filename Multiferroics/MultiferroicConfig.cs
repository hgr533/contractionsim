using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Multiferroics
{
    // Configuration classes for different applications
    public class MaterialConfig
    {
        public double InitialPolarization { get; set; } = 0.1;
        public double InitialMagnetization { get; set; } = 1e-6;
        public double InitialStrain { get; set; } = 0.0;
        public double CouplingStrength { get; set; } = 1e-8;
        public double ElectricField { get; set; } = 0.0;
        public double MagneticField { get; set; } = 0.0;
        public double MechanicalStress { get; set; } = 0.0;

        // Physical limits for clamping
        public double PolarizationMin { get; set; } = -1e-2;
        public double PolarizationMax { get; set; } = 1e-2;
        public double MagnetizationMin { get; set; } = -1e-5;
        public double MagnetizationMax { get; set; } = 1e-5;
        public double StrainMin { get; set; } = -0.1;
        public double StrainMax { get; set; } = 0.1;

        // Spatial and temporal variation parameters
        public double SpatialFrequency { get; set; } = 0.1;
        public double TemporalFrequency { get; set; } = 0.05;
        public double ElectricFieldAmplitude { get; set; } = 1e3;
        public double MagneticFieldAmplitude { get; set; } = 1e-3;
        public double MechanicalStressAmplitude { get; set; } = 1e6;
    }

    public enum ApplicationType
    {
        General,
        Sensor,
        Actuator,
        Memory,
        EnergyHarvesting,
        Custom
    }

    // Enhanced MultiferroicMaterial class with adaptable parameters
    public class MultiferroicMaterial
    {
        private readonly MaterialConfig _config;

        public double Polarization { get; private set; }
        public double Magnetization { get; private set; }
        public double Strain { get; private set; }
        public double CouplingStrength => _config.CouplingStrength;
        public double ODEIntegratorStepSize { get; private set; } = 1e-6;

        // Constructor with configuration
        public MultiferroicMaterial(MaterialConfig config)
        {
            _config = config ?? throw new ArgumentNullException(nameof(config));
            Polarization = _config.InitialPolarization;
            Magnetization = _config.InitialMagnetization;
            Strain = _config.InitialStrain;
        }

        // Legacy constructor for backward compatibility
        public MultiferroicMaterial(double initialPolarization, double initialMagnetization,
            double initialStrain, double couplingStrength, double electricField,
            double magneticField, double mechanicalStress)
        {
            _config = new MaterialConfig
            {
                InitialPolarization = initialPolarization,
                InitialMagnetization = initialMagnetization,
                InitialStrain = initialStrain,
                CouplingStrength = couplingStrength,
                ElectricField = electricField,
                MagneticField = magneticField,
                MechanicalStress = mechanicalStress
            };

            Polarization = initialPolarization;
            Magnetization = initialMagnetization;
            Strain = initialStrain;
        }

        // Simplified constructor
        public MultiferroicMaterial(double initialPolarization, double electricField,
            double magneticField, double mechanicalStress)
        {
            _config = new MaterialConfig
            {
                InitialPolarization = initialPolarization,
                ElectricField = electricField,
                MagneticField = magneticField,
                MechanicalStress = mechanicalStress
            };

            Polarization = initialPolarization;
            Magnetization = _config.InitialMagnetization;
            Strain = _config.InitialStrain;
        }

        public void UpdateProperties(double electricField, double magneticField, double mechanicalStress)
        {
            // Enhanced coupling model with configuration parameters
            Polarization += _config.CouplingStrength * electricField * Magnetization;
            Magnetization += _config.CouplingStrength * magneticField * Strain;
            Strain += _config.CouplingStrength * mechanicalStress * Polarization;

            // Clamp values using configuration limits
            Polarization = Math.Clamp(Polarization, _config.PolarizationMin, _config.PolarizationMax);
            Magnetization = Math.Clamp(Magnetization, _config.MagnetizationMin, _config.MagnetizationMax);
            Strain = Math.Clamp(Strain, _config.StrainMin, _config.StrainMax);
        }

        public double EnergyDensity(double positionX = 0.0, double positionY = 0.0, double time = 0.0)
        {
            // Enhanced spatial and temporal variation using configuration
            double electricField = Math.Sin(positionX * _config.SpatialFrequency) * _config.ElectricFieldAmplitude;
            double magneticField = Math.Cos(positionX * _config.SpatialFrequency) * _config.MagneticFieldAmplitude;
            double mechanicalStress = Math.Sin(time * _config.TemporalFrequency) * _config.MechanicalStressAmplitude;

            // Update material states
            UpdateProperties(electricField, magneticField, mechanicalStress);

            // Calculate energy density (you can enhance this based on your physics model)
            double electricEnergy = 0.5 * electricField * Polarization;
            double magneticEnergy = 0.5 * magneticField * Magnetization;
            double mechanicalEnergy = 0.5 * mechanicalStress * Strain;
            double couplingEnergy = _config.CouplingStrength * Polarization * Magnetization * Strain;

            return electricEnergy + magneticEnergy + mechanicalEnergy + couplingEnergy;
        }

        // Method to update configuration dynamically
        public void UpdateConfiguration(MaterialConfig newConfig)
        {
            if (newConfig != null)
            {
                // Copy new configuration values
                _config.CouplingStrength = newConfig.CouplingStrength;
                _config.SpatialFrequency = newConfig.SpatialFrequency;
                _config.TemporalFrequency = newConfig.TemporalFrequency;
                _config.ElectricFieldAmplitude = newConfig.ElectricFieldAmplitude;
                _config.MagneticFieldAmplitude = newConfig.MagneticFieldAmplitude;
                _config.MechanicalStressAmplitude = newConfig.MechanicalStressAmplitude;
            }
        }
    }

    // Enhanced factory with application-specific configurations
    public static class MaterialFactory
    {
        private static readonly Dictionary<ApplicationType, MaterialConfig> _applicationConfigs =
            new Dictionary<ApplicationType, MaterialConfig>
            {
                [ApplicationType.General] = new MaterialConfig
                {
                    InitialPolarization = 0.1,
                    CouplingStrength = 1e-8,
                    ElectricFieldAmplitude = 1e3,
                    MagneticFieldAmplitude = 1e-3,
                    MechanicalStressAmplitude = 1e6
                },

                [ApplicationType.Sensor] = new MaterialConfig
                {
                    InitialPolarization = 0.05,  // Lower for higher sensitivity
                    CouplingStrength = 5e-8,     // Higher coupling for better response
                    ElectricFieldAmplitude = 500,  // Lower bias field
                    MagneticFieldAmplitude = 5e-4, // Lower magnetic field
                    MechanicalStressAmplitude = 1e5, // Lower stress for sensitivity
                    SpatialFrequency = 0.2,      // Higher spatial resolution
                    TemporalFrequency = 0.1      // Faster temporal response
                },

                [ApplicationType.Actuator] = new MaterialConfig
                {
                    InitialPolarization = 0.2,   // Higher for larger output
                    CouplingStrength = 2e-7,     // Strong coupling for actuation
                    ElectricFieldAmplitude = 5e3, // High driving field
                    MagneticFieldAmplitude = 5e-3, // Strong magnetic field
                    MechanicalStressAmplitude = 5e6, // High stress capability
                    PolarizationMax = 5e-2,      // Higher limits for actuation
                    StrainMax = 0.2              // Larger strain capability
                },

                [ApplicationType.Memory] = new MaterialConfig
                {
                    InitialPolarization = 0.0,   // Start neutral
                    CouplingStrength = 1e-7,     // Strong for switching
                    ElectricFieldAmplitude = 2e3, // Switching field
                    MagneticFieldAmplitude = 2e-3, // Magnetic write field
                    MechanicalStressAmplitude = 2e6, // Moderate stress
                    TemporalFrequency = 0.01     // Slower for memory stability
                },

                [ApplicationType.EnergyHarvesting] = new MaterialConfig
                {
                    InitialPolarization = 0.1,
                    CouplingStrength = 1e-7,     // Optimized for energy conversion
                    ElectricFieldAmplitude = 1.5e3,
                    MagneticFieldAmplitude = 1.5e-3,
                    MechanicalStressAmplitude = 3e6, // Higher for energy harvesting
                    SpatialFrequency = 0.05,     // Lower for broader coverage
                    TemporalFreequency = 0.2     // Higher frequency for AC harvesting
                }
            };

        // Create material with default configuration
        public static MultiferroicMaterial CreateDefaultMaterial()
        {
            return new MultiferroicMaterial(_applicationConfigs[ApplicationType.General]);
        }

        // Create material for specific application
        public static MultiferroicMaterial CreateMaterial(ApplicationType applicationType)
        {
            if (_applicationConfigs.TryGetValue(applicationType, out var config))
            {
                return new MultiferroicMaterial(config);
            }
            return CreateDefaultMaterial();
        }

        // Create material with custom fields (backward compatibility)
        public static MultiferroicMaterial CreateMaterial(double electricField, double magneticField)
        {
            var config = new MaterialConfig(_applicationConfigs[ApplicationType.General])
            {
                ElectricField = electricField,
                MagneticField = magneticField
            };
            return new MultiferroicMaterial(config);
        }

        // Create material with full custom configuration
        public static MultiferroicMaterial CreateCustomMaterial(MaterialConfig customConfig)
        {
            return new MultiferroicMaterial(customConfig);
        }

        // Create material with application base + custom overrides
        public static MultiferroicMaterial CreateMaterial(ApplicationType baseType,
            Action<MaterialConfig> customizer)
        {
            var config = new MaterialConfig(_applicationConfigs[baseType]);
            customizer?.Invoke(config);
            return new MultiferroicMaterial(config);
        }

        // Get available application types
        public static ApplicationType[] GetAvailableApplications()
        {
            return (ApplicationType[])Enum.GetValues(typeof(ApplicationType));
        }

        // Get configuration for inspection
        public static MaterialConfig GetApplicationConfig(ApplicationType applicationType)
        {
            return _applicationConfigs.TryGetValue(applicationType, out var config)
                ? new MaterialConfig(config)
                : new MaterialConfig();
        }
    }

    // Extension for MaterialConfig copy constructor
    public static class MaterialConfigExtensions
    {
        public static MaterialConfig(this MaterialConfig source, MaterialConfig other)
        {
            if (other == null) return source;

            source.InitialPolarization = other.InitialPolarization;
            source.InitialMagnetization = other.InitialMagnetization;
            source.InitialStrain = other.InitialStrain;
            source.CouplingStrength = other.CouplingStrength;
            source.ElectricField = other.ElectricField;
            source.MagneticField = other.MagneticField;
            source.MechanicalStress = other.MechanicalStress;
            source.SpatialFrequency = other.SpatialFrequency;
            source.TemporalFrequency = other.TemporalFrequency;
            source.ElectricFieldAmplitude = other.ElectricFieldAmplitude;
            source.MagneticFieldAmplitude = other.MagneticFieldAmplitude;
            source.MechanicalStressAmplitude = other.MechanicalStressAmplitude;

            return source;
        }
    }

    // Example usage and optimizer integration
    public class MultiferroicOptimizer
    {
        private MultiferroicMaterial _material;
        private ApplicationType _applicationType;

        public MultiferroicOptimizer(ApplicationType applicationType = ApplicationType.General)
        {
            _applicationType = applicationType;
            _material = MaterialFactory.CreateMaterial(applicationType);
        }

        public MultiferroicOptimizer(MaterialConfig customConfig)
        {
            _applicationType = ApplicationType.Custom;
            _material = MaterialFactory.CreateCustomMaterial(customConfig);
        }

        public void OptimizeForApplication()
        {
            Console.WriteLine($"Optimizing for {_applicationType} application");

            // Example optimization loop
            for (int i = 0; i < 100; i++)
            {
                double x = i * 0.1;
                double t = i * 0.01;

                double energy = _material.EnergyDensity(x, 0, t);

                if (i % 20 == 0)
                {
                    Console.WriteLine($"Step {i}: Energy = {energy:E3}, " +
                        $"P = {_material.Polarization:E3}, " +
                        $"M = {_material.Magnetization:E3}, " +
                        $"S = {_material.Strain:E3}");
                }
            }
        }

        public void AdaptToNewApplication(ApplicationType newType)
        {
            _applicationType = newType;
            _material = MaterialFactory.CreateMaterial(newType);
            Console.WriteLine($"Adapted to {newType} application");
        }
    }
}

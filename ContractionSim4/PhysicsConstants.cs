
internal class PhysicsConstants
{
    // Fundamental constants
    internal static double LightSpeed { get; set; } = 299792458.0;
    internal static double PlanckLength { get; set; } = 1.616e-35;

    internal static double PlanckConstant = 6.62607015e-34; // J*s
    internal static double GravitationalConstant { get; set; } = 6.67430e-11;
    internal static double Gamma { get; set; } = 0.237;
    internal static double CoulombConstant { get; set; } = 8.99e9;
    internal static double VacuumPermittivity { get; set; } = 8.854187817e-12; // F/m
    internal static double VacuumPermeability { get; set; } = 4 * Math.PI * 1e-7; // H/m
    internal static double BoltzmannConstant { get; set; } = 1.380649e-23; // J/K
    internal static double ElementaryCharge { get; set; } = 1.602176634e-19; // C

    // Material scales
    internal static double DefaultPolarization { get; set; } = 0.1; // C/m^2
    internal static double DefaultMagnetization { get; set; } = 1e-6; // A/m
    internal static double DefaultStrain { get; set; } = 0.0;

    // Simulation parameters

    internal static double ODEIntegratorStepSize { get; set; } = 1e-6;
    internal static double DefaultTimeStep { get; set; } = 1e-6; // seconds

    // Multiferroic coupling
    internal static double TypicalCouplingStrength { get; set; } = 1e-8;

    // Numerical
    internal static double Tolerance { get; set; } = 1e-10;
    internal static double SmallNumber { get; set; } = 1e-12;

    // Hyperbolic transform parameters (for your optimizer if needed)
    internal static double DefaultChaosFactor { get; set; } = 0.5;

    // Limits
    internal static double MaxElectricField { get; set; } = 1e6; // V/m
    internal static double MaxMagneticField { get; set; } = 1.0; // T
    internal static double MaxMechanicStress { get; set; } = 1e8; // Pa

}
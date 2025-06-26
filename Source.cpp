#include "TzimtzumSimulation.h"
#include "Engine/World.h"

ATzimtzumSimulation::ATzimtzumSimulation()
{
    PrimaryActorTick.bCanEverTick = true;
    RootScene = CreateDefaultSubobject<USceneComponent>(TEXT("RootScene"));
    RootComponent = RootScene;
}

void ATzimtzumSimulation::BeginPlay()
{
    Super::BeginPlay();
    ChaosFactor = 3.9f;
    BestFitness = -FLT_MAX;
}

void ATzimtzumSimulation::Tick(float DeltaTime)
{
    Super::Tick(DeltaTime);
}

AMerkabahMovement::AMerkabahMovement()
{
    Speed = 340;
    InnerPath = GetWorld()->SpawnActor<AMerkabahMovement>(AMerkabahMovement::StaticClass());
    InnerPath->Speed = Speed / 2;
    Position = FVector2(0, 0);
    Material = FMultiferroicMaterial();
    Name = FString::Printf(TEXT("Movement_%d"), Speed);
}

void AMerkabahMovement::Move(const FVector2& Target, float ElectricField, float MagneticField, float MechanicalStress)
{
    Material.UpdateProperties(ElectricField, MagneticField, MechanicalStress);
    float Efficiency = 1.0f - Material.EnergyDensity() * 1e6f;
    UE_LOG(LogTemp, Log, TEXT("%s moves to %s at %f units/s, inner path at %f units/s, Energy Density: %f"),
        *Name, *Target.ToString(), Speed * Efficiency, InnerPath->Speed * Efficiency, Material.EnergyDensity());
    Position = Target;
}

void FMultiferroicMaterial::UpdateProperties(float ElectricField, float MagneticField, float MechanicalStress)
{
    Polarization += CouplingStrength * ElectricField * Magnetization;
    Magnetization += CouplingStrength * MagneticField * Strain;
    Strain += CouplingStrength * MechanicalStress * Polarization;
    Polarization = FMath::Clamp(Polarization, -0.01f, 0.01f);
    Magnetization = FMath::Clamp(Magnetization, -0.00001f, 0.00001f);
    Strain = FMath::Clamp(Strain, -0.1f, 0.1f);
}

float FMultiferroicMaterial::EnergyDensity() const
{
    const float PermittivityVacuum = 8.854e-12f;
    return 0.5f * (Polarization * Polarization / PermittivityVacuum + Magnetization * Magnetization + Strain * Strain);
}

AMerkabah::AMerkabah()
{
    Throne = GetWorld()->SpawnActor<AMerkabahThrone>(AMerkabahThrone::StaticClass());
}

void AMerkabah::Ascend()
{
    UE_LOG(LogTemp, Log, TEXT("Merkabah ascends."));
}

AMerkabahThrone::AMerkabahThrone()
{
    Wheels.Add(GetWorld()->SpawnActor<AMerkabahWheel>(AMerkabahWheel::StaticClass()));
    Wheels[0]->RotationSpeed = 340;
    Wheels.Add(GetWorld()->SpawnActor<AMerkabahWheel>(AMerkabahWheel::StaticClass()));
    Wheels[1]->RotationSpeed = 212;
    Angels.Add(GetWorld()->SpawnActor<AMerkabahAngel>(AMerkabahAngel::StaticClass()));
    Angels[0]->Name = "Ophanim1"; Angels[0]->EnergyLevel = 100; Angels[0]->Type = "Lion";
    Angels.Add(GetWorld()->SpawnActor<AMerkabahAngel>(AMerkabahAngel::StaticClass()));
    Angels[1]->Name = "Ophanim2"; Angels[1]->EnergyLevel = 100; Angels[1]->Type = "Ox";
    Angels.Add(GetWorld()->SpawnActor<AMerkabahAngel>(AMerkabahAngel::StaticClass()));
    Angels[2]->Name = "Ophanim3"; Angels[2]->EnergyLevel = 100; Angels[2]->Type = "Eagle";
    Angels.Add(GetWorld()->SpawnActor<AMerkabahAngel>(AMerkabahAngel::StaticClass()));
    Angels[3]->Name = "Ophanim4"; Angels[3]->EnergyLevel = 100; Angels[3]->Type = "Man";
}

void AMerkabahThrone::Operate()
{
    UE_LOG(LogTemp, Log, TEXT("Throne operates with %d energy."), 1000);
}

AMerkabahWheel::AMerkabahWheel()
{
    RotationSpeed = 0;
}

AMerkabahAngel::AMerkabahAngel()
{
    Name = "DefaultAngel";
    EnergyLevel = 0;
    Type = "Default";
}

ANPC::ANPC()
{
    Controller = GetWorld()->SpawnActor<AMerkabahAIController>(AMerkabahAIController::StaticClass());
}

void ANPC::Act()
{
    UE_LOG(LogTemp, Log, TEXT("NPC acts."));
}

AMerkabahAIController::AMerkabahAIController()
{
    FMultiferroicMaterial Material;
    Movements.Add(GetWorld()->SpawnActor<AMerkabahMovement>(AMerkabahMovement::StaticClass()));
    Movements[0]->Speed = 340;
    Movements[0]->Material = Material;
    Movements.Add(GetWorld()->SpawnActor<AMerkabahMovement>(AMerkabahMovement::StaticClass()));
    Movements[1]->Speed = 212;
    Movements[1]->Material = Material;
    Perceptions.Add(GetWorld()->SpawnActor<AMerkabahPerception>(AMerkabahPerception::StaticClass()));
    Perceptions[0]->Name = "Vision1"; Perceptions[0]->EnergyLevel = 100; Perceptions[0]->Type = "Vision";
    Perceptions.Add(GetWorld()->SpawnActor<AMerkabahPerception>(AMerkabahPerception::StaticClass()));
    Perceptions[1]->Name = "Hearing1"; Perceptions[1]->EnergyLevel = 100; Perceptions[1]->Type = "Hearing";
    CurrentState = "Patrol";
}

void AMerkabahAIController::Execute()
{
    UE_LOG(LogTemp, Log, TEXT("Controller executes in %s state with %d energy."), *CurrentState, 500);
}

AMerkabahPerception::AMerkabahPerception()
{
    Name = "DefaultPerception";
    EnergyLevel = 0;
    Type = "Default";
}

void AMerkabahPerception::Execute()
{
    UE_LOG(LogTemp, Log, TEXT("Perception executes."));
}

AHumanBrain::AHumanBrain()
{
    EnergySource = nullptr;
    NeuroinflammationLevel = 0;
}

void AHumanBrain::AdjustEfficiencyForInflammation() {}
void AHumanBrain::RegenerateNeurons() {}
void AHumanBrain::ProcessCognitiveTask() {}
void AHumanBrain::ProcessMotorTask() {}

AEnergy::AEnergy()
{
    Value = 0;
}

TPair<TArray<FVector2>, float> ATzimtzumSimulation::OptimizeMerkabahPath(AMerkabahMovement* Movement, const FVector2& Target, int32 MaxIterations, float ElectricField, float MagneticField, float MechanicalStress)
{
    TArray<float> CurrentSolution = { (float)Movement->Position.X, (float)Movement->Position.Y };
    float Beta1 = 0.9f, Beta2 = 0.999f, Epsilon = 1e-8f;
    TArray<float> M = { 0.0f, 0.0f }, V = { 0.0f, 0.0f };
    float Time = 0.0f, BestFitness = FLT_MAX;
    TArray<float> BestSolution = CurrentSolution;

    for (int32 Iter = 0; Iter < MaxIterations; Iter++)
    {
        float Loss = FMath::Sqrt(FMath::Pow(CurrentSolution[0] - Target.X, 2) + FMath::Pow(CurrentSolution[1] - Target.Y, 2));
        Movement->Material.UpdateProperties(ElectricField, MagneticField, MechanicalStress);
        Loss += Movement->Material.EnergyDensity() * 1e6f;

        TArray<float> Gradient = {
            2 * (CurrentSolution[0] - Target.X) / (FMath::Sqrt(FMath::Pow(CurrentSolution[0] - Target.X, 2) + FMath::Pow(CurrentSolution[1] - Target.Y, 2)) + Epsilon),
            2 * (CurrentSolution[1] - Target.Y) / (FMath::Sqrt(FMath::Pow(CurrentSolution[0] - Target.X, 2) + FMath::Pow(CurrentSolution[1] - Target.Y, 2)) + Epsilon)
        };

        for (int32 i = 0; i < 2; i++)
        {
            M[i] = Beta1 * M[i] + (1 - Beta1) * Gradient[i];
            V[i] = Beta2 * V[i] + (1 - Beta2) * Gradient[i] * Gradient[i];
            float MHat = M[i] / (1 - FMath::Pow(Beta1, Iter + 1));
            float VHat = V[i] / (1 - FMath::Pow(Beta2, Iter + 1));
            CurrentSolution[i] -= 0.01f * MHat / (FMath::Sqrt(VHat) + Epsilon);
            CurrentSolution[i] = FMath::Clamp(CurrentSolution[i], -1000.0f, 1000.0f);
        }

        if (Loss < BestFitness)
        {
            BestFitness = Loss;
            BestSolution = CurrentSolution;
        }
        Time += 0.1f;
    }

    TArray<FVector2> Path = { FVector2((int32)BestSolution[0], (int32)BestSolution[1]) };
    return TPair<TArray<FVector2>, float>(Path, BestFitness);
}

float ATzimtzumSimulation::LimitModel(float X, float Boundary) { return X < Boundary ? FLT_MAX : 1.0f / X; }
float ATzimtzumSimulation::ContractionMapping(float X, float K, int32 Iterations)
{
    float Result = X;
    for (int32 i = 0; i < Iterations; i++) Result = K * Result;
    return Result;
}
TPair<float, float> ATzimtzumSimulation::DomainChange(float X, float Y, float Radius)
{
    float Distance = FMath::Sqrt(X * X + Y * Y);
    return TPair<float, float>(Distance >= Radius ? FLT_MAX : 0.0f, FMath::Exp(-(X * X + Y * Y)) * QuantizedArea(Distance));
}
float ATzimtzumSimulation::DerivativeModel(float T) { return -1.0f / (T * T); }
float ATzimtzumSimulation::FourierTransform(float T, float Omega) { return FMath::Sqrt(PI) * FMath::Exp(-Omega * Omega / 4); }
float ATzimtzumSimulation::QuantizedArea(float Distance)
{
    const float PlanckLength = 1.616e-35f;
    const float Gamma = 0.237f;
    int32 QuantizedDistance = (int32)(Distance / PlanckLength);
    return 8 * PI * Gamma * PlanckLength * PlanckLength * FMath::Sqrt(QuantizedDistance * (QuantizedDistance + 1));
}
float ATzimtzumSimulation::AccretionGrowth(float CurrentMass, float MaxMass, float GrowthRate, float Time) { return MaxMass / (1 + FMath::Exp(-GrowthRate * (Time - 1))); }
float ATzimtzumSimulation::ElasticDeflection(float Force, float YoungsModulus, float Length, float MomentInertia, float Position)
{
    return (Force * FMath::Pow(Position, 2) / (6 * YoungsModulus * MomentInertia)) * (3 * Length - Position);
}
float ATzimtzumSimulation::HawkingMassLoss(float Mass, float TimeStep)
{
    const float GravitationalConstant = 6.67430e-11f;
    const float LightSpeed = 299792458.0f;
    const float PlanckConstant = 1.0545718e-34f;
    return Mass - (PlanckConstant * FMath::Pow(LightSpeed, 6) / (15360 * PI * FMath::Pow(GravitationalConstant, 2) * FMath::Pow(Mass, 2))) * TimeStep;
}

void ATzimtzumSimulation::UnifiedFieldEvolution(int32 Cycles)
{
    AMerkabah* Merkabah = GetWorld()->SpawnActor<AMerkabah>(AMerkabah::StaticClass());
    ANPC* Npc = GetWorld()->SpawnActor<ANPC>(ANPC::StaticClass());
    AHumanBrain* Brain = GetWorld()->SpawnActor<AHumanBrain>(AHumanBrain::StaticClass());

    for (int32 Cycle = 0; Cycle < Cycles; Cycle++)
    {
        float CycleDuration = 5.0f, FoliageDensity = 0.5f, HawkingLeak = 0.001f, TotalEnergy = 100.0f,
            WheelSpeed = 340.0f, HarmonyFactor = 395.0f, BrainEnergy = 5000.0f, NeuroStress = 15.0f, MultiferroicEnergy = 10.0f;

        for (int32 i = 0; i < 10; i++)
        {
            float TestDuration = 5.0f + i * 0.5f;
            float TestLoad = CycleDuration + FoliageDensity + HawkingLeak + TotalEnergy / WheelSpeed + MultiferroicEnergy;
            if (TestLoad < 80.0f && TestLoad < CycleDuration) CycleDuration = TestDuration;
        }

        Brain->EnergySource = GetWorld()->SpawnActor<AEnergy>(AEnergy::StaticClass());
        Brain->EnergySource->Value = (int32)BrainEnergy;
        Brain->NeuroinflammationLevel = (int32)NeuroStress;
        Brain->AdjustEfficiencyForInflammation();

        Merkabah->Throne->Wheels[0]->RotationSpeed = (int32)WheelSpeed;
        Merkabah->Throne->Wheels[1]->RotationSpeed = (int32)(WheelSpeed / 2);
        Npc->Controller->Movements[0]->Speed = (int32)WheelSpeed;
        Npc->Controller->Movements[1]->Speed = (int32)(WheelSpeed / 2);

        float ElectricField = 8.99e9f * FoliageDensity / (CycleDuration + 1e-10f); // CoulombConstant
        float MagneticField = 6.67430e-11f * TotalEnergy / (WheelSpeed + 1e-10f); // GravitationalConstant
        float MechanicalStress = ElasticDeflection(5.0f, 1e7f, 0.1f, 1e-8f, 0.1f);
        Npc->Controller->Movements[0]->Move(FVector2(50, 50), ElectricField, MagneticField, MechanicalStress);

        UE_LOG(LogTemp, Log, TEXT("Cycle %d: Load: %f ms"), Cycle, CycleDuration);

        float T = 1.0f / (Cycle + 1);
        float DimensionalFactor = FMath::Exp(-FMath::Loge(10.0f) * T);
        float ContractionRate = DerivativeModel(T) * DimensionalFactor;
        UE_LOG(LogTemp, Log, TEXT("  Tzimtzum contracts field, rate=%f"), ContractionRate);

        float KerrMass = 10.0f;
        float RemainingMass = KerrMass;
        while (RemainingMass > 0) RemainingMass = HawkingMassLoss(RemainingMass, CycleDuration);

        float VoidArea = QuantizedArea(1.0f);
        UE_LOG(LogTemp, Log, TEXT("  Boom! Mass hits zero - new universe with quantized area=%f"), VoidArea);

        float StarMass = AccretionGrowth(0.0f, 1.0f, 0.1f, T);
        for (int32 StarCount = 0; StarCount < 5; StarCount++)
        {
            float Omega = StarCount + 1.0f;
            float FusionGlow = FourierTransform(T, Omega) * FoliageDensity;
            UE_LOG(LogTemp, Log, TEXT("    Star %d forms with mass %f - fusion glow=%f"), StarCount, StarMass, FusionGlow);
        }

        float FoliageDeflection = ElasticDeflection(5.0f, 1e7f, 0.1f, 1e-8f, 0.1f) * (1 - FoliageDensity);
        UE_LOG(LogTemp, Log, TEXT("  Nanite Foliage deflects by %f m"), FoliageDeflection);

        Merkabah->Ascend();
        Npc->Act();
        Brain->RegenerateNeurons();
        if (Cycle % 2 == 0) Brain->ProcessCognitiveTask();
        else Brain->ProcessMotorTask();

        auto [Path, PathFitness] = OptimizeMerkabahPath(Npc->Controller->Movements[0], FVector2(50, 50), 50, ElectricField, MagneticField, MechanicalStress);
        UE_LOG(LogTemp, Log, TEXT("  Optimized Merkabah path fitness: %f"), PathFitness);
    }
}

void ATzimtzumSimulation::UnifiedFieldEvolution2(int32 Cycles)
{
    TArray<float> InitialParams = { 10.0f, 0.5f, 0.0001f, 100.0f, 340.0f, 395.0f, 5000.0f, 15.0f, 10.0f };
    float BestEnergy = FLT_MAX;
    TArray<float> BestParams = InitialParams;

    for (int32 Cycle = 0; Cycle < Cycles; Cycle++)
    {
        float CycleDuration = 5.0f, FoliageDensity = 0.5f, HawkingLeak = 0.001f, TotalEnergy = 100.0f,
            WheelSpeed = 340.0f, HarmonyFactor = 395.0f, BrainEnergy = 5000.0f, NeuroStress = 15.0f, MultiferroicEnergy = 10.0f;

        for (int32 i = 0; i < 10; i++)
        {
            float TestDuration = 5.0f + i * 0.5f;
            float TestLoad = CycleDuration + FoliageDensity + HawkingLeak + TotalEnergy / WheelSpeed + MultiferroicEnergy;
            if (TestLoad < 80.0f && TestLoad < CycleDuration) CycleDuration = TestDuration;
        }

        UE_LOG(LogTemp, Log, TEXT("Cycle %d: Load: %f ms"), Cycle, CycleDuration);
        UE_LOG(LogTemp, Log, TEXT("  Optimized: KerrMass=%f, FoliageDensity=%f, HawkingLeak=%f, Energy=%f, WheelSpeed=%f, Harmony=%f, BrainEnergy=%f, NeuroStress=%f, MultiferroicEnergy=%f, EnergyDev=%f"),
            BestParams[0], FoliageDensity, HawkingLeak, TotalEnergy, WheelSpeed, HarmonyFactor, BrainEnergy, NeuroStress, MultiferroicEnergy, BestEnergy);
    }
}
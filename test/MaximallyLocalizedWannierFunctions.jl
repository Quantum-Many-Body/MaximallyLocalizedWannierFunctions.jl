using LinearAlgebra: norm
using MaximallyLocalizedWannierFunctions
using MaximallyLocalizedWannierFunctions: finitedifferences
using QuantumLattices: Bond, BrillouinZone, Fock, Hilbert, Hopping, Lattice, Onsite, dimension, dtype, periods, reciprocals
using TightBindingApproximation: TBA, optimize!

@testset "finitedifferences" begin
    one = finitedifferences([[1.0]])
    @test one[1] ≈ [0.5, 0.5]
    @test one[2] ≈ [[1.0], [-1.0]]

    square = finitedifferences([[1.0, 0.0], [0.0, 1.0]])
    @test square[1] ≈ [0.5, 0.5, 0.5, 0.5]
    @test square[2] ≈ [[0.0, 1.0], [-0.0, -1.0], [1.0, 0.0], [-1.0, -0.0]]

    hexagon = finitedifferences([[1.0, 0.0], [0.5, √3/2]])
    @test hexagon[1] ≈ [1/3, 1/3, 1/3, 1/3, 1/3, 1/3]
    @test hexagon[2] ≈ [[0.5, √3/2], [-0.5, -√3/2], [-0.5, √3/2], [0.5, -√3/2], [1.0, 0.0], [-1.0, -0.0]]

    rectangle = finitedifferences([[1.0, 0.0], [0.0, 0.6]])
    @test rectangle[1] ≈ [25/18, 25/18, 0.5, 0.5]
    @test rectangle[2] ≈ [[0.0, 0.6], [-0.0, -0.6], [1.0, 0.0], [-1.0, -0.0]]
end

@testset "δ" begin
    wannier = δ([1.0 0.0; 0.0 1.0], [1.0], [1, 2])
    @test wannier([0.0]) ≈ [1 0; 0 1]
    @test wannier([π/2]) ≈ [-1im 0; 0 -1im]
end

@testset "MLWF" begin
    lattice = Lattice([0.0], [0.5]; vectors=[[1.0]])
    hilbert = Hilbert(1=>Fock{:f}(1, 1), 2=>Fock{:f}(1, 1))
    t = Hopping(:t, 1.0, 1)
    Δ = Onsite(:Δ, 0.2, amplitude=bond::Bond->isodd(bond[1].site) ? 1 : -1)
    tba = TBA(lattice, hilbert, (t, Δ))

    mlwf = MLWF(tba, BrillouinZone(reciprocals(lattice), 11), [1], δ([1.0; 0.0;;], [0.0], [2]))
    @test length(mlwf) == 1
    @test count(mlwf) == 11
    @test dimension(mlwf) == 1
    @test periods(mlwf) == (11,)
    @test dtype(mlwf) == Float64

    @test rₙ(mlwf) ≈ [[0.5]]
    @test Ω(mlwf) ≈ Ω₁(mlwf)+Ω₂(mlwf) ≈ sum(map((r, s)->r-norm(s)^2, r²ₙ(mlwf), rₙ(mlwf)))

    optimize!(mlwf; maxiter=20001, verbose=1000, rtol=10^-12)
    @test rₙ(mlwf) ≈ [[0.5]]
    @test Ω(mlwf) ≈ 0.2072121283426236
    @test mlwf([0.0]) ≈ [
        -0.029746433399352297; 0.0017831600771407163;
        0.032361504424905534; -0.005632935030896409;
        -0.04149291044963081; 0.010464791382276278;
        0.06281127483355528; -0.01755511362021527;
        -0.11891373334386489; 0.030463922928913825;
        0.41561002387503665; -0.7805006984480971;
        0.41560965970959984; 0.030464442715804917;
        -0.1189139737018432; -0.01755512724558972;
        0.06281132757963175; 0.010464799504735051;
        -0.041492931194757894; -0.0056329409456026755;
        0.03236151242766532; 0.0017831651296530832
    ]

    ham = Hamiltonian(mlwf)
    @test dimension(ham) == 1
    @test ham(; k=[0.0])[1, 1] ≈ -2.0099751242241757
    @test ham(; k=[π/2])[1, 1] ≈ -1.4257609220682836
end

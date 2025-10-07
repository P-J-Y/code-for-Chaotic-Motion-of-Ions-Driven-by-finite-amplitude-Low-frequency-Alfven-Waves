using ChaosTools
using JLD2
using LaTeXStrings
using Statistics, LinearAlgebra
using Parameters: @with_kw
using CurveFit
using FractalDimensions
using DynamicalSystems
using OrdinaryDiffEq: Vern9 # accessing the ODE solvers
diffeq = (alg = Vern9(), abstol = 1e-12, reltol = 1e-12)


@with_kw mutable struct Args
    t1::Float64 = 1000
    τ::Float64 = 500
end

function ϵ(t, args)
    # ϵ change slowly with time
    if t > args.t1
        return exp(-(t-args.t1)^2 / args.τ^2)
    else
        return 1.0
    end
end

function cpaw_rule_with_ϵ(u,p,t)
    t1 = p[4]; τ = p[5]
    args = Args(t1 = t1, τ = τ)
    Bw = ϵ(t, args)*p[1]; kx = p[2]; kz = p[3]
    α = atan(kx, kz)
    du1 = kx*u[2] + kz*u[4]
    du2 = u[3]*(1+Bw*sin(α)*sin(u[1]))- Bw*u[4]*cos(u[1])
    du3 = - u[2]*(1+Bw*sin(α)*sin(u[1]))- Bw*u[4]*sin(u[1])*cos(α)
    du4 = Bw*cos(u[1])*u[2]+Bw*u[3]*sin(u[1])*cos(α)
    return SVector(du1, du2, du3, du4)
end

function cpaw_rule(u,p,t)
    Bw = p[1]; kx = p[2]; kz = p[3]
    α = atan(kx, kz)
    du1 = kx*u[2] + kz*u[4]
    du2 = u[3]*(1+Bw*sin(α)*sin(u[1]))- Bw*u[4]*cos(u[1])
    du3 = - u[2]*(1+Bw*sin(α)*sin(u[1]))- Bw*u[4]*sin(u[1])*cos(α)
    du4 = Bw*cos(u[1])*u[2]+Bw*u[3]*sin(u[1])*cos(α)
    return SVector(du1, du2, du3, du4)
end

function cpaw_rule_with_xyz(u,p,t)
    Bw = p[1]; kx = p[2]; kz = p[3]
    α = atan(kx, kz)
    du1 = kx*u[2] + kz*u[4]
    du2 = u[3]*(1+Bw*sin(α)*sin(u[1]))- Bw*u[4]*cos(u[1])
    du3 = - u[2]*(1+Bw*sin(α)*sin(u[1]))- Bw*u[4]*sin(u[1])*cos(α)
    du4 = Bw*cos(u[1])*u[2]+Bw*u[3]*sin(u[1])*cos(α)
    du5 = u[2] #x
    du6 = u[3] #y
    du7 = u[4] #z
    return SVector(du1, du2, du3, du4, du5, du6, du7)
end

function B_cpaw(ψ, p)
    # Calculate magnetic field of cpaw
    Bw = p[1]; kx = p[2]; kz = p[3]
    α = atan(kx, kz)
    Bx = -Bw*cos(α)*sin.(ψ)
    By = Bw*cos.(ψ)
    Bz = Bw*sin(α)*sin.(ψ)
    B = NaN * ones(length(ψ), 3)
    B[:,1] = Bx
    B[:,2] = By
    B[:,3] = Bz.+1 # +1 for background field
    return B
end

function B_cpaw_without_B0(ψ, p)
    # 计算cpaw的磁场
    Bw = p[1]; kx = p[2]; kz = p[3]
    α = atan(kx, kz)
    Bx = -Bw*cos(α)*sin.(ψ)
    By = Bw*cos.(ψ)
    Bz = Bw*sin(α)*sin.(ψ)
    B = NaN * ones(length(ψ), 3)
    B[:,1] = Bx
    B[:,2] = By
    B[:,3] = Bz 
    return B
end

function B_cpaw_fieldLine_rule(u,p,t)
    Bw = p[1]; kx = p[2]; kz = p[3]
    α = atan(kx, kz)
    ψ = kx*u[1] + kz*u[3]
    du1 = -Bw*cos(α)*sin(ψ)
    du2 = Bw*cos(ψ)
    du3 = 1+Bw*sin(α)*sin(ψ)
    return SVector(du1, du2, du3)
end

function B_cpaw_fieldLine_rule_inv(u,p,t)
    Bw = p[1]; kx = p[2]; kz = p[3]
    α = atan(kx, kz)
    ψ = kx*u[1] + kz*u[3]
    du1 = -Bw*cos(α)*sin(ψ)
    du2 = Bw*cos(ψ)
    du3 = 1+Bw*sin(α)*sin(ψ)
    return SVector(-du1, -du2, -du3)
end

function temperature(states)
    vm = mean(states) # Calculate the average speed
    v2 = sum(vm[2:end].^2) # Calculate the sum of squares of speeds
    T = 1. - v2
    return T/2
end

function mod2π(x;bias=0)
    return mod.(x .- bias, 2π) .+ bias
end

# 笛卡尔坐标系2球坐标系  Cartesian coordinate system to Spherical coordinate system
function cartesianToSpherical(X)
    r = sqrt.(X[:,2].^2 + X[:,3].^2 + X[:,4].^2)
    theta = acos.(X[:,4] ./ r) # polar angle
    phi = atan.(X[:,3], X[:,2]) # azimuthal angle
    phi = mod2π(phi; bias=0)
    return r, theta, phi
end

function vyLaw(u0,u1)
    # Poincare截面选取的判据 Criteria for PSOS
    if ((u0[3] * u1[3] < 0) || (u1[3] == 0)) && (u0[2] >=0) && (u1[2] >= 0)
        return true
    else
        return false
    end
end

function vxLaw(u0,u1)
    # Poincare截面选取的判据 Criteria for PSOS
    if ((u0[2] * u1[2] < 0) || (u1[2] == 0)) && (u0[3] >=0) && (u1[3] >= 0)
        return true
    else
        return false
    end
end

function psiLaw(u0,u1)
    # Poincare截面选取的判据 Criteria for PSOS
    psi1 = mod2π(u0[1]; bias=0)
    psi2 = mod2π(u1[1]; bias=0)
    if (psi1<5pi/4) || (psi2<5pi/4) || (psi1>7π/4) || (psi2>7π/4)
        return false
    elseif ((psi1 - 3π/2)*(psi2 - 3π/2) < 0) || (psi2 == 3π/2)
        return true
    else
        return false
    end
end
    

# rMin(bw,kx,kz) = sqrt.(kx.^2 .+ kz.^2).*( 1 .+ bw.^2 - 2*bw.*kx./sqrt.(kx.^2 .+ kz.^2) ) ./kz ./bw ./kx
rMin(bw,kx,kz) = sqrt.(kx.^2 .+ kz.^2).*( 1 .+ bw.^2 - 2*bw.*kx./sqrt.(kx.^2 .+ kz.^2) ).^(3/2) ./kz ./bw ./kx
# rMin(bw,kx,kz) = ( 1 .+ bw.^2 - 2*bw.*kx./sqrt.(kx.^2 .+ kz.^2) ) ./kz ./bw
# rMin(bw,kx,kz) = ( 1 .+ bw.^2 - 2*bw.*kx./sqrt.(kx.^2 .+ kz.^2) ).^(3/2) ./kz ./bw

function threshold_curve(kx,kz,C)
    k = sqrt.(kx.^2 .+ kz.^2)
    sinα = kx ./ k
    Δ = ( (2 .+ C*kz).*sinα ).^2 .- 4
    Δ[Δ .< 0.] .= NaN
    Btm = 0.5.*(
        (2 .+ C*kz).*sinα .- sqrt.(Δ)
    )
    Btp = 0.5.*(
        (2 .+ C*kz).*sinα .+ sqrt.(Δ)
    )
    return Btm, Btp
end

figIds = ["(a)", "(b)", "(c)", "(d)", "(e)", "(f)", "(g)", "(h)", "(i)", "(j)",
            "(k)", "(l)", "(m)", "(n)", "(o)", "(p)", "(q)", "(r)", "(s)"]
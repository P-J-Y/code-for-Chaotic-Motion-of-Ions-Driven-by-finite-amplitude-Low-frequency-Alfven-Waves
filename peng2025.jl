include("utils.jl")

#######################################################
# 1. fig1 Lyapunov exponent in two-dimensional parameter space
#######################################################

using Plots
using Plots.PlotMeasures

r=1;
theta = π
phi = 0
ϕ = 0 # Initial phase
Ttr = 100

u0 = [ϕ, r*sin(theta)*cos(phi),
      r*sin(theta)*sin(phi), r*cos(theta)]# ψ x y z
p0 =  [0.2, 0.5, 0.25] # Parameters bw kx kz
cpaw = CoupledODEs(cpaw_rule, u0, p0;diffeq)

bws = 0.:0.002:1.;
kxs = 0.:0.002:1.;
λs = zeros(length(bws), length(kxs))
# Since `DynamicalSystem`s are mutable, we need to copy to parallelize
systems = [deepcopy(cpaw) for _ in 1:Threads.nthreads()-1]
pushfirst!(systems, cpaw)
Threads.@threads for i in eachindex(bws)
    for j in eachindex(kxs)
        system = systems[Threads.threadid()]
        set_parameter!(system, 1, bws[i])
        set_parameter!(system, 2, kxs[j])
        λs[i,j] = lyapunov(system, 10000; Ttr = Ttr, u0=u0)
    end
end
# Save the results
save("data/singleParticle/lyapunov_bw_kx.jld2", "bws", bws, "kxs", kxs, "λs", λs)
# @load "data/singleParticle/lyapunov_bw_kx.jld2" bws kxs λs

figid=1
p1 = heatmap(
    kxs, bws, λs,
    xlims = (0, 1),
    ylims = (0, 1),
    xlabel = L"k_xv_A/\Omega_{i}",
    ylabel = L"B_w/B_0",
    title = L"\lambda_{m}",
    colorbar = false,
    c=:speed,
    framestyle = :box,
    grid = false,
    annotation = ((0.02, 0.98), text(figIds[figid], 12, :top, :left)),
    aspect_ratio = 1,
    clims = (0, 0.11),
    left_margin = 4mm,
    top_margin = 0mm,
)
annotate!(p1, (0.2, 0.1), text("Regular", 12, :left, :blue))
annotate!(p1, (0.6, 0.6), text("Chaos", 12, :left, :white))
cbx = [0.,1.]
cby = 0.:0.001:0.11
cbz = repeat(cby, outer = (1,length(cbx)))
cb1 = heatmap(cbx, cby, cbz,
    color=:speed, grid = false, title = L"\lambda_{m}",
    framestyle = :box, legend=false, colorbar = false, clims = (0, 0.11),
    xlims = (0, 1.),ylims = (0, 0.11),
    yticks = [], xticks=[],
    top_margin = 6mm, bottom_margin = 12mm,
    left_margin = 0mm, right_margin = 0mm,)
cb1_2 = twinx(cb1)
heatmap!(cb1_2,xlims = (0, 1.),ylims = (0, 0.11),
yticks = [0.,0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.1],)
for (i,they) in enumerate([0.,0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.1])
    plot!(cb1, cbx, [they,they], color = :black, linewidth = 1.5)
end
figid += 1

r=1;
theta = π
phi = 0
Nϕ = 1000
ϕ = 0.:2π/Nϕ:(2π-2π/Nϕ) # Initial phase
Δt = 0.025
regularity_T = 2000
lyapunov_T = 10000
Ttr=100;
gali_threshold = 5e-12
chaos_threshold = 0.99
kx = 0.5 # Base kx
kz = 0.25 # Base kz
u0s = [[ϕi, r*sin(theta)*cos(phi),
      r*sin(theta)*sin(phi), r*cos(theta)] for ϕi in ϕ] # ψ x y z

bws = 0.:0.001:0.6;
isChaos = Array{Bool}(undef, length(bws), Nϕ) # 是否是混沌的
# reg = zeros(length(bws), Nϕ) # regularity
λs = zeros(length(bws), Nϕ) # Lyapunov exponent

p0 =  [bws[1], kx, kz] # Parameters bw kx kz t1 t2 τ
cpaw = CoupledODEs(cpaw_rule, u0s[1], p0; diffeq)
systems = [deepcopy(cpaw) for _ in 1:Threads.nthreads()-1]
pushfirst!(systems, cpaw)

Threads.@threads for i in eachindex(bws)
    system = systems[Threads.threadid()]
    set_parameter!(system, 1, bws[i])
    for (ϕi, u0) in enumerate(u0s)
        # reg[i,ϕi] = gali(system, regularity_T, 4; u0=u0,threshold=5e-12)[2][end] / regularity_T
        λs[i,ϕi] = lyapunov(system, lyapunov_T; Ttr = Ttr, u0=u0)
        # if reg[i,ϕi] < chaos_threshold
        if λs[i,ϕi] > 0.01
            isChaos[i,ϕi] = true # 混沌
        else
            isChaos[i,ϕi] = false # 非混沌
        end
    end
    @info "Finished i=$i"
end

cr = sum(isChaos, dims=2) ./ Nϕ # chaos ratio

save("data/coldPlasma/isChaos_bw_from_lyapunov.jld2", "bws", bws, "isChaos", isChaos, "ϕ", ϕ, "cr", cr,
    "λs", λs)
# @load "data/coldPlasma/isChaos_bw_from_lyapunov.jld2" bws isChaos ϕ cr λs
p3 = heatmap(bws, ϕ./pi, λs', xlabel = L"B_w/B_0", title = L"\lambda_{m}",
    ylabel = L"\psi_0"*" [π]", xlims = (0, 0.6),
    ylims = (0, 2),
    color=:speed, grid = false,
    framestyle = :semi, legend=false, colorbar = false, clims = (0, 0.11),
    colorbar_title = L"\lambda_{m}", bottom_margin=0mm,
    annotation = ((0.02, 0.98), text(figIds[figid], 12, :top, :left, :black)),
    aspect_ratio = 0.3,
    left_margin = 0mm,
    top_margin = 0mm,
)
annotate!(p3, (0.85, 0.5), text("CR", 15, :left, :red))
annotate!(p3, (0.075, 0.75), text("Regular", 12, :left, :blue))
annotate!(p3, (0.63, 0.75), text("Chaos", 12, :left, :white))
p3_2 = twinx()
plot!(p3_2, bws, cr, ylabel = nothing, xlims = (0, 0.6), ylims = (0, 1.),
    label = "CR", legend = false,
    linewidth = 2.5, color = :red, grid = false, framestyle = :semi,
    yguidefontcolor=:red,
    yguide_position = :right,
    foreground_color_axis = :red,
    foreground_color_border = :red,
    foreground_color_text = :red,
    aspect_ratio = 0.6,
    )

l = @layout([a b c{0.025w}])
plot(p1,p3,cb1, layout=l,
dpi = 1200,
size = (800, 380),
)
savefig("figure/PRL/lyapunov_bw_kx.png")

###############################################################
# 2.Single-particle magnetic moment and other time series
###############################################################
using Plots
r=1;
theta = π
phi = 0
ϕ = 0 # Initial phase
Δt = 2pi/250
T = 2pi*500+Δt # Total time

bw = 0.5 # Base Bw
kx = 0.5 # Base kx
kz = 0.05 # Base kz
k = sqrt(kx^2 + kz^2) # 计算k的模

u0 = [ϕ, r*sin(theta)*cos(phi),
      r*sin(theta)*sin(phi), r*cos(theta),pi/kx,0,pi/kz]# ψ x y z
p0 =  [bw, kx, kz] # Parameters bw kx kz

cpaw = CoupledODEs(cpaw_rule_with_xyz, u0, p0;diffeq) #L1
step!(cpaw, 0, true) # 先走一段时间，避免暂态过程影响
X_traj, t = trajectory(cpaw, T; Δt = Δt);
X = Matrix(X_traj) # Convert to matrix for easier manipulation

bs = B_cpaw(X[:,1], p0) # 计算磁场
bNorms = vec(sqrt.(sum(abs2, bs[:, 1:3], dims=2))) # 计算磁场的模
# v_parallel = vec( sum(X[:, 2:4] .* bs[:, 1:3], dims=2) ./ bNorms )# 计算平行速度
# μGyro = 0.5 * (1 .- v_parallel.^2) ./ bNorms # 计算磁矩
# p1 = plot(t, bNorms, xlabel = "t", ylabel = L"|B|/B_0",
#     legend = false, grid = false, framestyle = :box,
#     linewidth = 1., color = :black)
# p2 = plot(t, bNorms.^2 ./ kz ./bw, xlabel = "t", ylabel = L"R_c",
#     legend = false, grid = false, framestyle = :box, yscale = :log10,
#     linewidth = 1., color = :black)
# plot(p1, p2, layout = (2,1), size = (600,400))

# gyro period开始的时间
gyro_period_n = Int(round(2pi / Δt ))
gyro_period_start = 1:gyro_period_n:(length(t)-gyro_period_n)
gyro_period_center = gyro_period_start .+ (gyro_period_n÷2)
tGyro = t[gyro_period_start] .+ pi # 计算每个gyro period的中心时间

gyro_period_num = length(gyro_period_start)
μGyro = zeros(gyro_period_num) # 存储每个gyro period的平均磁矩
vzGyro = zeros(gyro_period_num) # 存储每个gyro period的平均vz
bGyro = zeros(gyro_period_num, 3) # 存储每个gyro period的平均磁场
bGyroNorm = zeros(gyro_period_num) # 存储每个gyro period的平均磁场模
vpGyro = zeros(gyro_period_num) # 存储每个gyro period的平均平行速度
vperpGyro = zeros(gyro_period_num) # 存储每个gyro period的平均垂直速度
Reff = zeros(gyro_period_num) # 存储每个gyro period的有效曲率半径
ρ = zeros(gyro_period_num) # 存储每个gyro period的回旋半径

# Threads.@threads for i in 1:gyro_period_num
#     start = gyro_period_start[i]
#     end_ = min(start + gyro_period_n - 1, length(t))
#     μGyro[i] = 0.5*mean(1 .- X[start:end_, 4].^2) # 计算每个gyro period的平均磁场
# end

Threads.@threads for i in 1:gyro_period_num
    start = gyro_period_start[i]
    end_ = min(start + gyro_period_n - 1, length(t))
    thebGyro = vec(mean(bs[start:end_, :], dims=1))
    bGyro[i, :] = thebGyro
    bGyroNorm[i] = norm(thebGyro)
    for j in 1:gyro_period_n
        v_parallel = dot(vec(X[start+j-1, 2:4]), thebGyro) / norm(thebGyro)
        μGyro[i] += 0.5 * (1 - v_parallel^2) / norm(thebGyro) / gyro_period_n
        vpGyro[i] += v_parallel / gyro_period_n
    end
    vzGyro[i] = mean(X[start:end_, 4]) # 计算每个gyro period的平均vz`
end

vperpGyro = sqrt.(1 .- vpGyro.^2) # 计算每个gyro period的平均垂直速度
Reff = k * bGyroNorm.^2 ./ kz ./bw ./ kx # 计算每个gyro period的有效曲率半径
ρ = vperpGyro ./ bGyroNorm # 计算每个gyro period的回旋半径

# 算一下拉雅普诺夫指数
λ = lyapunov(cpaw, 10000; Ttr = 100, u0 = u0)
λ = round(λ, digits=3)

# 计算一下local growth rates LGR, 计算一个回旋周期内的增长率
λlocal = local_growth_rates(cpaw, X_traj[gyro_period_center]; Δt = 2pi)
λmeans = mean(λlocal; dims = 2)

p1 = plot(
    tGyro, μGyro,
    ylabel = L"\mu_{m}^*",
    color = :black,
    framestyle = :box,
    grid = false,
    legend = false,
    # dpi = 1200,
    xlims = (0, t[end]),
    ylims = (0., 1.),
    annotation = ((0.02, 0.98), text(figIds[3], 12, :top, :left), :black)
)
quiver!(p1, [505], [0.37], quiver=([0], [-0.15]), color=:black, linewidth=2)
# quiver!(p1, [660], [0.37], quiver=([0], [-0.15]), color=:black, linewidth=2)
p2 = plot(
    tGyro, vpGyro,
    # xlabel = "Time",
    title = L"B_w/B_0="*"$bw, "*L"k_xv_A/\Omega_i="*"$kx, "*L"k_zv_A/\Omega_i="*"$kz",
    ylabel = L"v_{\parallel}/v_A",
    color = :black,
    framestyle = :box,
    grid = false,
    label = :none,
    legend = :topright,
    # dpi = 1200,
    xlims = (0, t[end]),
    ylims = (-1, 1.),
    annotate = ((0.02, 0.98), text(figIds[1], 12, :top, :left), :black),
)
plot!(
    p2,
    [0, t[end]],
    [0., 0.],
    color = :black,
    linewidth = 1.5,
    linestyle = :dash,
    label = L"v_{\parallel}=0",
)

p3 = plot(
    tGyro, bGyroNorm,
    # xlabel = "Time",
    ylabel = L"|B|/B_0",
    color = :black,
    framestyle = :box,
    grid = false,
    label = :none,
    legend = false,
    # dpi = 1200,
    xlims = (0, t[end]),
    annotate = ((0.02, 0.98), text(figIds[2], 12, :top, :left), :black),
)

p4 = plot(
    tGyro, Reff./ρ,
    xlabel = "t "*L"~[1/\Omega_i]",
    ylabel = L"P_{eff.}=R_c/\rho_i sin\alpha",
    color = :black,
    framestyle = :semi,
    grid = false,
    label = :none,
    legend = :topright,
    # dpi = 1200,
    xlims = (0, t[end]),
    annotate = ((0.02, 0.98), text(figIds[4], 12, :top, :left), :black),
    yscale = :log10,
)
plot!(
    p4,
    [0, t[end]],
    [20, 20],
    color = :black,
    linewidth = 1.5,
    linestyle = :dash,
    label = L"P_{eff.}=20",
)
p4_2 = twinx()
plot!(p4_2, tGyro, λmeans,
    color = :green,
    linewidth = 1.,
    ylabel = L"\lambda_{local}",
    legend = false,
    yguidefontcolor=:green,
    framestyle = :semi,
    grid = false,
    foreground_color_axis = :green,
    foreground_color_border = :green,
    foreground_color_text = :green,
    xlims = (0, t[end]),
    ylims = (0.12,0.3),
)

plot(p2, p3, p1, p4,
    layout = (4, 1),
    dpi = 1200,
    size = (800, 800),
)
savefig("figure/singleParticle/magneticMoment_2_1mark.png")

#############################################################
# 3. fig3(a) Magnetic moment change and effective relative curvature radius under different initial states
##############################################################

r=1;
theta = range(0., pi, length=5); # Initial phase
phi = 0
Nϕ = 10
ϕ = range(0., 2*pi, length=Nϕ); # Initial phase
Δt = 2pi/250
T = 2pi*500+Δt # Total time

bw = 0.5 # Base Bw
kx = 0.5 # Base kx
kz = 0.05 # Base kz
k = sqrt(kx^2 + kz^2) # 计算k的模

u0s = [[ϕi, r*sin(thetai)*cos(phi),
      r*sin(thetai)*sin(phi), 
      r*cos(thetai),
        pi/kx, 0, pi/kz] # ψ x y z
      for ϕi in ϕ for thetai in theta] # ψ x y z
p0 =  [bw, kx, kz] # Parameters bw kx kz

bins = range(10, stop=100, length=21)
bin_centers = (bins[1:end-1] .+ bins[2:end])/2
ΔμGyro_means = zeros(length(bin_centers),length(u0s))
ΔμGyro_stds = zeros(length(bin_centers),length(u0s))

p = plot(
    [20, 20],
    [0, 0.25],
    color = :black,
    linewidth = 1.5,
    linestyle = :dash,
    legend = :topright,
    label = L"P_{eff.}=20",
    xlabel = L"P_{eff.}",
    ylabel = L"\Delta\mu_{m}^*",
    title = L"B_w/B_0="*"$bw, "*L"k_xv_A/\Omega_i="*"$kx, "*L"k_zv_A/\Omega_i="*"$kz",
    framestyle = :box,
    # grid = false,
    dpi = 1200,
    xscale = :log10,
    xlims = (3, 400),
    ylims = (0, 0.2),
)

for u_i in eachindex(u0s)
u0 = u0s[u_i]    
cpaw = CoupledODEs(cpaw_rule_with_xyz, u0, p0;diffeq) #L1
step!(cpaw, 0, true) # 先走一段时间，避免暂态过程影响
X, t = trajectory(cpaw, T; Δt = Δt);
X = Matrix(X) # Convert to matrix for easier manipulation
bs = B_cpaw(X[:,1], p0) # 计算磁场
bNorms = vec(sqrt.(sum(abs2, bs[:, 1:3], dims=2))) # 计算磁场的模
# gyro period开始的时间
gyro_period_n = Int(round(2pi / Δt ))
gyro_period_start = 1:gyro_period_n:(length(t)-gyro_period_n)
tGyro = t[gyro_period_start] .+ pi # 计算每个gyro period的中心时间

gyro_period_num = length(gyro_period_start)
μGyro = zeros(gyro_period_num) # 存储每个gyro period的平均磁矩
vzGyro = zeros(gyro_period_num) # 存储每个gyro period的平均vz
bGyro = zeros(gyro_period_num, 3) # 存储每个gyro period的平均磁场
bGyroNorm = zeros(gyro_period_num) # 存储每个gyro period的平均磁场模
vpGyro = zeros(gyro_period_num) # 存储每个gyro period的平均平行速度
vperpGyro = zeros(gyro_period_num) # 存储每个gyro period的平均垂直速度
Reff = zeros(gyro_period_num) # 存储每个gyro period的有效曲率半径
ρ = zeros(gyro_period_num) # 存储每个gyro period的回旋半径

Threads.@threads for i in 1:gyro_period_num
    start = gyro_period_start[i]
    end_ = min(start + gyro_period_n - 1, length(t))
    thebGyro = vec(mean(bs[start:end_, :], dims=1))
    bGyro[i, :] = thebGyro
    bGyroNorm[i] = norm(thebGyro)
    for j in 1:gyro_period_n
        v_parallel = dot(vec(X[start+j-1, 2:4]), thebGyro) / norm(thebGyro)
        μGyro[i] += 0.5 * (1 - v_parallel^2) / norm(thebGyro) / gyro_period_n
        vpGyro[i] += v_parallel / gyro_period_n
    end
    vzGyro[i] = mean(X[start:end_, 4]) # 计算每个gyro period的平均vz`
end

vperpGyro = sqrt.(1 .- vpGyro.^2) # 计算每个gyro period的平均垂直速度
# Reff = k * bGyroNorm.^2 ./ kz ./bw ./ kx # 计算每个gyro period的有效曲率半径
Reff = k * bGyroNorm.^3 ./ kz ./bw ./ kx # 计算每个gyro period的有效曲率半径
# Reff = k^2 * bGyroNorm.^2 ./ kz ./ kx # 计算每个gyro period的有效曲率半径
# Reff = bGyroNorm.^3 ./ kz ./bw
# Reff = bGyroNorm.^2 ./ kz ./bw
ρ = vperpGyro ./ bGyroNorm # 计算每个gyro period的回旋半径

ΔμGyro = 0.5*(abs.(μGyro[3:end].- μGyro[2:end-1]) .+ abs.(μGyro[1:end-2].- μGyro[2:end-1])) # 计算磁矩的变化
# ΔμGyro = abs.(μGyro[3:end].- μGyro[1:end-2])
scatter!(p,
    Reff[2:end-1]./ρ[2:end-1],
    # Reff[2:end-1],
    ΔμGyro,
    markersize = 0.5,
    markerstrokewidth = 0,
    markerstrokestyle = :none,
    label=:none,
)
# 从0-100分成100份，计算每个区间内的平均值

# bin_means = zeros(length(bin_centers))
# # bin_counts = zeros(length(bin_centers))
# # bin_middles = zeros(length(bin_centers))
# bin_stds = zeros(length(bin_centers))
# for i in 1:20
#     bin_indices = findall((bins[i] .<= Reff[2:end-1]) .& (Reff[2:end-1] .< bins[i+1]))
#     # bin_indices = findall((bins[i] .<= Reff[2:end-1]./bw) .& (Reff[2:end-1]./bw .< bins[i+1]))
#     bin_means[i] = mean(ΔμGyro[bin_indices])
#     # bin_counts[i] = length(bin_indices)
#     if !isempty(bin_indices)
#         # bin_middles[i] = middle(ΔμGyro[bin_indices])
#         bin_stds[i] = std(ΔμGyro[bin_indices],mean=bin_means[i])
#     else
#         bin_stds[i] = NaN
#         # bin_middles[i] = NaN
#     end
# end
# ΔμGyro_means[:,u_i] = bin_means
# ΔμGyro_stds[:,u_i] = bin_stds
@info "Finished u0 $u_i "
end
savefig(p, "figure/singleParticle/dmu/dmu_final_1.png")
#################################################################
# 3. fig3(b) Particle Trajectory
#################################################################
using GLMakie
GLMakie.activate!()
the_title = L"k_xv_A/\Omega_i=0.5,~ k_zv_A/\Omega_i=0.05,~ B_w/B_0=0.5"
cs = [:red,:red,:red,:red]
fig = Figure(; size=(600, 600))
ax1 = Axis3(fig[1, 1]; 
xlabel = L"x\Omega_i/v_A", ylabel = L"y\Omega_i/v_A",
zlabel = L"z\Omega_i/v_A",
xlabelfont = :bold, ylabelfont = :bold, zlabelfont = :bold,
xlabelsize=20,ylabelsize=20,zlabelsize=20,
aspect = (1,1,1),
# title = the_title,
titlesize = 20,
perspectiveness=.2,
elevation = 0.05π,
azimuth = 0.05π,
protrusions = 50
)
# set_ambient_light!(ax1.scene, :white)

# zlims!(ax1, -370, -310) # 0.5_1
# xlims!(ax1, 2, 8)
# ylims!(ax1, -15, 5)
# zlims!(ax1, -490, -430) # 0.5_2
# xlims!(ax1, 2, 8)
# ylims!(ax1, -15, 5)
zlims!(ax1, -710, -220) # 0.5_3
xlims!(ax1, 2, 8)
ylims!(ax1, -30, 15)

GLMakie.lines!(ax1, 
    X[:,5], X[:,6], X[:,7],
    color = :black,
)
GLMakie.lines!(ax1, 
    X1[:,5], X1[:,6], X1[:,7],
    color = :black,
    # linestyle = :dash,
)
GLMakie.lines!(ax1, 
    X2[:,5], X2[:,6], X2[:,7],
    color = :black,
    # linestyle = :dash,
)
GLMakie.lines!(ax1, 
    X3[:,5], X3[:,6], X3[:,7],
    color = :black,
    # linestyle = :dash,
)
GLMakie.lines!(ax1, 
    X4[:,5], X4[:,6], X4[:,7],
    color = :black,
    # linestyle = :dash,
)
GLMakie.lines!(ax1, 
    X5[:,5], X5[:,6], X5[:,7],
    color = :black,
    # linestyle = :dash,
)
GLMakie.lines!(ax1, 
    X6[:,5], X6[:,6], X6[:,7],
    color = :black,
    # linestyle = :dash,
)
GLMakie.lines!(ax1, 
    X7[:,5], X7[:,6], X7[:,7],
    color = :black,
    # linestyle = :dash,
)
GLMakie.lines!(ax1, 
    X8[:,5], X8[:,6], X8[:,7],
    color = :black,
    # linestyle = :dash,
)

u0s = [
    [μx[i], μy[i], μz[i]] for i in [2,]
] # 初始条件，ψ=0, x, y=0, z
push!(u0s, [X4[12000,5], X4[12000,6], X4[12000,7]])
p0 =  [bw, kx, kz] # Parameters bw kx kz
cpawFieldLine = CoupledODEs(B_cpaw_fieldLine_rule, u0s[1], p0)
systems = [deepcopy(cpawFieldLine) for _ in 1:Threads.nthreads()-1]
pushfirst!(systems, cpawFieldLine)
T = 200
Δt = 0.025
N_steps = Int(T/Δt)+1
N_initial_conditions = length(u0s)
bxs = zeros(N_steps, length(u0s)) # 存储x方向的磁场
bys = zeros(N_steps, length(u0s)) # 存储y方向的磁场
bzs = zeros(N_steps, length(u0s)) # 存储z方向的磁场
Threads.@threads for i in eachindex(u0s)
    system = systems[Threads.threadid()]
    local X, _ = trajectory(system, T, u0s[i]; Δt = Δt)
    local X = Matrix(X) # Convert to matrix for easier manipulation
    bxs[:, i] = X[:, 1] # 取出x方向的磁场
    bys[:, i] = X[:, 2] # 取出y方向的磁场
    bzs[:, i] = X[:, 3] # 取出z方向的
end
for i in 1:length(u0s)
    GLMakie.lines!(ax1, 
        bxs[:, i], 
        bys[:, i], 
        bzs[:, i],
        color = cs[i],
        alpha=1.,
    )
end

u0s = [
    [μx[i], μy[i], μz[i]] for i in [2,]
] # 初始条件，ψ=0, x, y=0, z
push!(u0s, [X4[12000,5], X4[12000,6], X4[12000,7]])
p0 =  [bw, kx, kz] # Parameters bw kx kz
cpawFieldLine = CoupledODEs(B_cpaw_fieldLine_rule_inv, u0s[1], p0)
systems = [deepcopy(cpawFieldLine) for _ in 1:Threads.nthreads()-1]
pushfirst!(systems, cpawFieldLine)
T = 500
Δt = 0.025
N_steps = Int(T/Δt)+1
N_initial_conditions = length(u0s)
bxs = zeros(N_steps, length(u0s)) # 存储x方向的磁场
bys = zeros(N_steps, length(u0s)) # 存储y方向的磁场
bzs = zeros(N_steps, length(u0s)) # 存储z方向的磁场
Threads.@threads for i in eachindex(u0s)
    system = systems[Threads.threadid()]
    local X, _ = trajectory(system, T, u0s[i]; Δt = Δt)
    local X = Matrix(X) # Convert to matrix for easier manipulation
    bxs[:, i] = X[:, 1] # 取出x方向的磁场
    bys[:, i] = X[:, 2] # 取出y方向的磁场
    bzs[:, i] = X[:, 3] # 取出z方向的
end
for i in 1:length(u0s)
    GLMakie.lines!(ax1, 
        bxs[:, i], 
        bys[:, i], 
        bzs[:, i],
        color = cs[i],
        alpha=1.,
    )
end
scatter!(ax1, X[1,5], X[1,6], X[1,7], color = :green, markersize=20)
fig



inch = 96
save("figure/singleParticle/magFieldLineSwitch05_3.png",fig,px_per_unit = 300/inch)
#################################################################
# 4. cr in a two-dimensional parameter space and determine the chaos border
#################################################################
# change kz to calculate cr for kz=0.1, 0.25, 0.5
using Plots
using Plots.PlotMeasures

r=1;
theta = range(0., pi, length=40); # Initial phase
phi = 0
# ϕ = 2π*rand(100) # Initial phase
Nϕ = 100
ϕ = range(0., 2*pi, length=Nϕ); # Initial phase
# regularity_T = 2000
# gali_threshold = 5e-12
# chaos_threshold = 0.99
lyapunov_T = 10000
Ttr = 100
kz=0.1 # Base kz

u0s = [[ϕi, r*sin(thetai)*cos(phi),
      r*sin(thetai)*sin(phi), r*cos(thetai)] for ϕi in ϕ for thetai in theta] # ψ x y z
p0 =  [0.2, 0.5, kz] # Parameters bw kx kz
cpaw = CoupledODEs(cpaw_rule, u0s[1], p0; diffeq)

bws = range(0., 1., length=150); # 画图用
kxs = range(0., 1., length=150); 
λs = zeros(length(bws), length(kxs),length(u0s)) # 最大Lyapunov指数
# Since `DynamicalSystem`s are mutable, we need to copy to parallelize
systems = [deepcopy(cpaw) for _ in 1:Threads.nthreads()-1]
pushfirst!(systems, cpaw)
@info "Starting calculation kz=$kz"
for i in eachindex(bws)
    for j in eachindex(kxs)
            Threads.@threads for u_i in eachindex(u0s)
                system = systems[Threads.threadid()]
                set_parameter!(system, 1, bws[i])
                set_parameter!(system, 2, kxs[j])
                λs[i,j,u_i] = lyapunov(system, lyapunov_T; Ttr = Ttr, u0=u0s[u_i])
            end
    end
@info "Finished bwi=$i"
end

# Save the results
@save "data/coldPlasma/lyapunov_bw_kx_kz01.jld2" bws kxs λs

########################################################################

cr_level = 0.01
C = 20
figid = 1
@load "data/coldPlasma/lyapunov_bw_kx_kz01.jld2"
cr = dropdims(mean(λs.>0.01, dims=3),dims=3)
p1 = heatmap(
    kxs, bws, cr,
    xlabel = L"k_xv_A/\Omega_i",
    ylabel = L"B_w/B_0",
    # xlims = (0, 0.4),
    # ylims = (0, 0.6),
    xlims = (0., 1.),
    ylims = (0., 1.),
    aspect_ratio = 1.,
    title = "CR, "*L"k_zv_A/\Omega_i=0.1",
    colorbar = false,
    c=:speed,
    clims = (0, 1),
    # cbar_title = "Lyapunov",
    # cbar_ticks = (-0.5, 0, 0.5),
    dpi = 1200,
    framestyle = :box,
    legend = :topright,
    annotation = ((0.02, 0.98), text(figIds[figid], 12, :top, :left))
)
# scatter!(kxs[1:4:end], min_bws_01[1:4:end], markersize = 3, color = :blue, label = "CR=0.1", marker = :cross)
contour!(p1, kxs, bws, cr, levels = [cr_level], color = :blue, linewidth = 3., label = "CR=0.1", linestyle = :solid,
    legend = :topright,clabels=true)
kz = 0.1
# Btm, Btp = threshold_curve(kxs, kz, C)
# plot!(p1, kxs, Btm; color = :red, linewidth = 1.5, label = L"R_{eff.}^{m}=30",
#     linestyle = :dash, legend = :topright)
# bws = 0.01:0.01:1.
# kxs = 0.01:0.01:1.
BWS = repeat(bws, outer=(1,length(kxs)))
KXS = repeat(kxs', outer=(length(bws), 1))
R = rMin(BWS, KXS, kz)
p1_2 = twinx(p1)
contour!(p1_2, kxs, bws, R, levels = [C], color = :red, linewidth = 1.5, linestyle = :dash,
    legend = :topright, xlims = (0., 1.), ylims = (0., 1.), colorbar = false,
    aspect_ratio = 1.,yticks=:none)
plot!(p1_2,[1.2,1.3],[1.2,1.3], label=L"CR="*"$cr_level", color = :blue, linewidth = 1.5, linestyle = :solid)
plot!(p1_2,[1.2,1.3],[1.2,1.3], label=L"P_{eff.}^m="*"$C", color = :red, linewidth = 1.5, linestyle = :dash)


figid += 1
@load "data/coldPlasma/lyapunov_bw_kx_kz025.jld2"
cr = dropdims(mean(λs.>0.01, dims=3),dims=3)
p2 = heatmap(
    kxs, bws, cr,
    xlabel = L"k_xv_A/\Omega_i",
    ylabel = L"B_w/B_0",
    # xlims = (0, 0.4),
    # ylims = (0, 0.6),
    xlims = (0., 1.),
    ylims = (0., 1.),
    aspect_ratio = 1.,
    title = "CR, "*L"k_zv_A/\Omega_i=0.25",
    colorbar = false,
    c=:speed,
    clims = (0, 1),
    # cbar_title = "Lyapunov",
    # cbar_ticks = (-0.5, 0, 0.5),
    dpi = 1200,
    framestyle = :box,
    legend = :topright,
    annotation = ((0.02, 0.98), text(figIds[figid], 12, :top, :left))
)
# scatter!(kxs[1:8:end], min_bws_025[1:8:end], markersize = 3, color = :blue, label = "CR=0.1", marker = :cross)
contour!(kxs, bws, cr, levels = [cr_level], color = :blue, linewidth = 3., label = "CR=0.1", linestyle = :solid,
    legend = :topright,clabels=true)
kz = 0.25
# Btm, Btp = threshold_curve(kxs, kz, C)
# plot!(p2, kxs, Btm; color = :red, linewidth = 1.5, label = L"R_{eff.}^{m}=30",
#     linestyle = :dash, legend = :topright)
BWS = repeat(bws, outer=(1,length(kxs)))
KXS = repeat(kxs', outer=(length(bws), 1))
R = rMin(BWS, KXS, kz)
p2_2 = twinx(p2)
contour!(p2_2, kxs, bws, R, levels = [C], color = :red, linewidth = 1.5, linestyle = :dash,
    legend = :topright, xlims = (0., 1.), ylims = (0., 1.), colorbar = false,
    aspect_ratio = 1.,yticks=:none)
plot!(p2_2,[1.2,1.3],[1.2,1.3], label=L"CR="*"$cr_level", color = :blue, linewidth = 1.5, linestyle = :solid)
plot!(p2_2,[1.2,1.3],[1.2,1.3], label=L"P_{eff.}^m="*"$C", color = :red, linewidth = 1.5, linestyle = :dash)


figid += 1
@load "data/coldPlasma/lyapunov_bw_kx_kz05.jld2"
cr = dropdims(mean(λs.>0.01, dims=3),dims=3)
p3 = heatmap(
    kxs, bws, cr,
    xlabel = L"k_xv_A/\Omega_i",
    ylabel = L"B_w/B_0",
    # xlims = (0, 0.4),
    # ylims = (0, 0.6),
    xlims = (0., 1.),
    ylims = (0., 1.),
    aspect_ratio = 1.,
    title = "CR, "*L"k_zv_A/\Omega_i=0.5",
    colorbar = false,
    c=:speed,
    clims = (0, 1),
    # cbar_title = "Lyapunov",
    # cbar_ticks = (-0.5, 0, 0.5),
    dpi = 1200,
    framestyle = :box,
    legend = :topright,
    annotation = ((0.02, 0.98), text(figIds[figid], 12, :top, :left))
)
# scatter!(kxs[1:4:end], min_bws_05[1:4:end], markersize = 3, color = :blue, label = "CR=0.1", marker = :cross)
contour!(kxs, bws, cr, levels = [cr_level], color = :blue, linewidth = 3., label = "CR=0.1", linestyle = :solid,
    legend = :topright,clabels=true)
kz = 0.5
# Btm, Btp = threshold_curve(kxs, kz, C)
# plot!(p3, kxs, Btm; color = :red, linewidth = 1.5, label = L"R_{eff.}^{m}=30",
#     linestyle = :dash, legend = :topright)
BWS = repeat(bws, outer=(1,length(kxs)))
KXS = repeat(kxs', outer=(length(bws), 1))
R = rMin(BWS, KXS, kz)
p3_2 = twinx(p3)
contour!(p3_2, kxs, bws, R, levels = [C], color = :red, linewidth = 1.5, linestyle = :dash,
    legend = :topright, xlims = (0., 1.), ylims = (0., 1.), colorbar = false,
    aspect_ratio = 1.,yticks=:none)
plot!(p3_2,[1.2,1.3],[1.2,1.3], label=L"CR="*"$cr_level", color = :blue, linewidth = 1.5, linestyle = :solid)
plot!(p3_2,[1.2,1.3],[1.2,1.3], label=L"P_{eff.}^m="*"$C", color = :red, linewidth = 1.5, linestyle = :dash)


figid += 1
@load "data/coldPlasma/lyapunov_bw_kx_kz01.jld2"
cr = dropdims(mean(λs.>0.01, dims=3),dims=3)
p4 = contour(kxs, bws, cr, levels = [cr_level], color = :black, linewidth = 1.5, label = "ω=0.1", linestyle = :dash,
    legend = :bottomright, clabels=false, cbar= false,
    minorgrid = true,framestyle = :box,
    title = "CR=$cr_level",
    # xlims = (0, 0.4),
    # ylims = (0, 0.6),
    # aspect_ratio = 0.67,
    xlims = (0., 1.),
    ylims = (0., 1.),
    aspect_ratio = 1.,
    xlabel = L"k_xv_A/\Omega_i",
    ylabel = L"B_w/B_0",
    dpi = 1200,
    annotation = ((0.02, 0.98), text(figIds[figid], 12, :top, :left)),
)
@load "data/coldPlasma/lyapunov_bw_kx_kz025.jld2"
cr = dropdims(mean(λs.>0.01, dims=3),dims=3)
contour!(kxs, bws, cr, levels = [0.05], color = :red, linewidth = 1.5, label = "ω=0.25", linestyle = :dash,
    legend = :bottomright, clabels=false,cbar= false,)
@load "data/coldPlasma/lyapunov_bw_kx_kz05.jld2"
cr = dropdims(mean(λs.>0.01, dims=3),dims=3)
contour!(kxs, bws, cr, levels = [0.05], color = :blue, linewidth = 1.5, label = "ω=0.5", linestyle = :dash,
    legend = :bottomright, clabels=false,cbar= false,)
plot!([1.2,1.3],[1.2,1.3], label=L"k_zv_A/\Omega_i=0.1", color = :black, linewidth = 1.5, linestyle = :dash, legend=:topright)
plot!([1.2,1.3],[1.2,1.3], label=L"k_zv_A/\Omega_i=0.25", color = :red, linewidth = 1.5, linestyle = :dash)
plot!([1.2,1.3],[1.2,1.3], label=L"k_zv_A/\Omega_i=0.5", color = :blue, linewidth = 1.5, linestyle = :dash)

cbx = [0.,1.]
cby = 0.:0.01:1.
cbz = repeat(cby, outer = (1,length(cbx)))
cb3 = heatmap(cbx, cby, cbz,
    color=:speed, grid = false, title = "CR",
    framestyle = :box, legend=false, colorbar = false, clims = (0, 1.),
    xlims = (0, 1.),ylims = (0, 1.),
    yticks = [], xticks=[],
    top_margin = 2mm, bottom_margin = 12mm,
    left_margin = 0mm, right_margin = 0mm,)
cb3_2 = twinx(cb3)
heatmap!(cb3_2,xlims = (0, 1.),ylims = (0, 1.),
yticks = [0.,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.],)
for (i,they) in enumerate([0.,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.])
    plot!(cb3, cbx, [they,they], color = :black, linewidth = 1.5)
end

l = @layout [grid(2, 2) a{0.025w}]
plot(p1,p2,p3,p4,cb3, layout=l,
dpi = 1200,
size = (700, 680),
)
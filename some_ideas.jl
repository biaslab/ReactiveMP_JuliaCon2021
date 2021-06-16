### A Pluto.jl notebook ###
# v0.14.8

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    quote
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : missing
        el
    end
end

# ╔═╡ bdf461c7-918d-4f78-968d-aa3dfd11d3b0
begin 
	using Pkg
	Pkg.activate(".")
	import DataStructures: CircularBuffer
	
	using MacroTools, PlutoUI, Distributions
	using ColorSchemes
	using Rocket, GraphPPL, ReactiveMP
	
	import Plots
end

# ╔═╡ 773890b4-f844-4a2b-946b-b56f7f6ac375
begin
	using JSServe
	Page()
end

# ╔═╡ 3f028a44-cdd0-11eb-2fb0-6b696babeb9b
begin
	using WGLMakie
	WGLMakie.set_theme!(resolution = (1300, 800))
end

# ╔═╡ 08ec8032-40ba-4bef-9dce-53389a531cab
make_subscription_reference() = Ref{Rocket.Teardown}(voidTeardown)

# ╔═╡ a91b698e-24fc-47d0-9742-b6a151eacdd3
npoints = 30

# ╔═╡ 12f072b6-1e9d-42d0-9784-15ab2ed32e1c
begin
	
	local f = Figure()
	
	local colors = collect(cgrad(:Blues_9, npoints))
	local sizes  = range(2, 6, length=npoints)
	
	real_states  = Node(map(_ -> Point2f0(rand(), rand()), 1:npoints))
	noisy_states = Node(map(_ -> Point2f0(0.0, 0.0), 1:npoints))
	
	scatter(f[1, 1], noisy_states, markersize = sizes, color = :green)
	lines!(f[1, 1], real_states, linewidth = sizes, color = colors)
	limits!(-50, 50, -50, 50)
	
	states_dim1 = map(i -> map(r -> r[1], i), real_states)
	states_dim2 = map(i -> map(r -> r[2], i), real_states)
	
	noisy_dim1 = map(i -> map(r -> r[1], i), noisy_states)
	noisy_dim2 = map(i -> map(r -> r[2], i), noisy_states)

	scatter(f[1, 2][1, 1], noisy_dim1, markersize = sizes, color = :green)
	lines!(f[1, 2][1, 1], states_dim1, color = colors)
	limits!(0, npoints, -50, 50)
	
	scatter(f[1, 2][2, 1], noisy_dim2, markersize = sizes, color = :green)
	lines!(f[1, 2][2, 1], states_dim2, color = colors)
	limits!(0, npoints, -50, 50)
	
	
	fig = current_figure()
end

# ╔═╡ aa3f4b38-ff13-492b-9eb3-bda2c80f5c5d
@bind speed PlutoUI.Slider(10:100, default = 50, show_value = true)

# ╔═╡ 320d3d53-bf88-4be0-a189-a3f1e8c4b24a
stream = timer(100, Int(10 * round(100 / speed)))

# ╔═╡ 5be725e6-732d-42e3-8967-3e8251946d72
check_box = @bind is_subscribed CheckBox(default = false)

# ╔═╡ c189ddad-c8f6-45c4-9cec-8d72a09f98cf
md"Subscribe? $(check_box)"

# ╔═╡ 61952b04-cc7e-46f2-b273-854e425866ca
subscription = make_subscription_reference()

# ╔═╡ a814a9fe-6c76-4e75-9b0d-7141e18d1f9d
md"""
## Smoothing model
"""

# ╔═╡ a327b4f6-b8b9-4536-b45d-c13b767a3606
@model function full_graph(npoints, A, P, Q)
	
	x = randomvar(npoints)
	y = datavar(Vector{Float64}, npoints)
	
	x[1] ~ MvNormalMeanCovariance([ 0.0, 0.0 ], [ 100.0 0.0; 0.0 100.0 ])
	y[1] ~ MvNormalMeanCovariance(x[1], Q)
	
	for i in 2:npoints
		x[i] ~ MvNormalMeanCovariance(A * x[i - 1], P)
		y[i] ~ MvNormalMeanCovariance(x[i], Q)
	end
	
	return x, y
end

# ╔═╡ 65894eb0-99a6-4d29-a23d-bbe1dab4a2e8
function inference_full_graph(data, A, P, Q)

	data    = convert(AbstractVector{Vector{Float64}}, data)
	npoints = length(data)
	
	model, (x, y) = full_graph(npoints, A, P, Q, options = (
		limit_stack_depth = 500,
	))
	
	xbuffer    = buffer(Marginal, npoints)
	xmarginals = getmarginals(x)
	
	subscription = subscribe!(xmarginals, xbuffer)
	
	ReactiveMP.update!(y, data)
	
	unsubscribe!(subscription)
	
	return getvalues(xbuffer)
end

# ╔═╡ 21b44508-ec16-4f15-9ba1-97fce4667873


# ╔═╡ 39b43b20-915b-44a6-947e-170234eae68a
md"""
## Utilities
"""

# ╔═╡ e738bbf7-8a66-411c-82f9-9716f749f30b
md"""
`DataGenerationProcess` encapsulates simple data generation process with a rotation matrix as a state transition matrix. States modelled with gaussian distributions and have their oun state transition noise. Observations are connected to states with Gaussian node as well and have extra noise.
"""

# ╔═╡ 649214d2-b661-49f5-ac36-50da9e358675
struct DataGenerationProcess
	angle        :: Float64
	states       :: CircularBuffer{Point2f0}
	observations :: CircularBuffer{Point2f0}
	
	state_transition_matrix :: Matrix{Float64}
	state_transition_noise  :: Matrix{Float64}
	observation_noise       :: Matrix{Float64}

	function DataGenerationProcess(angle::Float64, npoints, snoise, onoise)
		states       = CircularBuffer{Point2f0}(npoints)
		observations = CircularBuffer{Point2f0}(npoints)
		
		initial_state       = Point2f0(5.0, 0.0)
		initial_observation = rand(MvNormal(initial_state, onoise))
		
		push!(states, initial_state)
		push!(observations, initial_observation)
		
		A = [ cos(angle) -sin(angle); sin(angle) cos(angle) ]
		
		return new(angle, states, observations, A, snoise, onoise)
	end
end

# ╔═╡ 8de8cebd-b57e-438c-9cae-caf245c6a74b
data = DataGenerationProcess(π / 30, npoints, [ 10.0 0.0; 0.0 10.0 ])

# ╔═╡ de00e97d-e8a8-46a1-a144-04925a078e5a
data_stream = stream |> 
	map_to(DataGenerationProcess(π / 30, npoints, [ 10.0 0.0; 0.0 10.0 ])) |>
	map(Any, process -> generate_next(process))

# ╔═╡ 9216c513-b85e-487f-ae82-ed16e5463516
r = DataGenerationProcess(π / 30, npoints, [ 5.0 0.0; 0.0 5.0 ], [ 1.0 0.0; 0.0 1.0 ])

# ╔═╡ fae6b763-658c-4e89-a0c3-477612e6312f
function generate_next!(pendulum::DataGenerationProcess)
	x_k_min = last(pendulum.states)
	
	tmp = pendulum.state_transition_matrix * x_k_min
	x_k = rand(MvNormal(tmp, pendulum.state_transition_noise))
	y_k = rand(MvNormal(x_k, pendulum.observation_noise))
	
	push!(pendulum.states, x_k)
	push!(pendulum.observations, y_k)
	
	return pendulum.states, pendulum.observations
end

# ╔═╡ 5a4a85d1-bfa9-4451-84e7-835d2e9c610e
function generate_static(pendulum::DataGenerationProcess, npoints::Int)
	initial_state       = Point2f0(5.0, 0.0)
	initial_observation = rand(MvNormal(initial_state, pendulum.observation_noise))
	
	x_k = Vector{Point2f0}(undef, npoints)
	y_k = Vector{Point2f0}(undef, npoints)
	
	x_k[1] = initial_state
	y_k[1] = initial_observation
	
	for i in 2:npoints
		tmp = pendulum.state_transition_matrix * x_k[i - 1]
		x_k[i] = rand(MvNormal(tmp, pendulum.state_transition_noise))
		y_k[i] = rand(MvNormal(x_k[i], pendulum.observation_noise))
	end
	
	return x_k, y_k
end

# ╔═╡ 82b4d2d0-c6a9-4e94-9705-4e24bd83e62e
generate_static(r, 10)

# ╔═╡ a3fdf5ea-a1db-45f8-8e61-bdbdba083fd3
begin 
	local npoints = 400
	local P = [ 5.0 0.0; 0.0 5.0 ]
	local Q = [ 100.0 0.0; 0.0 100.0 ]
	local process = DataGenerationProcess(π / 30, npoints, P, Q)
	local A = process.state_transition_matrix
	local x, y  = generate_static(process, npoints)
	
	marginals = inference_full_graph(y, A, P, Q)
	
	local range = 100:1:200
	local dim   = (d) -> (a) -> map(e -> e[d...], a[range])
	
	local p1 = Plots.plot()
	local p2 = Plots.plot()
	
	p1 = Plots.scatter!(p1, y |> dim(1), ms = 2, label = "Observations")
	p1 = Plots.plot!(p1, x |> dim(1), label = "States")
	p1 = Plots.plot!(p1, 
		mean.(marginals) |> dim(1), ribbon = std.(marginals) |> dim((1, 1)),
		label = "Estimates"
	)
	
	p2 = Plots.scatter!(p2, y |> dim(2), ms = 2, label = "Observations")
	p2 = Plots.plot!(p2, x |> dim(2), label = "States")
	p2 = Plots.plot!(p2, 
		mean.(marginals) |> dim(2), ribbon = std.(marginals) |> dim((2, 2)),
		label = "Estimates"
	)
	
	Plots.plot(p1, p2, layout = Plots.@layout([ a; b ]), size = (800, 600))
end

# ╔═╡ 1996542b-c7aa-4bfe-ab60-9359407c8ad5
md"""
## Macro Utilities
"""

# ╔═╡ e4cfe8a2-fdd6-4c2c-acc9-8da715afbc8f
md"""
`@guard_subscription` macro unsubscribes every time from the same subscription reference before new subscription happens. Needed because Pluto executes cell automatically in a reactive manner so after some changes.
"""

# ╔═╡ b226adcc-e696-4336-8a6c-314295c59be3
macro guard_subscription(if_expression)
	@capture(if_expression, if condition_ block_ end) || error()
	
	@capture(block, subscription_ = subscribe!(stream_, actor_)) || error()
	
	output = quote 
		unsubscribe!($(subscription))
		yield()
		sleep(0.001)
		if $(condition)
			$(subscription) = subscribe!($(stream), $(actor))
		end
		nothing
	end
	
	return esc(output)
end

# ╔═╡ b7ae53d3-526f-44fa-91df-88d0a7b59d9b
@guard_subscription if is_subscribed
	subscription[] = subscribe!(stream, (_) -> begin
		try
			states, observations = generate_next!(data)

			real_states[]  = states
			noisy_states[] = observations
		catch e
			println(e)
		end
	end)	
end

# ╔═╡ Cell order:
# ╠═773890b4-f844-4a2b-946b-b56f7f6ac375
# ╠═3f028a44-cdd0-11eb-2fb0-6b696babeb9b
# ╠═bdf461c7-918d-4f78-968d-aa3dfd11d3b0
# ╟─08ec8032-40ba-4bef-9dce-53389a531cab
# ╠═a91b698e-24fc-47d0-9742-b6a151eacdd3
# ╠═12f072b6-1e9d-42d0-9784-15ab2ed32e1c
# ╟─c189ddad-c8f6-45c4-9cec-8d72a09f98cf
# ╠═aa3f4b38-ff13-492b-9eb3-bda2c80f5c5d
# ╠═320d3d53-bf88-4be0-a189-a3f1e8c4b24a
# ╟─5be725e6-732d-42e3-8967-3e8251946d72
# ╠═8de8cebd-b57e-438c-9cae-caf245c6a74b
# ╠═61952b04-cc7e-46f2-b273-854e425866ca
# ╠═de00e97d-e8a8-46a1-a144-04925a078e5a
# ╠═b7ae53d3-526f-44fa-91df-88d0a7b59d9b
# ╠═9216c513-b85e-487f-ae82-ed16e5463516
# ╠═82b4d2d0-c6a9-4e94-9705-4e24bd83e62e
# ╟─a814a9fe-6c76-4e75-9b0d-7141e18d1f9d
# ╠═a327b4f6-b8b9-4536-b45d-c13b767a3606
# ╠═65894eb0-99a6-4d29-a23d-bbe1dab4a2e8
# ╠═a3fdf5ea-a1db-45f8-8e61-bdbdba083fd3
# ╠═21b44508-ec16-4f15-9ba1-97fce4667873
# ╟─39b43b20-915b-44a6-947e-170234eae68a
# ╟─e738bbf7-8a66-411c-82f9-9716f749f30b
# ╠═649214d2-b661-49f5-ac36-50da9e358675
# ╠═fae6b763-658c-4e89-a0c3-477612e6312f
# ╠═5a4a85d1-bfa9-4451-84e7-835d2e9c610e
# ╟─1996542b-c7aa-4bfe-ab60-9359407c8ad5
# ╟─e4cfe8a2-fdd6-4c2c-acc9-8da715afbc8f
# ╠═b226adcc-e696-4336-8a6c-314295c59be3

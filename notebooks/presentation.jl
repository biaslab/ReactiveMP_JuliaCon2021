### A Pluto.jl notebook ###
# v0.19.27

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    quote
        local iv = try Base.loaded_modules[Base.PkgId(Base.UUID("6e696c72-6542-2067-7265-42206c756150"), "AbstractPlutoDingetjes")].Bonds.initial_value catch; b -> missing; end
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : iv(el)
        el
    end
end

# ╔═╡ ab2e4f47-a0cf-4c6b-9a7a-5f10b91a3079
import Pkg

# ╔═╡ 45d6751c-87e9-407a-b7d8-33a30b716c12
Pkg.activate("..")

# ╔═╡ 773890b4-f844-4a2b-946b-b56f7f6ac375
begin
	using JSServe
	Page()
end

# ╔═╡ 3f028a44-cdd0-11eb-2fb0-6b696babeb9b
begin
	using WGLMakie
	import WGLMakie: Point2f0
	WGLMakie.set_theme!(resolution = (1350, 800))
end

# ╔═╡ bdf461c7-918d-4f78-968d-aa3dfd11d3b0
begin 
	import DataStructures: CircularBuffer
	
	using MacroTools, PlutoUI, Distributions
	using ColorSchemes
	using Rocket, GraphPPL, ReactiveMP
	using Random
	using Colors
	using Images
	
	import Plots
end

# ╔═╡ fa08d3fc-ce86-49dd-9002-f8f698808295
md"""
## Data 
"""

# ╔═╡ 2dc6fb73-cd92-4b2a-a6ad-542092526ab0
# Testing parameters for static dataset
begin 
	# Seed for reproducability
	test_data_seed    = 123
	
	# Number of observations in static dataset
	test_data_npoints = 200
	
	# Angle change rate
	test_data_angle = π / 20
	
	# State transition noise covariance matrix
	test_data_state_transition_noise = [ 1.0 0.0; 0.0 1.0 ]
	
	# Observations noise covariance matrix
	test_data_observations_noise     = [ 200.0 0.0; 0.0 200.0 ]
end;

# ╔═╡ df9e5c18-8e1a-4f60-95db-6e84e8536ac7
md"""
Imagine that we have a dataset containing the trajectory of a moving object. We do not have direct access to the true location of the moving object (the hidden state), but only to noisy measurements of its location (the observations).
"""

# ╔═╡ 507c3c46-cf6c-4a19-a3e4-e852eae6321a
md"""
#### Linear Multivariate Gaussian State-space Model

We wish to estimate the true location of the moving object by only observing noisy measurements.
For this problem we may try to use a Linear Multivariate Gaussian State Space model (LMGSSM).
In this model we can perform both Kalman filtering and smoothing to estimate the hidden states of our system.

The model can be described with the following equations:

```math
\begin{equation}
  \begin{aligned}
    p(\mathbf{x}_k|\mathbf{x}_{k - 1}) & = \, \mathcal{N}(\mathbf{x}_k|\mathbf{A}\mathbf{x}_{k - 1}, \mathcal{P}) \\
    p(\mathbf{y}_k|\mathbf{x}_k) & = \, \mathcal{N}(\mathbf{y}_k|\mathbf{x}_{k}, \mathcal{Q}) \\
  \end{aligned}
\end{equation}
```

In this model, $\mathbf{x}_k$ denotes the current state of the system (the true location at time step $k$), $\mathbf{x}_{k - 1}$ denotes the previous state at time $k-1$, $\mathbf{A}$ is a constant system input and $\mathbf{y}_k$ is a noisy observation of $\mathbf{x}_k$. Furthermore we assume that the state transitions and the observations are corrupted by i.i.d. Gaussian noise with covariance matrices $\mathcal{P}$ and $\mathcal{Q}$, respectively.

The state space model can be visually represented by the following factor graph, where the shown graph represents a section is the full model that is repeated for each time step:

$(load("./../figs/ffg_png.png"))

Usually this type of model is used for linear differential equations where the measured quantities are linear functions of the state variables. For example, this can be the dynamics of a car or a noisy pendulum model [Bayesian Filtering and Smoothing, Särkkä, Simo, p.~44].
"""

# ╔═╡ 1eedb961-877d-4481-9441-b7b04c0cd361
md"""
## Filtering model
"""

# ╔═╡ f7169a26-25ec-453d-a3c8-068048e7d83d
md"""

```math
\begin{equation}
  \begin{aligned}
    p(\mathbf{x}_{\mathrm{prev}}) & = \, \mathcal{N}(\mathbf{x}_{\mathrm{prev}}|\mathbf{μ}_{\mathrm{prev}}, \mathbf{\Sigma}_{\mathrm{prev}}) \\
    p(\mathbf{x}|\mathbf{x}_{\mathrm{prev}}) & = \, \mathcal{N}(\mathbf{x}|\mathbf{A}\mathbf{x}_{\mathrm{prev}}, \mathcal{P}) \\
    p(\mathbf{y}|\mathbf{x}) & = \, \mathcal{N}(\mathbf{y}|\mathbf{x}, \mathcal{Q}) \\
  \end{aligned}
\end{equation}
```

"""

# ╔═╡ b8caf0eb-aad9-4506-823c-b7ecc23caf7c
# The `@model` macro accepts a regular Julia function.
@model function one_time_step_graph(A, P, Q)
	
	# `datavar` creates a data input in the model.
	# Later on during the inference procedure we are allowed 
	# to pass data to our model by referencing the created
	# `datavar` variables.
	x_prev_mean = datavar(Vector{Float64})
	x_prev_cov  = datavar(Matrix{Float64})
	
	# `x_prev` contains our prior knowledge about the previous state 
	# in our model at time step k - 1.
	# At k = 0, the variable `x_prev` acts as an initial prior on the
	# hidden state.
	x_prev ~ MvNormalMeanCovariance(x_prev_mean, x_prev_cov)
	
	# `x` is the current hidden state at time step k
	x ~ MvNormalMeanCovariance(A * x_prev, P)
	
	# `y` is an observation, which we will pass to the model during 
	# the inference procedure. Furthermore we specify that we can
	# model `y` from our hidden state `x`.
	y = datavar(Vector{Float64})
	
	y ~ MvNormalMeanCovariance(x, Q)
	
	# We return all variable references for later usage
	# during the inference procedure.
	return x_prev_mean, x_prev_cov, x, y
end

# ╔═╡ c4cc673c-4d44-44df-89b0-68744d473a15
# ReactiveMP.jl does not export any default inference procedures like 
# other probabilistic programming libraries do. Instead the user is free to 
# implement their own inference algorithms. In this example we create an 
# inference procedure which is compatible with both reactive infinite 
# real-time data streams and with static datasets.
function inference_filtering(data_stream, A, P, Q)
	
	# First we create our model:
	# The `@model` generated function returns a `model` reference 
	# and the same output as in the `return` statement as a second argument.
	model, (x_prev_mean, x_prev_cov, x, y) = one_time_step_graph(A, P, Q)
	
	# These are helper references for later usage in callbacks
	data_subscription  = Ref{Teardown}(voidTeardown)
	prior_subscription = Ref{Teardown}(voidTeardown)
	
	# The `getmarginal()` function returns an observable of the posterior marginals.
	x_posterior_marginals_observable = getmarginal(x)
	
	# This function will be called when someone subscribes on a stream 
	# of posterior marginals.
	start_callback = () -> begin
		
		# At the very beginning we use the `update!()` function 
		# to pass our initial prior to the model.
		ReactiveMP.update!(x_prev_mean, [ 0.0, 0.0 ])
		ReactiveMP.update!(x_prev_cov, [ 100.0 0.0; 0.0 100.0 ])
		
		# Here we create an infinite reaction loop.
		# As soon as a new posterior marginal is available we set 
		# it as our new prior for previous time step, such that we
		# can continue processing the next time step.
		prior_subscription[] = subscribe!(getmarginal(x), (mx) -> begin
				
			μ, Σ = mean_cov(mx)
				
			ReactiveMP.update!(x_prev_mean, μ)
			ReactiveMP.update!(x_prev_cov, Σ)	
				
		end)
		
		# We subscribe to a data stream and redirect all data 
		# to the observations `datavar` input variable.
		data_subscription[] = subscribe!(data_stream, (d) -> begin 
			ReactiveMP.update!(y, convert(Vector{Float64}, d))
		end)
	end
	
	# This function will be called when someone unsubscribes from a stream 
	# of posterior marginals.
	stop_callback = () -> begin
		unsubscribe!(data_subscription[])
		unsubscribe!(prior_subscription[])
	end
	
	return x_posterior_marginals_observable |> 
		tap_on_subscribe(start_callback, TapAfterSubscription()) |> 
		tap_on_unsubscribe(stop_callback)
end

# ╔═╡ 36344f8d-1fbb-4b54-b17e-46edcebb6c7d
# Parameters for the static inference example for the LMGSSM 
# and Kalman filter using message passing.
begin 
	# Seed for reproducability
	filtering_seed    = 42
	
	# Number of observations in static dataset
	filtering_npoints = 500
	
	# Angle change rate
	filtering_angle = π / 100
	
	# State transition noise covariance matrix
	filtering_state_transition_noise = [ 1.0 0.0; 0.0 1.0 ]
	
	# Observations noise covariance matrix
	fittering_observations_noise     = [ 200.0 0.0; 0.0 200.0 ]
end;

# ╔═╡ bc84b4d8-8916-4081-9d86-9322d8f4d200
md"""
### Real-time Setup 
"""

# ╔═╡ bb1e524b-7cff-412a-8c05-d60a350f28b8
md"""
Here we create reactive nodes for our interactive plots. It is worth to note that we use the `Makie.jl` plotting engine, which uses `Observables.jl` for reactivity. `ReactiveMP.jl` on the other hand uses the `Rocket.jl` reactive framework for achieving best possible performance and whilst retaining rich functionality.
"""

# ╔═╡ c135e271-10d3-4deb-abf2-6d5f65a6fc78
md"""
Here we create a data generation check box and subscription reference.
"""

# ╔═╡ 5752f563-22ef-4e9e-a3d0-6630cc911248
md"""
To make the plots look nicer we pass zeroed data upon unsubscription to the observables that we use for plotting.
"""

# ╔═╡ 6860f6cd-7507-455c-85e1-a89afea8890e
md"""
### Data generation process
"""

# ╔═╡ f5fe83ad-3be9-4d2c-85ad-da7633d7367e
md"""
As our synthetic data stream we create a timer observable, which emits a number after a prespecified duration. This stream is infinite and never completes. We re-map it to our data generation process and call the `generate_next!()` function on each timer emission. We also use the `share_replay` operator to share this stream of data between multiple listeners.
"""

# ╔═╡ 1b31dfed-7bbf-4c12-a70c-a3dbb7854d9d
md"""
### Real-time inference setup
"""

# ╔═╡ f670169a-72d9-4e55-b8ce-a6db4baec53b
vecsqrt(v) = sqrt.(v)

# ╔═╡ d5c1b0e8-fd51-4312-a06f-0287aad184ba
begin
	# Note: interactive visualisation is a bit fragile with changing parameters while             visualising at the same time, it is better to deselect both check boxes               before 	changing anything	

	# How many points are used for plotting
	# The inference procedure does not depend on this parameter since 
	# we perform Kalman filtering.
	npoints = 150
	
	# Rate at which data points are generated, should be somewhere 
	# between 10 and 100
	speed = 50
	
	# Angle used in rotation matrix, specifies the angle change rate
	angle = π / 70
	
	A = [ cos(angle) -sin(angle); sin(angle) cos(angle) ]
	
	# State transition noise covariance matrix
	P = [ 0.5 0.0; 0.0 0.5 ]
	
	# Observations noise covariance matrix
	Q = [ 1000.0 0.0; 0.0 1000.0 ]
end;

# ╔═╡ 65125537-9ee9-4b2d-92be-45ee1a914b24
begin 	
	real_states  = Observable(map(_ -> Point2f0(1000.0, 1000.0), 1:npoints))
	noisy_states = Observable(map(_ -> Point2f0(1000.0, 1000.0), 1:npoints))
end;

# ╔═╡ fa7fbcbc-e229-4553-88a8-17c53779725d
begin 
	inferred_states  = Observable(map(_ -> Point2f0(0, 0), 1:npoints))
	inferred_band_up = Observable(map(_ -> Point2f0(0, 0), 1:npoints))
	inferred_band_lp = Observable(map(_ -> Point2f0(0, 0), 1:npoints))
end;

# ╔═╡ 736867b0-d0b6-41b9-b4cc-b0de734714fa
begin
	inferred_buffer = CircularBuffer{Marginal}(npoints)
	fill!(inferred_buffer, ReactiveMP.Marginal(PointMass([ 0.0, 0.0 ]), false, false))
end;

# ╔═╡ a83a421f-0ffb-452d-a112-178b4c7b4ebd
begin 
	connect_noise_observations_check_box = 
		@bind is_observations_connected CheckBox(default = false);
end;

# ╔═╡ 261ca703-871a-4252-a056-157e7c48ae1c
md"""
Note: the first selection may lag a little bit due to plotting compilation, also it is better to wait until all cells have been initialised.
"""

# ╔═╡ 0033f1f2-c84b-47d5-8a58-1f57822c9a25
md"""
Select the 'Show data' checkbox to subscribe to a real-time data generation process and to visualise it on the interactive pane below.

Select the 'Run inference' checkbox to subscribe to the real-time inference procedure. Inference uses the same real-time dataset for inference and state estimation.

Select the 'Connect observations' to connect consecutive observations by a line.
"""

# ╔═╡ 54043ff7-05f2-48ef-89f4-050379eba3f9
begin
	local f = WGLMakie.Figure()
	
	local colors_data = collect(cgrad(:Blues_9, npoints))
	local colors_inferred = collect(cgrad(:Reds_9, npoints))
	local sizes  = abs2.(range(1, 3, length=npoints))	
	local widths = range(0, 2, length = npoints)
	
	scatter(f[1, 1], noisy_states, markersize = sizes, color = :green)
	lines!(f[1, 1], 
		real_states, linewidth = sizes, color = colors_data, 
	)
	lines!(f[1, 1], 
		inferred_states, linewidth = sizes, color = colors_inferred,
	)
	limits!(-50, 50, -50, 50)
	
	axis11 = WGLMakie.Axis(f[1, 1])
	axis11.ylabel = "y-position" 
	axis11.xlabel = "x-position" 
	axis11.xlabelsize = 26
	axis11.ylabelsize = 26
	
	inferred_states_dim1 = map(i -> map(r -> r[1], i), inferred_states)
	inferred_states_dim2 = map(i -> map(r -> r[2], i), inferred_states)
	
	inferred_band_dim1_up = map(i -> map(r -> r[1], i), inferred_band_up)
	inferred_band_dim1_lp = map(i -> map(r -> r[1], i), inferred_band_lp)
	
	inferred_band_dim2_up = map(i -> map(r -> r[2], i), inferred_band_up)
	inferred_band_dim2_lp = map(i -> map(r -> r[2], i), inferred_band_lp)

	local states_dim1 = map(i -> map(r -> r[1], i), real_states)
	local states_dim2 = map(i -> map(r -> r[2], i), real_states)
	
	local noisy_dim1 = map(i -> map(r -> r[1], i), noisy_states)
	local noisy_dim2 = map(i -> map(r -> r[2], i), noisy_states)
	
	lines(f[1, 2][1, 1], inferred_states_dim1, color = colors_inferred)
	band!(f[1, 2][1, 1], 
		1:npoints, inferred_band_dim1_up, inferred_band_dim1_lp,
		color = RGBA(0.9,0.0,0.0,0.2)
	)
	lines!(f[1, 2][1, 1], states_dim1, color = colors_data)
	scatter!(f[1, 2][1, 1], noisy_dim1, markersize = sizes, color = :green)
	if is_observations_connected
		lines!(f[1, 2][1, 1], noisy_dim1, linewidth = widths, color = :green)
	end
	limits!(0, npoints, -50, 50)
	
	axis1211 = WGLMakie.Axis(f[1, 2][1, 1])
	axis1211.ylabel = "x-position" 
	axis1211.xlabel = "Time step" 
	axis1211.xlabelsize = 26
	axis1211.ylabelsize = 26
	
	
	lines(f[1, 2][2, 1], inferred_states_dim2, color = colors_inferred)
	band!(f[1, 2][2, 1], 
		1:npoints, inferred_band_dim2_up, inferred_band_dim2_lp,
		color = RGBA(0.9,0.0,0.0,0.2)
	)
	lines!(f[1, 2][2, 1], states_dim2, color = colors_data)
	scatter!(f[1, 2][2, 1], noisy_dim2, markersize = sizes, color = :green)
	if is_observations_connected
		lines!(f[1, 2][2, 1], noisy_dim2, linewidth = widths, color = :green)
	end
	limits!(0, npoints, -50, 50)
	
	axis1211 = WGLMakie.Axis(f[1, 2][2, 1])
	axis1211.ylabel = "y-position" 
	axis1211.xlabel = "Time step" 
	axis1211.xlabelsize = 26
	axis1211.ylabelsize = 26
	
	on(events(f).mousebutton) do event
		return true
	end
	
	on(events(f).mouseposition) do event
		return true
	end
	
	on(events(f).scroll) do event
		return true
	end
	
	inferred_fig = current_figure()
end

# ╔═╡ 6cf1a1b0-3d6c-4e61-90e4-40540622c0ea


# ╔═╡ 1bcdc68f-0e19-4534-b79b-923e58702c18


# ╔═╡ 872b9127-0632-44f3-9f11-f7a4ef4956a2


# ╔═╡ a814a9fe-6c76-4e75-9b0d-7141e18d1f9d
md"""
## Smoothing model
"""

# ╔═╡ ea8f35f0-08da-4dee-bab5-69dd81e8ab3b
md"""
ReactiveMP.jl is flexible enough to run inference on a full graph with a static datasets. In this example we show a Kalman smoothing algorithm using sum-product message passing.
"""

# ╔═╡ 224f8617-12e9-4c08-ae9e-c4dd283b018a
md"""

```math
\begin{equation}
  \begin{aligned}
    p(\mathbf{x}_k|\mathbf{x}_{k - 1}) & = \, \mathcal{N}(\mathbf{x}_k|\mathbf{A}\mathbf{x}_{k - 1}, \mathcal{P}) \\
    p(\mathbf{y}_k|\mathbf{x}_k) & = \, \mathcal{N}(\mathbf{y}_k|\mathbf{x}_{k}, \mathcal{Q}) \\
  \end{aligned}
\end{equation}
```

"""

# ╔═╡ a327b4f6-b8b9-4536-b45d-c13b767a3606
# Here we create a full graph 
@model function full_graph(npoints, A, P, Q)
	
	x = randomvar(npoints) # A sequence of random variables with length `npoints`
	y = datavar(Vector{Float64}, npoints) # A sequence of data inputs
	
	# `constvar` creates a constant reference for constants in a model
	# (unnecessary for the actual inference results, but necessary for better performance)
	cA = constvar(A)
	cP = constvar(P)
	cQ = constvar(Q)
	
	# Our model specification closely resembles the data generating model.
	x[1] ~ MvNormalMeanCovariance([ 0.0, 0.0 ], [ 100.0 0.0; 0.0 100.0 ])
	y[1] ~ MvNormalMeanCovariance(x[1], cQ)
	
	for i in 2:npoints
		x[i] ~ MvNormalMeanCovariance(cA * x[i - 1], cP)
		y[i] ~ MvNormalMeanCovariance(x[i], cQ)
	end
	
	return x, y
end

# ╔═╡ 65894eb0-99a6-4d29-a23d-bbe1dab4a2e8
function inference_smoothing(data, A, P, Q)

	# We assume to use a static dataset here:
	data    = convert(AbstractVector{Vector{Float64}}, data)
	npoints = length(data)
	
	# We use the `limit_stack_depth` option for huge models with thousands of nodes.
	model, (x, y) = full_graph(npoints, A, P, Q, options = (
		limit_stack_depth = 500,
	))
	
	xbuffer    = buffer(Marginal, npoints)
	xmarginals = getmarginals(x)
	
	# As soon as new marginals are available 
	# we want to save them in the `xbuffer`.
	subscription = subscribe!(xmarginals, xbuffer)
	
	# Because we assume a static dataset, our workflow is easier then in a 
	# reactive real-time filtering setup. We may just pass all the data and wait 
	# synchronously for all posterior marginals to be updated.
	ReactiveMP.update!(y, data)
	
	# It is a good practise to unsubscribe, 
	# though in this example it is unecessary 
	unsubscribe!(subscription)
	
	# Inference results
	return getvalues(xbuffer)
end

# ╔═╡ 621bebd0-492e-49fc-a58a-ecaee47286f6
# Parameters for the LMGSSM, where we perform Kalman smoothing through
# message passing on a static dataset.
begin 
	# Seed for reproducability
	smoothing_seed    = Int(round(1000 * rand()))
	
	# Number of observations in static dataset
	smoothing_npoints = 1_000
	
	# Angle change rate
	smoothing_angle = π / 100
	
	# State transition noise covariance matrix
	smoothing_state_transition_noise = [ 1.0 0.0; 0.0 1.0 ]

	# Observations noise covariance matrix
	smoothing_observations_noise     = [ 200.0 0.0; 0.0 200.0 ]
end;

# ╔═╡ 0e2ca93e-f34b-4255-9b3b-f96b22bbad6a
md"""
For a lot of points plotting can become really slow, here are some sliders to show only part of the inferred states:
"""

# ╔═╡ ba7e8055-a4e6-4f6f-afe0-51192091f2f6
begin 
	smoothing_plot_window_size_slider = @bind smoothing_plot_window_size PlutoUI.Slider(1:smoothing_npoints, default = 1000, show_value = true)
	
	smoothing_window_start_slider = @bind smoothing_plot_window_start PlutoUI.Slider(1:smoothing_npoints, default = 1, show_value = true)
	
	md"""
	Plot window size: $(smoothing_plot_window_size_slider) Window start: $(smoothing_window_start_slider)
	"""
end

# ╔═╡ 3036773b-0ccf-4bc2-9275-feca31252c71
smoothing_plot_range = smoothing_plot_window_start:min((smoothing_plot_window_start + smoothing_plot_window_size - 1), smoothing_npoints)

# ╔═╡ 39b43b20-915b-44a6-947e-170234eae68a
md"""
## Utilities
"""

# ╔═╡ e59d6d36-d576-4b2d-a66b-a03e780afc9c
make_subscription_reference() = Ref{Rocket.Teardown}(voidTeardown)

# ╔═╡ 5be725e6-732d-42e3-8967-3e8251946d72
begin 
	data_generation_check_box = 
		@bind is_data_generation_subscribed CheckBox(default = false);
	
	data_generation_subscription = make_subscription_reference()
end;

# ╔═╡ 00844a01-0ccd-48d9-a961-ea1d8383da70
if !is_data_generation_subscribed
	@async begin
		local zeroed = map(_ -> Point2f0(1000.0, 1000.0), 1:npoints)
		sleep(0.1)
		real_states[]  = zeroed
		noisy_states[] = zeroed
	end
end;

# ╔═╡ 61f29db3-6deb-47ca-8473-f9ead5d9d2f0
data_generation_callback = (data) -> begin
	try
		states, observations = data[1], data[2]
		
		# Every time we receive new data from our stream we
		# just pass it to the observable nodes used for plotting.

		if is_data_generation_subscribed
			real_states[]  = states
			noisy_states[] = observations
		end
		
	catch e
		println(e)
	end
end

# ╔═╡ ab1148c0-16bf-4f14-874a-b06878c8e538
begin 
	inference_check_box    = @bind is_inference_subscribed CheckBox(default = false);
	inference_subscription = make_subscription_reference()
end;

# ╔═╡ 2e3b739c-de1a-46ed-93cd-cdb3dda8c7b2
if !is_inference_subscribed
	@async begin
		local zeroed = map(_ -> Point2f0(1000.0, 1000.0), 1:npoints)
		sleep(0.1)
		inferred_states[] = zeroed
		inferred_band_up[] = zeroed
		inferred_band_lp[] = zeroed
		empty!(inferred_buffer)
		fill!(inferred_buffer, ReactiveMP.Marginal(PointMass([ 0.0, 0.0 ]), false, false))
	end
end;

# ╔═╡ aa7b11f4-6a37-4c70-9fdf-174d9e1e2c3a
inference_callback = (marginal) -> begin
	try 
		
		# Every time a new posterior marginal is available
		# we put it in our circular buffer and pass new values 
		# to the Makie.jl visualisation backend to plot new values.
		
		push!(inferred_buffer, marginal)
		
		local ms = mean.(inferred_buffer)
		local σ  = vecsqrt.(var.(inferred_buffer))
		local up = ms .+ σ
		local lp = ms .- σ

		if is_inference_subscribed
			inferred_states[]  = ms
			inferred_band_up[] = up
			inferred_band_lp[] = lp
		end

	catch e
		println(e)
	end
end

# ╔═╡ 8b2113ff-6ade-40b5-87e9-3a62614e2a72
md"
Show data? $(data_generation_check_box) 
Run inference? $(inference_check_box)
"

# ╔═╡ e738bbf7-8a66-411c-82f9-9716f749f30b
md"""
`DataGenerationProcess` encapsulates a simple data generation process with a rotation matrix as state transition matrix. The states are modelled by a Gaussian distributions and have their own state transition noise. Observations are noisy realizations of the hidden states, where the noise is sampled from a Gaussian distribution.
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

# ╔═╡ fae6b763-658c-4e89-a0c3-477612e6312f
# The `generate_next!()`` function generates updated states and observations for the current process.
function generate_next!(process::DataGenerationProcess)
	x_k_min = last(process.states)
	
	tmp = process.state_transition_matrix * x_k_min
	x_k = rand(MvNormal(tmp, process.state_transition_noise))
	y_k = rand(MvNormal(x_k, process.observation_noise))
	
	push!(process.states, x_k)
	push!(process.observations, y_k)
	
	return process.states, process.observations
end

# ╔═╡ 5d74139d-f61c-4a63-a7bd-35c20f5c04af
# A utility method for recursion issue with the `DataGenerationProcess` constructor and 
# `generate_next!()` function
function make(::Type{ <: DataGenerationProcess }, angle::Float64, npoints, snoise, onoise)
	object = DataGenerationProcess(angle, npoints, snoise, onoise)
	for i in 1:npoints-1
		generate_next!(object)
	end
	return object
end

# ╔═╡ 399a5b08-ce77-4864-a957-d0be1d6468c1
realtime_data = make(DataGenerationProcess, angle, npoints, P, Q);

# ╔═╡ de00e97d-e8a8-46a1-a144-04925a078e5a
stream = timer(100, Int(10 * round(100 / speed))) |> 
	map_to(realtime_data) |> 
	map(Tuple, generate_next!) |>
	share_replay(1)

# ╔═╡ 0f2b1375-fe83-47b9-b23d-a6051b6175ce
filtering_data_stream = stream |> map(Any, d -> d[2][end])

# ╔═╡ f3c67020-ee41-4e2d-92db-3c1c62425dca
inferred_stream = inference_filtering(filtering_data_stream, A, P, Q)

# ╔═╡ 5a4a85d1-bfa9-4451-84e7-835d2e9c610e
function generate_static(process::DataGenerationProcess, npoints::Int; seed = nothing)
	
	rng = seed === nothing ? Random.GLOBAL_RNG : MersenneTwister(seed)
	
	init_state       = Point2f0(1.0, 0.0)
	init_observation = rand(rng, MvNormal(init_state, process.observation_noise))
	
	x_k = Vector{Point2f0}(undef, npoints)
	y_k = Vector{Point2f0}(undef, npoints)
	
	x_k[1] = init_state
	y_k[1] = init_observation
	
	for i in 2:npoints
		tmp = process.state_transition_matrix * x_k[i - 1]
		x_k[i] = rand(rng, MvNormal(tmp, process.state_transition_noise))
		y_k[i] = rand(rng, MvNormal(x_k[i], process.observation_noise))
	end
	
	return x_k, y_k
end

# ╔═╡ 48ca6f5a-33c2-4118-a72d-71e59c7a2b22
begin
	test_data_generator = make(DataGenerationProcess, 
		test_data_angle, test_data_npoints, 
		test_data_state_transition_noise, test_data_observations_noise
	)
	
	test_dataset = generate_static(
		test_data_generator, test_data_npoints, seed = test_data_seed
	)
end;

# ╔═╡ f4baf021-1eca-4677-916c-62036c2984c7
begin
	local npoints = test_data_npoints 
	local test_data_generator = make(DataGenerationProcess, 
		test_data_angle, test_data_npoints, 
		test_data_state_transition_noise, test_data_observations_noise
	)
	
	local test_dataset = generate_static(
		test_data_generator, test_data_npoints, seed = test_data_seed
	)
	
	local sizes  = abs2.(range(1, 1.7, length=npoints))
	local alphas = abs2.(range(0.5, 1.0, length=npoints))
	local colors_data = Plots.palette([:blue, :green], npoints)
	local x, y = test_dataset
	
	local dim   = (d) -> (a) -> map(e -> e[d...], a)
	
	local p = Plots.plot(xlimit = (-25, 25), ylimit = (-25, 25))
	local p1 = Plots.plot()
	local p2 = Plots.plot()
	
	p = Plots.scatter!(p, y, 
		color = colors_data, ms = sizes, alpha = alphas, label = false,
	)
	p = Plots.plot!(p, x, color = :teal, label = "Trajectory")
	
	p1 = Plots.scatter!(p1, y |> dim(1),
		alpha = alphas,
		ms = sizes, c = colors_data,
		label = false,
		xaxis = "Time step",
		yaxis = "x"
	)
	
	p2 = Plots.scatter!(p2, y |> dim(2), 
		alpha = alphas,
		ms = sizes, c = colors_data,
		label = false,
		xaxis = "Time step",
		yaxis = "y"
	)

	Plots.plot(p, p1, p2, layout = Plots.@layout([ a [ b; c ] ]), size = (800, 400))
end

# ╔═╡ 20442224-6b3e-4def-8921-05178156fa4f
begin
	local seed = filtering_seed
	local npoints = filtering_npoints
	local P = filtering_state_transition_noise
	local Q = fittering_observations_noise
	local process = DataGenerationProcess(filtering_angle, npoints, P, Q)
	local A = process.state_transition_matrix
	local x, y  = generate_static(process, npoints; seed = seed)
	local data_stream = from(y)
	
	local xmarginal_stream = inference_filtering(
		data_stream, A, P, Q
	)
	
	local xkeep = keep(Marginal)
	local subscription = subscribe!(xmarginal_stream, xkeep)
	
	unsubscribe!(subscription)
	
	local marginals = getvalues(xkeep)
	
	local range = 1:npoints
	local dim   = (d) -> (a) -> map(e -> e[d...], a)
	
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

# ╔═╡ b78366c9-4fcd-4b3b-a9c3-4d7f6ee9d8c0
begin 
	local seed = smoothing_seed
	local npoints = smoothing_npoints
	local P = smoothing_state_transition_noise
	local Q = smoothing_observations_noise
	local process = DataGenerationProcess(smoothing_angle, npoints, P, Q)
	local A = process.state_transition_matrix
	
	smoothing_x, smoothing_y  = generate_static(process, npoints; seed = seed)
	smoothing_marginals = inference_smoothing(smoothing_y, A, P, Q)
end;

# ╔═╡ a3fdf5ea-a1db-45f8-8e61-bdbdba083fd3
begin 
	local range = smoothing_plot_range
	local dim   = (d) -> (a) -> map(e -> e[d...], a[range])
	
	local p1 = Plots.plot()
	local p2 = Plots.plot()
	
	p1 = Plots.scatter!(p1, smoothing_y |> dim(1), ms = 2, label = "Observations")
	p1 = Plots.plot!(p1, smoothing_x |> dim(1), label = "States")
	p1 = Plots.plot!(p1, 
		mean.(smoothing_marginals) |> dim(1), 
		ribbon = std.(smoothing_marginals) |> dim((1, 1)),
		label = "Estimates"
	)
	
	p2 = Plots.scatter!(p2, smoothing_y |> dim(2), ms = 2, label = "Observations")
	p2 = Plots.plot!(p2, smoothing_x |> dim(2), label = "States")
	p2 = Plots.plot!(p2, 
		mean.(smoothing_marginals) |> dim(2), 
		ribbon = std.(smoothing_marginals) |> dim((2, 2)),
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
The `@guard_subscription` macro unsubscribes every time from the same subscription reference before a new subscription occurs. This is required because Pluto executes cells automatically in a reactive manner.
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
# The @guard_subscription macro is used to have a better control over the reactive subscription/unsubscription process in Pluto notebooks.
@guard_subscription if is_data_generation_subscribed
	data_generation_subscription[] = subscribe!(stream, data_generation_callback)	
end

# ╔═╡ d92dcc1b-4374-44ec-81b0-3fb74b4a9807
@guard_subscription if is_inference_subscribed
	inference_subscription[] = subscribe!(inferred_stream, inference_callback)
end

# ╔═╡ ce16ce1a-0933-4899-abba-52102c720f3a


# ╔═╡ Cell order:
# ╠═ab2e4f47-a0cf-4c6b-9a7a-5f10b91a3079
# ╠═45d6751c-87e9-407a-b7d8-33a30b716c12
# ╠═773890b4-f844-4a2b-946b-b56f7f6ac375
# ╠═3f028a44-cdd0-11eb-2fb0-6b696babeb9b
# ╠═bdf461c7-918d-4f78-968d-aa3dfd11d3b0
# ╟─fa08d3fc-ce86-49dd-9002-f8f698808295
# ╠═2dc6fb73-cd92-4b2a-a6ad-542092526ab0
# ╟─48ca6f5a-33c2-4118-a72d-71e59c7a2b22
# ╟─df9e5c18-8e1a-4f60-95db-6e84e8536ac7
# ╟─f4baf021-1eca-4677-916c-62036c2984c7
# ╟─507c3c46-cf6c-4a19-a3e4-e852eae6321a
# ╟─1eedb961-877d-4481-9441-b7b04c0cd361
# ╟─f7169a26-25ec-453d-a3c8-068048e7d83d
# ╠═b8caf0eb-aad9-4506-823c-b7ecc23caf7c
# ╟─c4cc673c-4d44-44df-89b0-68744d473a15
# ╟─36344f8d-1fbb-4b54-b17e-46edcebb6c7d
# ╟─20442224-6b3e-4def-8921-05178156fa4f
# ╟─bc84b4d8-8916-4081-9d86-9322d8f4d200
# ╟─bb1e524b-7cff-412a-8c05-d60a350f28b8
# ╠═65125537-9ee9-4b2d-92be-45ee1a914b24
# ╠═fa7fbcbc-e229-4553-88a8-17c53779725d
# ╟─c135e271-10d3-4deb-abf2-6d5f65a6fc78
# ╠═5be725e6-732d-42e3-8967-3e8251946d72
# ╟─5752f563-22ef-4e9e-a3d0-6630cc911248
# ╠═00844a01-0ccd-48d9-a961-ea1d8383da70
# ╟─6860f6cd-7507-455c-85e1-a89afea8890e
# ╠═399a5b08-ce77-4864-a957-d0be1d6468c1
# ╟─f5fe83ad-3be9-4d2c-85ad-da7633d7367e
# ╠═de00e97d-e8a8-46a1-a144-04925a078e5a
# ╠═61f29db3-6deb-47ca-8473-f9ead5d9d2f0
# ╠═b7ae53d3-526f-44fa-91df-88d0a7b59d9b
# ╟─1b31dfed-7bbf-4c12-a70c-a3dbb7854d9d
# ╠═736867b0-d0b6-41b9-b4cc-b0de734714fa
# ╠═ab1148c0-16bf-4f14-874a-b06878c8e538
# ╠═2e3b739c-de1a-46ed-93cd-cdb3dda8c7b2
# ╠═0f2b1375-fe83-47b9-b23d-a6051b6175ce
# ╠═f3c67020-ee41-4e2d-92db-3c1c62425dca
# ╟─f670169a-72d9-4e55-b8ce-a6db4baec53b
# ╟─aa7b11f4-6a37-4c70-9fdf-174d9e1e2c3a
# ╠═d92dcc1b-4374-44ec-81b0-3fb74b4a9807
# ╠═d5c1b0e8-fd51-4312-a06f-0287aad184ba
# ╟─a83a421f-0ffb-452d-a112-178b4c7b4ebd
# ╟─261ca703-871a-4252-a056-157e7c48ae1c
# ╟─0033f1f2-c84b-47d5-8a58-1f57822c9a25
# ╟─8b2113ff-6ade-40b5-87e9-3a62614e2a72
# ╠═54043ff7-05f2-48ef-89f4-050379eba3f9
# ╠═6cf1a1b0-3d6c-4e61-90e4-40540622c0ea
# ╠═1bcdc68f-0e19-4534-b79b-923e58702c18
# ╠═872b9127-0632-44f3-9f11-f7a4ef4956a2
# ╟─a814a9fe-6c76-4e75-9b0d-7141e18d1f9d
# ╟─ea8f35f0-08da-4dee-bab5-69dd81e8ab3b
# ╟─224f8617-12e9-4c08-ae9e-c4dd283b018a
# ╠═a327b4f6-b8b9-4536-b45d-c13b767a3606
# ╠═65894eb0-99a6-4d29-a23d-bbe1dab4a2e8
# ╠═621bebd0-492e-49fc-a58a-ecaee47286f6
# ╟─b78366c9-4fcd-4b3b-a9c3-4d7f6ee9d8c0
# ╟─0e2ca93e-f34b-4255-9b3b-f96b22bbad6a
# ╟─ba7e8055-a4e6-4f6f-afe0-51192091f2f6
# ╟─3036773b-0ccf-4bc2-9275-feca31252c71
# ╟─a3fdf5ea-a1db-45f8-8e61-bdbdba083fd3
# ╟─39b43b20-915b-44a6-947e-170234eae68a
# ╟─e59d6d36-d576-4b2d-a66b-a03e780afc9c
# ╟─e738bbf7-8a66-411c-82f9-9716f749f30b
# ╠═649214d2-b661-49f5-ac36-50da9e358675
# ╠═5d74139d-f61c-4a63-a7bd-35c20f5c04af
# ╠═fae6b763-658c-4e89-a0c3-477612e6312f
# ╠═5a4a85d1-bfa9-4451-84e7-835d2e9c610e
# ╟─1996542b-c7aa-4bfe-ab60-9359407c8ad5
# ╟─e4cfe8a2-fdd6-4c2c-acc9-8da715afbc8f
# ╠═b226adcc-e696-4336-8a6c-314295c59be3
# ╠═ce16ce1a-0933-4899-abba-52102c720f3a

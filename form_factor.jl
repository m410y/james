module FormFactor

using StaticArrays

file = open("form_factor.txt")
form_consts = Dict{String, Tuple{Int, SVector{9, Float64}}}()
for line in eachline(file)
    data = split(line)
    el_name = data[1]
    el_num = parse(Int, data[2])
    el_params = parse.(Float64, data[3:11])
    form_consts[el_name] = (el_num, el_params)
end
close(file)

function amp_eval(q::Float64, atom::Tuple{String, Float64})::Float64
    params = form_consts[atom[1]][2]
    amp = params[9]
    for (a, b) in zip(params[1:2:7], params[2:2:8])
        amp += a * exp(-b*(q/4/pi)^2)
    end
    return amp * atom[2]
end

function phase_eval(hkl::SVector{3, Int}, xyz::SVector{3, Float64})::Complex
    return exp(2*pi*im*hkl'xyz)
end

export ff_eval
function ff_eval(q::Float64, hkl::SVector{3, Int}, cell::Vector{Tuple{Tuple{String, Float64}, SVector{3, Float64}}})::Float64
    ff = 0.0
    for (atom, xyz) in cell
        ff += amp_eval(q, atom) * phase_eval(hkl, xyz)
    end
    return abs(ff)
end

end
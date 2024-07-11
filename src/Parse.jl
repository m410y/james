"""
Parsing sfrm, p4p files
"""
module Parse

include("ParseSFRM.jl")
import .ParseSFRM: read_sfrm, read_sfrm_header

export sfrm_full, sfrm_header, sfrm_image, p4p_header

function sfrm_full(filename::AbstractString)::Tuple{Dict{String, Vector{SubString}}, Matrix{Integer}}
    file = open(filename, "r")
    return read_sfrm(file)
end

function sfrm_header(filename::AbstractString)::Dict{String, Vector{SubString}}
    file = open(filename, "r")
    return read_sfrm_header(file)
end

function sfrm_image(filename::AbstractString)::Matrix{Integer}
    file = open(filename, "r")
    return read_sfrm(file)[2]
end

function p4p_header(filename::AbstractString)::Dict{String, Vector{SubString}}
    file = open(filename, "r")
    header = Dict{String, Vector{SubString}}()
    for line in eachline(file)
        startswith(line, "REF") && return header
        words = split(line)
        header[words[1]] = words[2:end]
    end
    return header
end

end # module
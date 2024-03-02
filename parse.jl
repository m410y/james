using PyCall
using Rotations

fabio = pyimport("fabio")

function sfrm_image(filenames::Array{String})::Matrix{Int32}
    image = fabio.open(filenames[1]).data[end:-1:1, :]

    for filename in filenames[2:1:end]
        image += fabio.open(filename).data[end:-1:1, :]
    end

    return image
end

function sfrm_meta(filename::String)::Dict
    meta = Dict{String, Float64}()
    header = fabio.openheader(filename).header

    angles = deg2rad.(parse.(Float64, split(header["ANGLES"])))
    meta["tth"] = angles[1]
    meta["omega"] = angles[2] + 0.5*deg2rad(parse(Float64, split(header["RANGE"])[1]))
    meta["phi"] = angles[3]
    meta["chi"] = angles[4]
    meta["d"] = 10*parse(Float64, split(header["DISTANC"])[2])
    meta["x_max"] = parse(Float64, split(header["NCOLS"])[1])
    meta["y_max"] = parse(Float64, split(header["NROWS"])[1])
    meta["xc"] = parse(Float64, split(header["CENTER"])[1])
    meta["yc"] = parse(Float64, split(header["CENTER"])[2])

    return meta
end

function p4p_orient(filename::String)::Tuple{Float64, RotMatrix3{Float64}}
    a = 0.0
    ort = zeros(3, 3)
    row = 1
    for line in eachline(open(filename))
        if startswith(line, "CELL ")
           a = parse(Float64, split(line)[2]) 
        elseif startswith(line, "ORT")
            ort[row, :] += parse.(Float64, split(line)[2:end])
            row += 1
            row > 3 && break
        end
    end

    return (a, nearest_rotation(ort))
end
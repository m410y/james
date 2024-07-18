module ParseSFRM

const LINE_LEN = 80
const BLOCK_SIZE = 512
const BLOCKS_MIN = 5
const KEY_LEN = 8
const DATA_ALIGNMENT = 16
# TODO: probably there is better way to do get DataType
const SIGNED_DTYPE = IdDict(1 => Int8, 2 => Int16, 4 => Int32)
const UNSIGNED_DTYPE = IdDict(1 => UInt8, 2 => UInt16, 4 => UInt32)

function read_header_blocks!(file::IOStream, header::Dict{String, Vector{SubString}}, blocks_num::Integer)::Nothing
    chunk_size = BLOCK_SIZE * blocks_num
    header_string = Vector{UInt8}(undef, chunk_size)
    read!(file, header_string)
    for line_start in 1:LINE_LEN:chunk_size
        line = String(header_string[line_start:line_start + LINE_LEN - 1])
        key = rstrip(line[1:KEY_LEN - 1])
        val = split(line[KEY_LEN + 1:end])
        haskey(header, key) || (header[key] = Vector{SubString}())
        append!(header[key], val)
    end
end

function read_typed_array(file::IOStream, size::Integer, length::Integer; unsigned=false)::Union{Vector{Integer}, Nothing}
    (size > 0 && length > 0) || return nothing
    array_type = unsigned ? UNSIGNED_DTYPE[size] : SIGNED_DTYPE[size]
    read_len = ((length * size + DATA_ALIGNMENT - 1) ÷ DATA_ALIGNMENT) * DATA_ALIGNMENT
    data = Vector{array_type}(undef, read_len ÷ size)
    read!(file, data)
    return data[1:length]
end

function data_merge_overflow!(data::AbstractArray, overflow::Union{AbstractArray, Nothing}, pivot::Integer)::Nothing
    isnothing(overflow) && return
    data[findall(data .== pivot)] = overflow
    return
end

function read_sfrm_header(file::IOStream)::Dict{String, Vector{SubString}}
    header = Dict{String, Vector{SubString}}()
    read_header_blocks!(file, header, BLOCKS_MIN)
    blocks_remain = parse(Int, header["HDRBLKS"][1]) - BLOCKS_MIN
    read_header_blocks!(file, header, blocks_remain)
    return header
end

function read_sfrm(file::IOStream)::Tuple{Dict{String, Vector{SubString}}, Matrix{Integer}}
    header = read_sfrm_header(file)
    data_bpp, under_bpp = parse.(Int, header["NPIXELB"][1:2])
    rows = parse(Int, header["NROWS"][1])
    cols = parse(Int, header["NCOLS"][1])
    under_len, over1_len, over2_len = parse.(Int, header["NOVERFL"][1:3])
    baseline = under_bpp > 0 ? parse(Int, header["NEXP"][3]) : 0
    data = read_typed_array(file, data_bpp, rows * cols, unsigned=true)
    under = read_typed_array(file, under_bpp, under_len, unsigned=false)
    over1 = read_typed_array(file, 2, over1_len, unsigned=true)
    over2 = read_typed_array(file, 4, over2_len, unsigned=false)
    data = Int32.(data)
    data_merge_overflow!(data, under, 0)
    data_merge_overflow!(data, over1, typemax(UInt8))
    data_merge_overflow!(data, over2, typemax(UInt16))
    data .+= baseline
    image = transpose(reshape(data, (cols, rows))[begin:end, end:-1:begin])
    return header, image
end

end # module
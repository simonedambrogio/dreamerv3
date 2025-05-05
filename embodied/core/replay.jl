using Random
using UUIDs
using DataStructures: Deque # Efficient double-ended queue
using StatsBase

# --- Chunk Struct ---
# Holds a fixed-size sequence of transitions.
mutable struct Chunk
    uuid::UUID                         # Unique identifier for the chunk
    data::Dict{Symbol, AbstractArray}  # Stores arrays for :observation, :action, etc.
    length::Int                        # Current number of steps stored
    max_size::Int                      # Capacity (chunksize)
    succ::Union{UUID, Nothing}         # UUID of the next chunk in the sequence
    lock::ReentrantLock                # Add a lock for thread-safe append!

    # Constructor for an empty chunk
    function Chunk(chunksize::Int)
        new(uuid4(), Dict{Symbol, AbstractArray}(), 0, chunksize, nothing, ReentrantLock())
    end
end

# Append a single step (dictionary) to the chunk
function append!(chunk::Chunk, step::Dict{Symbol, Any}, verbose::Bool=false)
    lock(chunk.lock) do # Assuming Chunk might need its own lock if modified outside Replay context
        if chunk.length == 0
            # Initialize data arrays based on the first step added
            for (key, value) in step
                # Determine array element type and shape correctly
                if value isa AbstractArray
                    # If the value is an array, store its elements
                    el_type = eltype(value)
                    val_shape = size(value)
                    array_shape = (val_shape..., chunk.max_size)
                    chunk.data[key] = similar(Array{el_type}, array_shape)
                     verbose && println("Initialized chunk data for key '$key' with eltype=$el_type and shape=$array_shape") # Debug
                else
                    # If the value is a scalar
                    el_type = typeof(value)
                    # Shape for scalars over time: (chunk.max_size,)
                    array_shape = (chunk.max_size,)
                    chunk.data[key] = similar(Array{el_type}, array_shape)
                    verbose && println("Initialized chunk data for key '$key' with eltype=$el_type and shape=$array_shape") # Debug
                end
            end
        elseif chunk.length >= chunk.max_size
            error("Chunk $(chunk.uuid) is full.")
        end

        # Append data for each key
        idx = chunk.length + 1
        for (key, value) in step
            if !haskey(chunk.data, key)
                # This part should ideally not be needed if all steps have consistent keys
                 @warn "Key $key from step not present in chunk during append. Initializing late."
                 # Initialize array for this new key
                 if value isa AbstractArray
                    el_type = eltype(value)
                    val_shape = size(value)
                    array_shape = (val_shape..., chunk.max_size)
                    chunk.data[key] = similar(Array{el_type}, array_shape)
                else
                    el_type = typeof(value)
                    array_shape = (chunk.max_size,)
                    chunk.data[key] = similar(Array{el_type}, array_shape)
                end
            end

            # Assign data
            try
                if value isa AbstractArray
                    # Assign array value to the time slice
                    dest_view = selectdim(chunk.data[key], ndims(chunk.data[key]), idx)
                    dest_view .= value
                else
                    # Assign scalar value
                    chunk.data[key][idx] = value
                end
            catch e
                verbose && println("Error assigning value for key $key at index $idx:")
                if value isa AbstractArray
                    dest_view = selectdim(chunk.data[key], ndims(chunk.data[key]), idx)
                    verbose && println("  Destination view size: $(size(dest_view))")
                    verbose && println("  Value size: $(size(value))")
                else
                     verbose && println("  Destination array slot type: $(eltype(chunk.data[key]))")
                end
                verbose && println("  Value type: $(typeof(value))")
                verbose && println("  Array type: $(typeof(chunk.data[key]))")
                rethrow(e)
            end
        end
        chunk.length += 1
    end # unlock
    return
end

# Extract a slice of data for all keys from a chunk
function slice(chunk::Chunk, start_idx::Int, len::Int)
    @assert start_idx >= 1 && start_idx <= chunk.length "Invalid start_idx $(start_idx) for chunk length $(chunk.length)"
    end_idx = start_idx + len - 1
    @assert end_idx <= chunk.length "Slice exceeds chunk bounds (start: $start_idx, len: $len, chunk len: $(chunk.length))"

    sliced_data = Dict{Symbol, AbstractArray}()
    for (key, array) in chunk.data
        # Select the time dimension (last dimension)
        sliced_data[key] = selectdim(array, ndims(array), start_idx:end_idx)
    end
    return sliced_data
end


# --- Replay Struct ---
mutable struct Replay
    length::Int                       # Sequence length for sampling
    capacity::Int                     # Max number of sequences (items)
    chunksize::Int                    # Steps per chunk

    # Core storage
    items::Dict{Int, Tuple{UUID, Int}} # Maps itemid -> (chunkid, start_index)
    chunks::Dict{UUID, Chunk}          # Maps chunkid -> Chunk object
    fifo::Deque{Int}                   # Stores itemid in insertion order (for removal)
    itemid_counter::Int                # Generates unique item IDs

    # Tracking current write position for workers/streams
    current_chunk::Dict{Int, UUID} # Maps worker_id -> chunkid
    current_index::Dict{Int, Int} # Maps worker_id -> index in current_chunk
    streams::Dict{Int, Deque{Tuple{UUID, Int}}} # step history per worker

    rng::AbstractRNG
    lock::ReentrantLock             # Thread safety

    # Constructor
    function Replay(;
        length::Int,
        capacity::Int,
        chunksize::Int = 1024,
        seed::Int = 0
    )
        @assert length > 0 "Sequence length must be positive"
        @assert capacity > 0 "Capacity must be positive"
        @assert chunksize >= length "Chunksize must be >= sequence length"

        new(
            length, capacity, chunksize,
            Dict{Int, Tuple{UUID, Int}}(), # items
            Dict{UUID, Chunk}(),          # chunks
            Deque{Int}(),                 # fifo
            0,                            # itemid_counter
            Dict{Int, UUID}(),            # current_chunk
            Dict{Int, Int}(),             # current_index
            Dict{Int, Deque{Tuple{UUID, Int}}}(), # streams
            MersenneTwister(seed),
            ReentrantLock()
        )
    end
end

Base.length(replay::Replay) = Base.length(replay.items) # Number of stored sequences

# --- Replay Methods ---

# Helper to finish a chunk and start a new one for a worker
function _complete_chunk(replay::Replay, chunk::Chunk, worker::Int)
    succ = Chunk(replay.chunksize)
    replay.chunks[succ.uuid] = succ
    replay.current_chunk[worker] = succ.uuid
    replay.current_index[worker] = 0
    chunk.succ = succ.uuid
    return succ
end

# Internal method to insert a new sequence starting point into items and fifo
function _insert!(replay::Replay, chunkid::UUID, index::Int)
    lock(replay.lock) do
        # Remove oldest if capacity is reached
        while length(replay.items) >= replay.capacity
            _remove!(replay)
        end

        # Add new item
        replay.itemid_counter += 1
        itemid = replay.itemid_counter
        replay.items[itemid] = (chunkid, index)
        push!(replay.fifo, itemid) # Add to the end of the queue
    end
end

# Internal method to remove the oldest sequence
function _remove!(replay::Replay)
    # Assumes lock is already held
    if isempty(replay.fifo)
        @warn "Attempted to remove from empty fifo/items."
        return
    end
    itemid = popfirst!(replay.fifo) # Remove from the front of the queue
    if haskey(replay.items, itemid)
        chunkid, index = replay.items[itemid]
        delete!(replay.items, itemid)
        # Note: In the Python version, there's reference counting for chunks.
        # For simplicity, we'll skip strict chunk deletion for now.
        # Proper deletion would require tracking references from `items` and `streams`.
    else
        @warn "Item ID $itemid from fifo not found in items dict during removal."
    end
end

# Add a single step transition to the replay buffer
function add!(replay::Replay, step::Dict{Symbol, Any}, worker::Int=0, verbose::Bool=false)
    # Ensure step contains necessary keys (optional check)
    # required_keys = [:observation, :action, :reward, :is_first, :is_last, :is_terminal]
    # @assert all(k -> haskey(step, k), required_keys) "Step dictionary is missing required keys."

    lock(replay.lock) do
        # Get or create the stream deque for the worker
        stream = get!(replay.streams, worker, Deque{Tuple{UUID, Int}}())

        # Get or create the current chunk for the worker
        if !haskey(replay.current_chunk, worker)
            chunk = Chunk(replay.chunksize)
            replay.chunks[chunk.uuid] = chunk
            replay.current_chunk[worker] = chunk.uuid
            replay.current_index[worker] = 0
            verbose && println("Worker $worker started new chunk $(chunk.uuid)")
        end

        # Get current chunk details
        chunkid = replay.current_chunk[worker]
        index = replay.current_index[worker]
        chunk = replay.chunks[chunkid]

        # Append step to the chunk
        append!(chunk, step)
        push!(stream, (chunkid, index)) # Store (chunkid, index_within_chunk)

        # Update current index
        index += 1
        replay.current_index[worker] = index

        # Complete chunk if full
        if index >= chunk.max_size
            _complete_chunk(replay, chunk, worker)
            verbose && println("Worker $worker completed chunk $(chunk.uuid), started new one $(replay.current_chunk[worker])")

        end

        # If stream is long enough, insert the oldest step as a potential sequence start
        if length(stream) >= replay.length
            oldest_chunkid, oldest_index = popfirst!(stream)
            # Indices in `items` refer to the *start* index of a valid sequence
            # So, the index stored is the one that was just popped from the front
            _insert!(replay, oldest_chunkid, oldest_index + 1) # +1 because indices are 1-based
        end
    end # unlock
end

# --- Sampling Functions ---

# Internal function to retrieve a complete sequence, handling chunk boundaries
function _getseq(replay::Replay, chunkid::UUID, start_index::Int, verbose::Bool=false)
    local chunk::Chunk
    try
        chunk = replay.chunks[chunkid]
    catch e
        @error "Chunk ID $chunkid not found in replay.chunks during _getseq. Item ID might point to a deleted chunk."
        rethrow(e)
    end

    @assert start_index >= 1 "start_index must be 1-based ($start_index)"
    # Correct calculation for available steps from start_index *inclusive*
    available = chunk.length - start_index + 1

    if available < 0
         error("Calculated negative available steps ($available) in chunk $(chunk.uuid). start_index: $start_index, chunk.length: $(chunk.length)")
    end

    # If the whole sequence fits in the current chunk
    if available >= replay.length
        # Directly slice the required length
        return slice(chunk, start_index, replay.length)
    else
        # Sequence spans multiple chunks
        parts = Dict{Symbol, Vector{AbstractArray}}() # Store parts per key
        keys_in_chunk = keys(chunk.data)

        # Get the first part from the starting chunk
        first_part_data = slice(chunk, start_index, available)
        for key in keys_in_chunk
            parts[key] = [first_part_data[key]]
        end

        remaining = replay.length - available
        current_chunk = chunk

        # Loop to get remaining parts from successor chunks
        while remaining > 0
            next_chunk_id = current_chunk.succ
            if isnothing(next_chunk_id)
                error("Sequence trace hit end of chunk chain (succ is nothing) in chunk $(current_chunk.uuid) while $(remaining) steps were still needed.")
            end
            try
                current_chunk = replay.chunks[next_chunk_id]
            catch e
                @error "Successor Chunk ID $next_chunk_id not found in replay.chunks during _getseq chain following."
                rethrow(e)
            end

            used = min(remaining, current_chunk.length)
            if used == 0 && remaining > 0
                error("Successor chunk $(current_chunk.uuid) has length 0, but $remaining steps are still needed.")
            end

            next_part_data = slice(current_chunk, 1, used) # Start from index 1 of the next chunk
            for key in keys_in_chunk # Assume keys are consistent
                 if !haskey(next_part_data, key)
                     @warn "Key $key missing in successor chunk $(current_chunk.uuid) during sequence assembly."
                     # Handle missing key, e.g., skip or fill with default? For now, error might be safer.
                     error("Key $key missing in successor chunk $(current_chunk.uuid).")
                 end
                 # Append the array slice to the list for this key
                 push!(parts[key], next_part_data[key])
            end

            remaining -= used
        end

        # Concatenate the parts for each key along the time dimension
        final_seq = Dict{Symbol, AbstractArray}()
        for (key, array_list) in parts
            # Determine the time dimension (last dimension for scalars, second-to-last for arrays)
            time_dim = ndims(array_list[1])
            try
                 final_seq[key] = cat(array_list...; dims=time_dim)
            catch e
                 verbose && println("Error concatenating parts for key $key:")
                 for (i, part) in enumerate(array_list)
                     verbose && println("  Part $i shape: $(size(part)), type: $(eltype(part))")
                 end
                 verbose && println("  Target dimension: $time_dim")
                 rethrow(e)
            end
        end

        return final_seq
    end
end

# Sample a batch of sequences
function StatsBase.sample(replay::Replay, batch_size::Int, verbose::Bool=false)
    lock(replay.lock) do # Lock for reading items safely
        if length(replay.items) == 0
            error("Cannot sample from empty replay buffer.")
        end
        if length(replay.items) < batch_size
             @warn "Sampling batch size $batch_size larger than number of available items $(length(replay.items)). Sampling with replacement implicitly."
             # Note: rand with size > population size samples with replacement by default
        end

        # Get available item IDs
        item_ids = collect(keys(replay.items))

        # Sample item IDs randomly
        sampled_ids = rand(replay.rng, item_ids, batch_size)

        # Retrieve sequences using _getseq
        sequences = Dict{Symbol, AbstractArray}[] # Store individual sequence dicts
        for itemid in sampled_ids
            chunkid, start_index = replay.items[itemid]
            try
                seq_data = _getseq(replay, chunkid, start_index)
                push!(sequences, seq_data)
            catch e
                 @error "Failed to retrieve sequence for itemid=$itemid (chunkid=$chunkid, start_index=$start_index)."
                 # Decide how to handle errors: skip this sample, retry, or rethrow?
                 # For now, rethrow to make errors visible.
                 rethrow(e)
            end
        end

        # Assemble the batch
        first_seq = sequences[1]
        batch_dict = Dict{Symbol, AbstractArray}()
        for key in keys(first_seq)
            # Get shape and type from the first sequence's data for this key
            first_data = first_seq[key]
            el_type = eltype(first_data)
            # Shape of data for one step (all dims except time)
            step_shape = size(first_data)[1:end-1]
            seq_len = size(first_data)[end] # Should be replay.length

            # Create batch array shape: (step_dims..., sequence_length, batch_size)
            batch_shape = (step_shape..., seq_len, batch_size)
            batch_array = similar(Array{el_type}, batch_shape)

            # Fill the batch array
            for i in 1:batch_size
                 # Select the slice for the i-th batch element
                 batch_slice = selectdim(batch_array, ndims(batch_array), i)
                 try
                     current_seq_data = sequences[i][key]
                     if size(batch_slice) == size(current_seq_data)
                        batch_slice .= current_seq_data
                     else
                         error("Shape mismatch during batch assembly for key '$key'. Batch slice: $(size(batch_slice)), Sequence data: $(size(current_seq_data))")
                     end
                 catch e
                      verbose && println("Error assembling batch for key $key at batch index $i:")
                      verbose && println("  Expected slice shape: $(size(batch_slice))")
                      if haskey(sequences[i], key)
                          verbose && println("  Actual sequence data shape: $(size(sequences[i][key]))")
                      else
                           verbose && println("  Key $key not found in sequence $i")
                      end
                      rethrow(e)
                 end
            end
            batch_dict[key] = batch_array
        end

        return batch_dict
    end # unlock
end

# verbose && println("Replay methods (add!, _insert!, _remove!, _complete_chunk) defined.")
# verbose && println("Replay sampling methods (_getseq, sample) defined.")

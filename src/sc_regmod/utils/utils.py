def detect_sequence_changes(data):
    seq_changes = []
    for i in range(data.get_alignment_length()):
        column_set = set(data[:, i])
        
        #This checks if the column has differences
        if(len(column_set) != 1):
            seq_changes.append(i)

    return seq_changes
    
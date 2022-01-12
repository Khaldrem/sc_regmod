
def find_position(s, c):
    out = []
    idx = s.find(c)
    while idx != -1:
        out.append(idx)
        idx = s.find(c, idx + 1)
    return out


# TODO chequear, entradas, finaliozar funcion
def get_column_index(data, column):
    freq = {}
    for c in set(data[:, column]):
        freq[c] = {}
        pos = find_position(data[:, column], c)
        freq[c]['count'] = len(pos)
        freq[c]['position'] = find_position(data[:, column], c)

    return freq


def create_index(data):
    index = []

    #Iterate over sequence length
    for i in range(data.get_alignment_length()):
        item = {}

        freq = get_column_index(data, i)
        item['index'] = i
        item['freq'] = freq
        index.append(item)

    return index

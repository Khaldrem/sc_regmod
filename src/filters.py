from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio.Align import MultipleSeqAlignment

from src.io import read_phylip_file
from src.utils import get_filename


def eliminate_columns_based_on_list(data, cols_to_eliminate):
    cols_to_eliminate = set(cols_to_eliminate)
    new_data = []
    for row in data:
        new_seq = "".join([char for idx, char in enumerate(row.seq) if idx not in cols_to_eliminate])
        new_data.append(SeqRecord(new_seq, id=row.id))
    
    new_data = MultipleSeqAlignment(new_data)
    return new_data


def detect_sequence_changes(data):
    """
        Permite crear una lista, en la cual se guardan los indices
        de las columnas del archivo, que presentan mas de 1 solo caracter
        en esa columna.
    """
    seq_changes = []
    for i in range(data.get_alignment_length()):
        column_set = set(data[:, i])
        
        #This checks if the column has differences
        if(len(column_set) != 1):
            seq_changes.append(i)

    return seq_changes


def create_reduced_sequence(data, row, changes):
    """
        Toma una secuencia (fila) y elimina las columnas
        que no estan presentes en la lista changes.
        Reduciendo asi la secuencia de las filas que presentaban solo 1 caracter.
    """
    seq = ""
    for i in changes:
        seq = seq + data[row, i]

    return seq


def create_compressed_alignment(data, changes):
    """
        Desde la lista changes, la cual contiene todas los indices de 
        las columnas que presentan al menos 1 base diferente, reduce los
        datos y crea una nueva secuencia para escribir
    """
    if(changes == []):
        #print("Can't compress this file, because there are no changes in it.")
        return []

    #Generate new MultipleSeqAlignment
    align = []
    for row in range(len(data)):
        seq = create_reduced_sequence(data, row, changes)
        align.append(SeqRecord(Seq(seq), id=data[row].id))
    
    return MultipleSeqAlignment(align)


def eliminate_rows(data, ids):
    """
        Elimina las filas que se presentan 
        en el arreglo de ids.
    """
    new_data = []

    for row in range(len(data)):
        if data[row].id not in ids:
            new_data.append(SeqRecord(data[row].seq, id=data[row].id))
    
    return MultipleSeqAlignment(new_data)


def clean_data(filepath, eliminated_rows):
    data = read_phylip_file(filepath)
    data = eliminate_rows(data, eliminated_rows)

    #Indices que presentan al menos 2 bases en su columna
    changes = detect_sequence_changes(data)

    #Si esta vacio significa que el archivo presenta en 
    #todas sus columnas una sola base
    if changes != []:
        #Elimina columnas que no presenten variacion en sus bases
        new_data = create_compressed_alignment(data, changes)
        return new_data, changes
    else:
        print(f"File: {get_filename(filepath)} todas sus columnas poseen un solo caracter.")
        return data, changes
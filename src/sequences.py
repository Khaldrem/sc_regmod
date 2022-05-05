from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio.Align import MultipleSeqAlignment


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
    if(changes == []):
        #print("Can't compress this file, because there are no changes in it.")
        return []

    #Generate new MultipleSeqAlignment
    align = []
    for row in range(len(data)):
        seq = create_reduced_sequence(data, row, changes)
        align.append(SeqRecord(Seq(seq), id=data[row].id))
    
    return MultipleSeqAlignment(align)

from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio.Align import MultipleSeqAlignment

#TODO cheq entradas, doc
def create_reduced_sequence(data, row, changes):
    seq = ""
    for i in changes:
        seq = seq + data[row, i]

    return seq


#TODO cheq entradas, doc
def create_compressed_alignment(data, changes):
    if(changes == []):
        print("Can't compress this file, because there are no changes in it.")
        return []

    #Generate new MultipleSeqAlignment
    align = []
    for row in range(len(data)):
        seq = create_reduced_sequence(data, row, changes)
        align.append(SeqRecord(Seq(seq), id=data[row].id))
    
    return MultipleSeqAlignment(align)
